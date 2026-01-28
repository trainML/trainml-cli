import os
import re
import asyncio
import aiohttp
import aiofiles
import hashlib
import logging
from aiohttp.client_exceptions import (
    ClientResponseError,
    ClientConnectorError,
    ServerTimeoutError,
    ServerDisconnectedError,
    ClientOSError,
    ClientPayloadError,
    InvalidURL,
)
from trainml.exceptions import ConnectionError, TrainMLException

MAX_RETRIES = 5
RETRY_BACKOFF = 2  # Exponential backoff base (2^attempt)
PARALLEL_UPLOADS = 10  # Max concurrent uploads
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB
RETRY_STATUSES = {
    502,
    503,
    504,
}  # Server errors to retry during upload/download
# Additional retries for DNS/connection errors (ClientConnectorError)
DNS_MAX_RETRIES = 7  # More retries for DNS resolution issues
DNS_INITIAL_DELAY = 1  # Initial delay in seconds before first DNS retry


def normalize_endpoint(endpoint):
    """
    Normalize endpoint URL to ensure it has a protocol.

    Args:
        endpoint: Endpoint URL (with or without protocol)

    Returns:
        Normalized endpoint URL with https:// protocol

    Raises:
        ValueError: If endpoint is empty
    """
    if not endpoint:
        raise ValueError("Endpoint URL cannot be empty")

    # Remove trailing slashes
    endpoint = endpoint.rstrip("/")

    # Add https:// if no protocol is specified
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"

    return endpoint


async def ping_endpoint(
    endpoint, auth_token, max_retries=MAX_RETRIES, retry_backoff=RETRY_BACKOFF
):
    """
    Ping the endpoint to ensure it's ready before upload/download operations.

    Retries on all errors (404, 500, DNS errors, etc.) with exponential backoff
    until a 200 response is received. This handles startup timing issues.

    Creates a fresh TCPConnector for each attempt to force fresh DNS resolution
    and avoid stale DNS cache issues.

    Args:
        endpoint: Server endpoint URL
        auth_token: Authentication token
        max_retries: Maximum number of retry attempts
        retry_backoff: Exponential backoff base

    Raises:
        ConnectionError: If ping never returns 200 after max retries
        ClientConnectorError: If DNS/connection errors persist after max retries
    """
    endpoint = normalize_endpoint(endpoint)
    attempt = 1
    effective_max_retries = max_retries

    while attempt <= effective_max_retries:
        # Create a fresh connector for each attempt to force DNS re-resolution
        # This helps avoid stale DNS cache issues
        connector = None
        try:
            connector = aiohttp.TCPConnector(limit=1, limit_per_host=1)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"{endpoint}/ping",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=30,
                ) as response:
                    if response.status == 200:
                        logging.debug(
                            f"Endpoint {endpoint} is ready (ping successful)"
                        )
                        return
                    # For any non-200 status, retry
                    text = await response.text()
                    raise ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=text,
                    )
        except ClientResponseError as e:
            # Retry on any HTTP error status
            if attempt < effective_max_retries:
                logging.debug(
                    f"Ping attempt {attempt}/{effective_max_retries} failed with status {e.status}: {str(e)}"
                )
                await asyncio.sleep(retry_backoff**attempt)
                attempt += 1
                continue
            raise ConnectionError(
                f"Endpoint {endpoint} ping failed after {effective_max_retries} attempts. "
                f"Last error: HTTP {e.status} - {str(e)}"
            )
        except ClientConnectorError as e:
            # DNS resolution errors need more retries and initial delay
            if effective_max_retries == max_retries:
                effective_max_retries = max(max_retries, DNS_MAX_RETRIES)

            if attempt < effective_max_retries:
                # Use initial delay for first retry, then exponential backoff
                if attempt == 1:
                    delay = DNS_INITIAL_DELAY
                else:
                    delay = retry_backoff ** (attempt - 1)
                logging.debug(
                    f"Ping attempt {attempt}/{effective_max_retries} failed due to DNS/connection error: {str(e)}"
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            raise ConnectionError(
                f"Endpoint {endpoint} ping failed after {effective_max_retries} attempts due to DNS/connection error: {str(e)}"
            )
        except (
            ServerDisconnectedError,
            ClientOSError,
            ServerTimeoutError,
            ClientPayloadError,
            asyncio.TimeoutError,
        ) as e:
            if attempt < effective_max_retries:
                logging.debug(
                    f"Ping attempt {attempt}/{effective_max_retries} failed: {str(e)}"
                )
                await asyncio.sleep(retry_backoff**attempt)
                attempt += 1
                continue
            raise ConnectionError(
                f"Endpoint {endpoint} ping failed after {effective_max_retries} attempts: {str(e)}"
            )
        finally:
            # Ensure connector is closed to free resources and clear DNS cache
            # This forces fresh DNS resolution on the next attempt
            if connector is not None:
                try:
                    await connector.close()
                except Exception:
                    # Ignore errors during cleanup
                    pass


async def retry_request(
    func, *args, max_retries=MAX_RETRIES, retry_backoff=RETRY_BACKOFF, **kwargs
):
    """
    Shared retry logic for network requests.

    For DNS/connection errors (ClientConnectorError), uses more retries and
    an initial delay to handle transient DNS resolution issues.
    """
    attempt = 1
    effective_max_retries = max_retries

    while attempt <= effective_max_retries:
        try:
            return await func(*args, **kwargs)
        except ClientResponseError as e:
            if e.status in RETRY_STATUSES and attempt < max_retries:
                logging.debug(
                    f"Retry {attempt}/{max_retries} due to {e.status}: {str(e)}"
                )
                await asyncio.sleep(retry_backoff**attempt)
                attempt += 1
                continue
            raise
        except ClientConnectorError as e:
            # DNS resolution errors need more retries and initial delay
            # Update effective_max_retries if this is the first DNS error
            if effective_max_retries == max_retries:
                effective_max_retries = max(max_retries, DNS_MAX_RETRIES)

            if attempt < effective_max_retries:
                # Use initial delay for first retry, then exponential backoff
                if attempt == 1:
                    delay = DNS_INITIAL_DELAY
                else:
                    delay = retry_backoff ** (attempt - 1)
                logging.debug(
                    f"Retry {attempt}/{effective_max_retries} due to DNS/connection error: {str(e)}"
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            raise
        except (
            ServerDisconnectedError,
            ClientOSError,
            ServerTimeoutError,
            ClientPayloadError,
            asyncio.TimeoutError,
        ) as e:
            if attempt < max_retries:
                logging.debug(f"Retry {attempt}/{max_retries} due to {str(e)}")
                await asyncio.sleep(retry_backoff**attempt)
                attempt += 1
                continue
            raise


async def upload_chunk(
    session,
    endpoint,
    auth_token,
    total_size,
    data,
    offset,
):
    """Uploads a single chunk with retry logic."""
    start = offset
    end = offset + len(data) - 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{total_size}",
        "Authorization": f"Bearer {auth_token}",
    }

    async def _upload():
        async with session.put(
            f"{endpoint}/upload",
            headers=headers,
            data=data,
            timeout=30,
        ) as response:
            if response.status == 200:
                await response.release()
                return response
            elif response.status in RETRY_STATUSES:
                text = await response.text()
                raise ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=text,
                )
            else:
                text = await response.text()
                raise ConnectionError(
                    f"Chunk {start}-{end} failed with status {response.status}: {text}"
                )

    await retry_request(_upload)


async def upload(endpoint, auth_token, path):
    """
    Upload a local file or directory as a TAR stream to the server.

    Args:
        endpoint: Server endpoint URL
        auth_token: Authentication token
        path: Local file or directory path to upload

    Raises:
        ValueError: If path doesn't exist or is invalid
        ConnectionError: If upload fails or endpoint ping fails
        TrainMLException: For other errors
    """
    # Normalize endpoint URL to ensure it has a protocol
    endpoint = normalize_endpoint(endpoint)

    # Ping endpoint to ensure it's ready before starting upload
    await ping_endpoint(endpoint, auth_token)

    # Expand user home directory (~) in path
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise ValueError(f"Path not found: {path}")

    # Determine if it's a file or directory and build tar command accordingly
    abs_path = os.path.abspath(path)

    if os.path.isfile(path):
        # For a single file, create a tar with just that file
        file_name = os.path.basename(abs_path)
        parent_dir = os.path.dirname(abs_path)
        # Use tar -c to create archive with single file, stream to stdout
        # -C changes to parent directory so the file appears at root of tar
        command = ["tar", "-c", "-C", parent_dir, file_name]
        desc = f"Uploading file {file_name}"
    elif os.path.isdir(path):
        # For a directory, archive its contents at the root of the tar file
        # -C changes to the directory itself, and . archives all contents
        command = ["tar", "-c", "-C", abs_path, "."]
        dir_name = os.path.basename(abs_path)
        desc = f"Uploading directory {dir_name}"
    else:
        raise ValueError(f"Path is neither a file nor directory: {path}")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # We need to know the total size for progress, but tar doesn't tell us
    # So we'll estimate or track bytes uploaded
    sha512 = hashlib.sha512()
    offset = 0
    semaphore = asyncio.Semaphore(PARALLEL_UPLOADS)

    async with aiohttp.ClientSession() as session:
        async with semaphore:
            while True:
                chunk = await process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    break  # End of stream

                sha512.update(chunk)
                chunk_offset = offset
                offset += len(chunk)

                # For total_size, we'll use a large number since we don't know the actual size
                # The server will handle the Content-Range correctly
                await upload_chunk(
                    session,
                    endpoint,
                    auth_token,
                    offset,  # Use current offset as total (will be updated)
                    chunk,
                    chunk_offset,
                )

        # Wait for process to finish
        await process.wait()
        if process.returncode != 0:
            stderr = await process.stderr.read()
            raise TrainMLException(
                f"tar command failed: {stderr.decode() if stderr else 'Unknown error'}"
            )

        # Finalize upload
        file_hash = sha512.hexdigest()

        async def _finalize():
            async with session.post(
                f"{endpoint}/finalize",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={"hash": file_hash},
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ConnectionError(f"Finalize failed: {text}")
                return await response.json()

        data = await retry_request(_finalize)
        logging.debug(f"Upload finalized: {data}")


async def download(endpoint, auth_token, target_directory, file_name=None):
    """
    Download a directory archive from the server and extract it.

    Args:
        endpoint: Server endpoint URL
        auth_token: Authentication token
        target_directory: Directory to extract files to (or save zip file)
        file_name: Optional filename override for zip archive (if ARCHIVE=true).
                   If not provided, filename is extracted from Content-Disposition header.

    Raises:
        ConnectionError: If download fails or endpoint ping fails
        TrainMLException: For other errors
    """
    # Normalize endpoint URL to ensure it has a protocol
    endpoint = normalize_endpoint(endpoint)

    # Ping endpoint to ensure it's ready before starting download
    await ping_endpoint(endpoint, auth_token)

    # Expand user home directory (~) in target_directory
    target_directory = os.path.expanduser(target_directory)

    if not os.path.isdir(target_directory):
        os.makedirs(target_directory, exist_ok=True)

    # First, check server info to see if ARCHIVE is set
    # If /info endpoint is not available, default to False (TAR stream mode)
    async with aiohttp.ClientSession() as session:
        use_archive = False
        try:

            async def _get_info():
                async with session.get(
                    f"{endpoint}/info",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=30,
                ) as response:
                    if response.status != 200:
                        try:
                            error_text = await response.text()
                        except Exception:
                            error_text = f"Unable to read response body (status: {response.status})"
                        raise ConnectionError(
                            f"Failed to get server info (status {response.status}): {error_text}"
                        )
                    return await response.json()

            info = await retry_request(_get_info)
            use_archive = info.get("archive", False)
        except InvalidURL as e:
            raise ConnectionError(
                f"Invalid endpoint URL: {endpoint}. "
                f"Please ensure the URL includes a protocol (http:// or https://). "
                f"Error: {str(e)}"
            )
        except (ConnectionError, ClientResponseError) as e:
            # If /info endpoint is not available (404) or other error,
            # default to TAR stream mode and continue
            if isinstance(e, ConnectionError) and "404" in str(e):
                logging.debug(
                    "Warning: /info endpoint not available, defaulting to TAR stream mode"
                )
            elif isinstance(e, ClientResponseError) and e.status == 404:
                logging.debug(
                    "Warning: /info endpoint not available, defaulting to TAR stream mode"
                )
            else:
                # For other errors, re-raise
                raise

        # Download the archive
        # Note: Do NOT use "async with session.get() as response" - exiting the
        # context manager would release/close the connection before we read the
        # body. We need the response (and connection) to stay open for streaming.
        async def _download():
            response = await session.get(
                f"{endpoint}/download",
                headers={"Authorization": f"Bearer {auth_token}"},
                timeout=None,  # No timeout for large downloads
            )
            if response.status != 200:
                text = await response.text()
                response.close()
                # Raise ClientResponseError for non-200 status
                # Note: 404 and other errors should be rare now since ping_endpoint ensures readiness
                raise ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=(
                        text
                        if text
                        else f"Download endpoint returned status {response.status}"
                    ),
                )
            return response

        response = await retry_request(_download)

        # Check Content-Type header as fallback to determine if it's a zip file
        content_type = response.headers.get("Content-Type", "").lower()
        content_length = response.headers.get("Content-Length")
        if "zip" in content_type and not use_archive:
            logging.debug(
                "Warning: Server returned zip content but /info indicated TAR mode. Using zip mode."
            )
            use_archive = True

        # Debug: Log response info
        if content_length:
            logging.debug(f"Response Content-Length: {content_length} bytes")
        logging.debug(f"Response Content-Type: {content_type}")

        try:
            if use_archive:
                # Save as ZIP file
                # Extract filename from Content-Disposition header if not provided
                if file_name is None:
                    content_disposition = response.headers.get(
                        "Content-Disposition", ""
                    )
                    # Parse filename from Content-Disposition: attachment; filename="filename.zip"
                    if "filename=" in content_disposition:
                        # Extract filename from quotes
                        match = re.search(
                            r'filename="?([^"]+)"?', content_disposition
                        )
                        if match:
                            file_name = match.group(1)
                        else:
                            # Fallback: try without quotes
                            match = re.search(
                                r"filename=([^;]+)", content_disposition
                            )
                            if match:
                                file_name = match.group(1).strip()

                    # Fallback if no filename in header
                    if file_name is None:
                        file_name = "archive.zip"

                # Ensure .zip extension
                if not file_name.endswith(".zip"):
                    file_name = file_name + ".zip"

                output_path = os.path.join(target_directory, file_name)

                total_bytes = 0
                async with aiofiles.open(output_path, "wb") as f:
                    # Stream the response content in chunks
                    async for chunk in response.content.iter_chunked(
                        CHUNK_SIZE
                    ):
                        await f.write(chunk)
                        total_bytes += len(chunk)

                if total_bytes == 0:
                    raise ConnectionError(
                        "Downloaded file is empty (0 bytes). "
                        "The server may not have any files to download, or there was an error streaming the response."
                    )

                logging.info(
                    f"Archive saved to: {output_path} ({total_bytes} bytes)"
                )
            else:
                # Extract TAR stream directly
                # Create tar extraction process
                command = ["tar", "-x", "-C", target_directory]

                extract_process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Stream response to tar process
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    extract_process.stdin.write(chunk)
                    await extract_process.stdin.drain()

                extract_process.stdin.close()
                await extract_process.wait()

                if extract_process.returncode != 0:
                    stderr = await extract_process.stderr.read()
                    raise TrainMLException(
                        f"tar extraction failed: {stderr.decode() if stderr else 'Unknown error'}"
                    )

                logging.info(f"Files extracted to: {target_directory}")
        finally:
            response.close()

        # Finalize download
        async def _finalize():
            async with session.post(
                f"{endpoint}/finalize",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={},
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ConnectionError(f"Finalize failed: {text}")
                return await response.json()

        data = await retry_request(_finalize)
        logging.debug(f"Download finalized: {data}")
