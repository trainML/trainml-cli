import json
import os
import re
import sys
import math
import time
import socket
import asyncio
import importlib.util
import aiohttp
import aiofiles
import hashlib
import logging
import uuid
from aiohttp.client_exceptions import (
    ClientResponseError,
    ClientConnectorError,
    ClientConnectorDNSError,
    ServerTimeoutError,
    ServerDisconnectedError,
    ClientOSError,
    ClientPayloadError,
    InvalidURL,
)
from aiohttp.resolver import AsyncResolver, DefaultResolver
from trainml.exceptions import ConnectionError as TrainMLConnectionError
from trainml.exceptions import TrainMLException

MAX_RETRIES = 5
RETRY_BACKOFF = 2  # Exponential backoff base (2^attempt)
PARALLEL_UPLOADS = 10  # Max concurrent uploads
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB
RETRY_STATUSES = {
    502,
    503,
    504,
    520,  # Cloudflare: Web server returned an unknown error
    521,  # Cloudflare: Web server is down
    522,  # Cloudflare: Connection timed out
    524,  # Cloudflare: A connection was made to the origin, but the origin did not return a complete HTTP response in time
}  # Server errors to retry during upload/download
# Additional retries for DNS/connection errors (ClientConnectorError)
DNS_MAX_RETRIES = 7  # More retries for DNS resolution issues
DNS_INITIAL_DELAY = 1  # Initial delay in seconds before first DNS retry
# Resolver mode for transfer sessions: the client's default OS resolver
# (honors enterprise DNS policy) until a name-not-found error is seen,
# then a dedicated c-ares channel per attempt against the system-configured
# nameservers (bypasses the local stub resolver's negative cache).
RESOLVER_MODE_DEFAULT = "default"
RESOLVER_MODE_CARES = "cares"
# Cadence between name-not-found (NXDOMAIN/NODATA) ping retries. Resolvers
# cache negative DNS answers, so we wait a fixed interval between retries
# rather than using exponential backoff.
DNS_NAME_ERROR_RETRY_DELAY = 60
# While a Cloudflare tunnel is recycling, the hostname can be unmapped and the
# edge returns 404 for a while. Retry requests at this fixed cadence to ride
# out the window.
NOT_FOUND_RETRY_DELAY = 15  # seconds to wait between 404 retries
NOT_FOUND_MAX_RETRIES = 9
# Ping warmup timeout: calculate retries so last retry is this many seconds after first try
PING_WARMUP_TIMEOUT = 8 * 60  # 8 minutes in seconds
PROGRESS_THROTTLE_SEC = 0.3  # Min interval between progress bar updates


def _format_size(n):
    """
    Format byte count as human-readable string (e.g. '1.2 MB', '500 B').

    Uses 1024 for KB/MB/GB.
    """
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


def _write_progress(
    current,
    total=None,
    desc="Uploading",
    last=False,
    show_progress=True,
):
    """
    Write a single-line progress update to stdout (when TTY and show_progress).

    Uses \\r to overwrite the line. If last is True, prints a newline after.
    """
    if not (show_progress and sys.stdout.isatty()):
        if last:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return
    if total is not None and total > 0:
        pct = min(100, int(100 * current / total))
        bar_width = 30
        filled = int(bar_width * current / total) if total else 0
        filled = min(filled, bar_width)
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        line = f"{desc}: {bar} {pct}% {_format_size(current)} / {_format_size(total)}"
    else:
        line = f"{desc}: {_format_size(current)}"
    sys.stdout.write("\r" + line)
    sys.stdout.flush()
    if last:
        sys.stdout.write("\n")
        sys.stdout.flush()


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


def calculate_ping_retries(timeout_seconds, backoff_base):
    """
    Calculate the number of retries needed for ping warmup to reach timeout.

    With exponential backoff, if we have n retries, the total wait time is:
    backoff_base^1 + backoff_base^2 + ... + backoff_base^(n-1) = (backoff_base^n - backoff_base) / (backoff_base - 1)

    For backoff_base = 2, this simplifies to: 2^n - 2

    Args:
        timeout_seconds: Total timeout in seconds
        backoff_base: Exponential backoff base

    Returns:
        Number of retries needed to reach or exceed the timeout
    """
    if backoff_base == 2:
        # Simplified calculation for base 2
        # 2^n - 2 >= timeout_seconds
        # 2^n >= timeout_seconds + 2
        # n >= log2(timeout_seconds + 2)
        n = math.ceil(math.log2(timeout_seconds + 2))
    else:
        # General case: (backoff_base^n - backoff_base) / (backoff_base - 1) >= timeout_seconds
        # backoff_base^n >= timeout_seconds * (backoff_base - 1) + backoff_base
        # n >= log_base(timeout_seconds * (backoff_base - 1) + backoff_base)
        target = timeout_seconds * (backoff_base - 1) + backoff_base
        n = math.ceil(math.log(target, backoff_base))

    return max(1, int(n))


def _aiodns_available():
    """
    Return True if aiodns (c-ares) is installed.

    aiodns is optional: when missing, transfers fall back to the default
    OS resolver (aiohttp[speedups] normally provides aiodns).
    """
    return importlib.util.find_spec("aiodns") is not None


def _is_dns_name_not_known(error):
    """
    Return True if a connector error is NXDOMAIN / name not known.

    Resolvers cache negative DNS answers (NXDOMAIN and NODATA) for the
    zone's SOA minimum TTL — for the service zone that is 1800s. Local
    stub resolvers (mDNSResponder, nss) add their own cache on top, so
    repeated getaddrinfo calls keep returning the cached failure even
    after the record exists. These failures need a long, fixed-cadence
    backoff plus a resolver that bypasses the local stub cache.

    Args:
        error: ClientConnectorError (or subclass) from aiohttp

    Returns:
        True if the error indicates the hostname does not exist
    """
    if isinstance(error, ClientConnectorDNSError):
        return True
    name_not_known_errnos = {socket.EAI_NONAME}
    eai_nodata = getattr(socket, "EAI_NODATA", None)
    if eai_nodata is not None:
        name_not_known_errnos.add(eai_nodata)

    os_error = getattr(error, "os_error", None)
    nested_error = getattr(error, "error", None)
    for candidate in (os_error, nested_error):
        if candidate is not None:
            errno = getattr(candidate, "errno", None)
            if errno in name_not_known_errnos:
                return True
    message = str(error).lower()
    return (
        "not known" in message
        or "no address associated" in message
        or "name or service not known" in message
        or "nodename nor servname" in message
        or "getaddrinfo failed" in message
        or "dns lookup failed" in message
        or "host not found" in message
        or "name does not resolve" in message
    )


def _ping_connector_retry_delay(error, attempt, retry_backoff):
    """Return sleep seconds before retrying a ping connector failure."""
    if isinstance(error, ClientConnectorDNSError) or _is_dns_name_not_known(error):
        return DNS_NAME_ERROR_RETRY_DELAY
    if attempt == 1:
        return DNS_INITIAL_DELAY
    return retry_backoff ** (attempt - 1)


def _create_ping_connector(mode=RESOLVER_MODE_DEFAULT):
    """
    Build the connector for one transfer/ping session.

    mode=RESOLVER_MODE_DEFAULT uses aiohttp's default OS resolver
    (ThreadedResolver / getaddrinfo), honoring enterprise DNS policy.
    mode=RESOLVER_MODE_CARES uses a dedicated aiodns (c-ares) channel
    against the system-configured nameservers. The loop keyword is
    passed deliberately: any keyword forces aiohttp to create a
    dedicated aiodns channel instead of the process-wide shared one,
    whose internal negative cache would defeat freshness. Each call
    must close both returned objects.

    Returns:
        Tuple of (connector, resolver_or_None).
    """
    if mode == RESOLVER_MODE_CARES:
        if not _aiodns_available():
            logging.warning(
                "aiodns is not installed; cannot bypass the local DNS "
                "negative cache. Retrying with the system resolver."
            )
            mode = RESOLVER_MODE_DEFAULT
    if mode == RESOLVER_MODE_CARES:
        resolver = AsyncResolver(loop=asyncio.get_running_loop())
        connector = aiohttp.TCPConnector(
            limit=1,
            limit_per_host=1,
            resolver=resolver,
            use_dns_cache=False,
        )
        return connector, resolver
    connector = aiohttp.TCPConnector(limit=1, limit_per_host=1)
    return connector, None


def _resolver_host_from_error(error):
    """Best-effort FQDN extraction from a connector error.

    Prefers the connection key (always the exact FQDN the transport was
    trying to reach), falling back to the formatted message.
    """
    connection_key = getattr(error, "connection_key", None)
    host = getattr(connection_key, "host", None)
    if isinstance(host, str) and host:
        return host
    message = str(error)
    if isinstance(error, ClientConnectorError) and len(error.args) > 1:
        nested = error.args[1]
        nested_host = getattr(getattr(nested, "connection_key", None), "host", None)
        nested_os_error = getattr(nested, "os_error", None)
        for candidate in (nested_host, nested_os_error, nested):
            if isinstance(candidate, str) and "connect to host" in candidate:
                message = candidate
                break
    match = re.search(r"connect to host\s+([\w.-]+)(?::\d+)?", message)
    return match.group(1) if match else None


class _TransferSessionState:
    """
    Owns one aiohttp.ClientSession and its connector/resolver pair.

    open() closes any prior session first and creates a new one with the
    given resolver mode, so a transfer can escalate from the default OS
    resolver to a dedicated c-ares channel (system nameservers) without
    touching the stream-offset bookkeeping.
    """

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.session = None
        self._connector = None
        self._resolver = None

    async def open(self, resolver_mode=RESOLVER_MODE_DEFAULT):
        await self.close()
        connector, resolver = _create_ping_connector(resolver_mode)
        self._connector = connector
        self._resolver = resolver
        self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self.session

    async def close(self):
        if self.session is not None:
            try:
                await self.session.close()
            except OSError:
                pass
            self.session = None
        if self._connector is not None:
            try:
                await self._connector.close()
            except OSError:
                pass
            self._connector = None
        if self._resolver is not None:
            try:
                await self._resolver.close()
            except OSError:
                pass
            self._resolver = None


# c-ares DNS record types (RFC 1035)
_RTYPE_CNAME = 5
_RTYPE_SOA = 6


def _record_rtype(record):
    """Best-effort integer rtype from a c-ares DNSRecord (or dict)."""
    if isinstance(record, dict):
        return record.get("type", record.get("rtype"))
    rtype = getattr(record, "rtype", None)
    if rtype is None:
        rtype = getattr(record, "type", None)
    return rtype


def _soa_minimum_from_dns_result(dns_result):
    """Read the SOA minimum field from a raw c-ares DNS reply, if present.

    The zone SOA may appear in the answer section (for an SOA-type query)
    or the authority section (for negative replies). The minimum lives on
    the record's data payload (SOARecordData.minimum).
    """
    sections = []
    for section in ("answer", "answers", "authority", "additional"):
        sections.extend(getattr(dns_result, section, None) or [])
    for record in sections:
        if _record_rtype(record) != _RTYPE_SOA:
            continue
        data = getattr(record, "data", record)
        minimum = getattr(data, "minimum", None)
        if minimum is None and isinstance(record, dict):
            minimum = record.get("minimum")
        if isinstance(minimum, int) and minimum > 0:
            return minimum
    return None


def _cnames_from_dns_result(dns_result):
    """Collect CNAME target names from a raw c-ares DNS reply."""
    cnames = []
    sections = []
    for section in ("answer", "answers", "authority", "additional"):
        sections.extend(getattr(dns_result, section, None) or [])
    for record in sections:
        if _record_rtype(record) != _RTYPE_CNAME:
            continue
        data = getattr(record, "data", record)
        target = getattr(data, "cname", None)
        if target is None and isinstance(data, dict):
            target = data.get("cname")
        if isinstance(target, bytes):
            target = target.decode("ascii", "replace")
        if target:
            cnames.append(str(target))
    return cnames


def _zone_candidates(host):
    """Return plausible zone apexes for a host, longest first.

    The immediate parent label is always included. Generic public TLD
    parents of ccTLDs/sld names (com, net, org, ...) are dropped —
    they are not the service zone and would at best add useless
    probe queries.
    """
    host = (host or "").rstrip(".")
    parts = host.split(".")
    candidates = [".".join(parts[i:]) for i in range(1, len(parts)) if parts[i:]]
    if len(parts) > 2:
        tail2 = parts[-2]
        if not (len(tail2) == 2 and tail2.isalpha() and parts[-1].isalpha()):
            candidates = candidates[:-1]
    return candidates


async def _run_query_dns(aiodns_resolver, host, qtype, timeout=30):
    """Run a raw c-ares query via aiodns, returning (result, error)."""
    try:
        task = asyncio.ensure_future(
            aiodns_resolver.query_dns(host, qtype, qclass="IN")
        )
        return (await asyncio.wait_for(task, timeout=timeout)), None
    except Exception as e:  # noqa: BLE001 - probe is best-effort
        return None, e


async def _probe_soa_minimum(host, aiodns_resolver):
    """
    Best-effort read of the zone's SOA minimum (negative-cache TTL).

    Queries candidate zone apexes (the host and each parent label) with
    type SOA until one yields a zone SOA. Apexes that exist return their
    SOA as the answer; apexes that do not exist are answered negatively
    with the true zone SOA. Any failure returns None and the caller falls
    back to a default wait.
    """
    last_error = None
    for apex in _zone_candidates(host):
        dns_result, error = await _run_query_dns(aiodns_resolver, apex, "SOA")
        if error is not None:
            last_error = error
            continue
        minimum = _soa_minimum_from_dns_result(dns_result)
        if minimum is not None:
            logging.info(
                "SOA minimum (negative-cache TTL) for %s (zone %s): %ss",
                host,
                apex,
                minimum,
            )
            return minimum
    if last_error is not None:
        logging.debug("SOA minimum probe for %s failed: %s", host, last_error)
    return None


async def _new_aiodns_resolver():
    """Create a dedicated aiodns (c-ares) channel on the system nameservers."""
    import aiodns

    kwargs = {}
    if hasattr(aiodns.DNSResolver, "query_dns"):
        kwargs["loop"] = asyncio.get_running_loop()
    return aiodns.DNSResolver(**kwargs)


async def _negative_cache_diagnostics(host, resolver, mode):
    """
    Build diagnostic metadata for a name-not-found failure.

    Creates a fresh dedicated c-ares channel (system nameservers) for the
    CNAME-chain and zone-SOA probes, so the probes themselves cannot be
    answered from a stale local cache. Purely informational: the retry
    cadence is a fixed DNS_NAME_ERROR_RETRY_DELAY regardless of the
    measured TTLs.

    Returns:
        Diagnostics dict (resolver_mode, aiodns_available, soa_minimum,
        cnames, host, and probe_error when a probe fails).
    """
    diagnostics = {
        "resolver_mode": mode,
        "aiodns_available": _aiodns_available(),
        "soa_minimum": None,
        "cnames": [],
        "host": host,
    }
    if not _aiodns_available():
        logging.warning(
            "aiodns is not installed; cannot bypass the local DNS "
            "negative cache for %s. Continuing with the system "
            "resolver; retries may keep hitting the local negative "
            "cache until it expires.",
            host,
        )
        return diagnostics
    probe_resolver = None
    try:
        probe_resolver = await _new_aiodns_resolver()
        dns_result, dns_error = await _run_query_dns(probe_resolver, host, "A")
        if dns_error is not None:
            diagnostics["probe_error"] = str(dns_error)
        else:
            diagnostics["cnames"] = _cnames_from_dns_result(dns_result)
        minimum = await _probe_soa_minimum(host, probe_resolver)
        if minimum is not None:
            diagnostics["soa_minimum"] = minimum
    except Exception as e:  # noqa: BLE001 - diagnostics must never raise
        diagnostics["probe_error"] = str(e)
    finally:
        if probe_resolver is not None:
            try:
                await probe_resolver.close()
            except Exception as e:  # noqa: BLE001 - cleanup must not raise
                logging.debug("error closing probe resolver: %s", e)
    return diagnostics


def _dns_failure_message(endpoint, max_attempt, ping_max_retries, e, diagnostics):
    """Build the final TrainMLConnectionError message for DNS-name failures."""
    host = diagnostics.get("host") or _resolver_host_from_error(e) or endpoint
    parts = [
        f"Endpoint {endpoint} ping failed after {max_attempt - 1} "
        f"retries ({max_attempt} of {ping_max_retries} attempts) "
        f"due to DNS name-not-found; last error: {str(e)}",
        f"Failed host: {host}",
    ]
    if diagnostics.get("cnames"):
        parts.append(f"CNAME chain: {', '.join(diagnostics['cnames'])}")
    if diagnostics.get("cnames_error"):
        parts.append(f"CNAME chain could not be read: {diagnostics['cnames_error']}")
    if diagnostics.get("soa_minimum") is not None:
        parts.append(
            "Zone SOA minimum (negative-cache TTL) for that name: "
            f"{diagnostics['soa_minimum']}s"
        )
    if not diagnostics.get("aiodns_available"):
        parts.append(
            "aiodns is not installed, so the local resolver negative cache "
            "could not be bypassed"
        )
    parts.append(
        "Resolvers cache DNS name-not-found answers for the zone SOA minimum "
        "TTL (up to 30 minutes); the record may be mid-creation or its "
        "negative answer cached. Retry shortly, or check that the job "
        "endpoint DNS record has been created."
    )
    return " | ".join(parts)


async def ping_endpoint(
    endpoint, auth_token, max_retries=MAX_RETRIES, retry_backoff=RETRY_BACKOFF
):
    """
    Ping the endpoint to ensure it's ready before upload/download operations.

    Retries on all errors (404, 500, DNS errors, etc.) with exponential backoff
    until a 200 response is received. This handles startup timing issues.

    Name-not-found (NXDOMAIN / NODATA) failures are retried a fixed number
    of times (the same retry count used for warmup, e.g. 9), waiting
    DNS_NAME_ERROR_RETRY_DELAY (1 minute) between attempts. There is no
    total-time budget: if we say we will retry N times we do so regardless
    of elapsed wall time. Resolvers cache negative DNS answers, and the
    record may not exist yet (created by the server on attach), so the
    record typically appears within a few minutes. After the first such
    failure the ping switches to a dedicated c-ares channel per attempt
    querying the system-configured nameservers directly, which bypasses the
    local stub resolver's negative cache (mDNSResponder / nss) while staying
    on the enterprise-allowed DNS servers. Without aiodns it keeps using the
    system resolver.

    For ping warmup, calculates retries dynamically to ensure the last retry
    occurs PING_WARMUP_TIMEOUT seconds after the first try.

    Args:
        endpoint: Server endpoint URL
        auth_token: Authentication token
        max_retries: Maximum number of retry attempts (ignored for ping warmup)
        retry_backoff: Exponential backoff base

    Raises:
        TrainMLConnectionError: If ping never returns 200 after max retries
    """
    endpoint = normalize_endpoint(endpoint)
    max_attempt = 1
    resolver_mode = RESOLVER_MODE_DEFAULT
    last_diagnostics = None
    # Calculate retries for ping warmup to reach PING_WARMUP_TIMEOUT
    # Allow max_retries to override when explicitly provided (for testing)
    if max_retries == MAX_RETRIES:
        # Use default, calculate retries for ping warmup
        ping_max_retries = calculate_ping_retries(PING_WARMUP_TIMEOUT, retry_backoff)
    else:
        # Use explicitly provided max_retries (for testing)
        ping_max_retries = max_retries

    while True:
        # Dedicated c-ares channel per attempt once in cares mode
        connector = None
        resolver = None
        try:
            connector, resolver = _create_ping_connector(resolver_mode)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"{endpoint}/ping",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=30,
                ) as response:
                    if response.status == 200:
                        logging.debug(
                            "Endpoint %s is ready (ping successful)", endpoint
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
            if max_attempt < ping_max_retries:
                logging.debug(
                    "Ping attempt %s/%s failed with status %s: %s",
                    max_attempt,
                    ping_max_retries,
                    e.status,
                    e,
                )
                await asyncio.sleep(retry_backoff**max_attempt)
                max_attempt += 1
                continue
            raise TrainMLConnectionError(
                f"Endpoint {endpoint} ping failed after {max_attempt - 1} attempts. "
                f"Last error: HTTP {e.status} - {str(e)}"
            ) from e
        except (ClientConnectorDNSError, ClientConnectorError) as e:
            # Connection refused / reset: short backoff. Name-not-found:
            # escalate to the cache-bypassing c-ares resolver and wait
            # out the negative-cache TTL.
            name_error = _is_dns_name_not_known(e)
            if name_error:
                diagnostics = await _negative_cache_diagnostics(
                    _resolver_host_from_error(e) or endpoint,
                    resolver,
                    resolver_mode,
                )
                last_diagnostics = diagnostics
                resolver_mode = (
                    RESOLVER_MODE_CARES
                    if diagnostics.get("aiodns_available")
                    else RESOLVER_MODE_DEFAULT
                )
                # Retry a fixed number of times at a fixed cadence; no time
                # budget. Give up only once the retry count is spent.
                if max_attempt < ping_max_retries:
                    wait = DNS_NAME_ERROR_RETRY_DELAY
                    logging.info(
                        "Ping attempt %s/%s failed due to DNS name error: %s; "
                        "retrying in %s seconds (aiodns available: %s, "
                        "next resolver: %s)",
                        max_attempt,
                        ping_max_retries,
                        e,
                        wait,
                        diagnostics.get("aiodns_available"),
                        resolver_mode,
                    )
                    await asyncio.sleep(wait)
                    max_attempt += 1
                    continue
                raise TrainMLConnectionError(
                    _dns_failure_message(
                        endpoint,
                        max_attempt,
                        ping_max_retries,
                        e,
                        last_diagnostics,
                    )
                ) from e
            if max_attempt < ping_max_retries:
                delay = _ping_connector_retry_delay(e, max_attempt, retry_backoff)
                logging.debug(
                    "Ping attempt %s/%s failed due to DNS/connection "
                    "error: %s; waiting %s seconds",
                    max_attempt,
                    ping_max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                max_attempt += 1
                continue
            raise TrainMLConnectionError(
                f"Endpoint {endpoint} ping failed after {max_attempt - 1} attempts "
                f"due to DNS/connection error: {str(e)}"
            ) from e
        except (
            ServerDisconnectedError,
            ClientOSError,
            ServerTimeoutError,
            ClientPayloadError,
            asyncio.TimeoutError,
        ) as e:
            if max_attempt < ping_max_retries:
                logging.debug(
                    "Ping attempt %s/%s failed: %s",
                    max_attempt,
                    ping_max_retries,
                    e,
                )
                delay = retry_backoff**max_attempt
                await asyncio.sleep(delay)
                max_attempt += 1
                continue
            raise TrainMLConnectionError(
                f"Endpoint {endpoint} ping failed after {max_attempt - 1} attempts: {str(e)}"
            ) from e
        finally:
            if connector is not None:
                try:
                    await connector.close()
                except OSError:
                    pass
            if resolver is not None:
                try:
                    await resolver.close()
                except OSError:
                    pass


async def retry_request(
    func, *args, max_retries=MAX_RETRIES, retry_backoff=RETRY_BACKOFF, **kwargs
):
    """
    Shared retry logic for network requests.

    For DNS/connection errors (ClientConnectorError), uses more retries and
    an initial delay to handle transient DNS resolution issues. Name-not-
    found (NXDOMAIN / NODATA) errors wait DNS_NAME_ERROR_RETRY_DELAY between
    attempts instead of exponential backoff, because resolvers cache
    negative DNS answers and short retries keep hitting that cache.
    """
    attempt = 1
    effective_max_retries = max_retries

    while attempt <= effective_max_retries:
        try:
            return await func(*args, **kwargs)
        except ClientResponseError as e:
            if e.status in RETRY_STATUSES and attempt < max_retries:
                logging.debug(
                    "Retry %s/%s due to %s: %s",
                    attempt,
                    max_retries,
                    e.status,
                    e,
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
                if _is_dns_name_not_known(e):
                    # Wait out the resolver negative-cache window
                    delay = DNS_NAME_ERROR_RETRY_DELAY
                else:
                    # Initial delay then exponential backoff
                    delay = (
                        DNS_INITIAL_DELAY
                        if attempt == 1
                        else retry_backoff ** (attempt - 1)
                    )
                logging.debug(
                    "Retry %s/%s due to DNS/connection error: %s; "
                    "waiting %s seconds",
                    attempt,
                    effective_max_retries,
                    e,
                    delay,
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
                logging.debug("Retry %s/%s due to %s", attempt, max_retries, e)
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
    upload_id="default",
):
    """Uploads a single chunk with retry logic."""
    start = offset
    end = offset + len(data) - 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{total_size}",
        "Authorization": f"Bearer {auth_token}",
        "Upload-Id": upload_id,
    }

    async def _upload():
        async with session.put(
            f"{endpoint}/upload",
            headers=headers,
            data=data,
        ) as response:
            if response.status == 200:
                try:
                    payload = await response.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                    payload = {}
                await response.release()
                expected_offset = payload.get("expected_offset")
                if isinstance(expected_offset, int):
                    return expected_offset
                return end + 1
            elif response.status in RETRY_STATUSES or response.status in (
                404,
                409,
            ):
                # 404: Cloudflare edge answer while the hostname is unmapped
                # (tunnel recycling) - the caller waits and retries.
                text = await response.text()
                raise ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=text,
                )
            else:
                text = await response.text()
                raise TrainMLConnectionError(
                    f"Chunk {start}-{end} failed with status {response.status}: {text}"
                )

    return await retry_request(_upload)


async def get_upload_status(session, endpoint, auth_token, upload_id="default"):
    async def _status():
        async with session.get(
            f"{endpoint}/upload/status",
            params={"upload_id": upload_id},
            headers={"Authorization": f"Bearer {auth_token}"},
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=text,
                )
            return await response.json()

    data = await retry_request(_status)
    expected_offset = data.get("expected_offset")
    if not isinstance(expected_offset, int) or expected_offset < 0:
        raise TrainMLConnectionError("Invalid upload status response from server")
    return expected_offset


async def upload(endpoint, auth_token, path, show_progress=True):
    """
    Upload a local file or directory as a TAR stream to the server.

    Args:
        endpoint: Server endpoint URL
        auth_token: Authentication token
        path: Local file or directory path to upload
        show_progress: If True and stdout is a TTY, show progress bar (default True)

    Raises:
        ValueError: If path doesn't exist or is invalid
        TrainMLConnectionError: If upload fails or endpoint ping fails
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

    # Approximate total for progress: known only for a single file (tar adds header)
    total_size_approx = os.path.getsize(path) if os.path.isfile(path) else None
    sha512 = hashlib.sha512()
    offset = 0
    last_progress_time = 0.0
    upload_id = str(uuid.uuid4())

    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=30,
        sock_read=10 * 60,
    )

    state = _TransferSessionState(timeout=timeout)
    try:
        await state.open(RESOLVER_MODE_DEFAULT)

        buffered_chunk = None
        buffered_start = 0
        not_found_retries = 0

        while True:
            if buffered_chunk is None:
                chunk = await process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    break  # End of stream
                buffered_chunk = chunk
                buffered_start = offset
                sha512.update(chunk)

            end = buffered_start + len(buffered_chunk) - 1
            now = time.perf_counter()
            if now - last_progress_time >= PROGRESS_THROTTLE_SEC:
                _write_progress(
                    buffered_start,
                    total=total_size_approx,
                    desc=desc,
                    show_progress=show_progress,
                )
                last_progress_time = now

            try:
                server_expected = await upload_chunk(
                    state.session,
                    endpoint,
                    auth_token,
                    buffered_start + len(buffered_chunk),
                    buffered_chunk,
                    buffered_start,
                    upload_id,
                )
                if not isinstance(server_expected, int):
                    server_expected = end + 1
                if server_expected != end + 1:
                    raise TrainMLConnectionError(
                        f"Server expected_offset mismatch: expected {end + 1}, got {server_expected}"
                    )
                offset = end + 1
                buffered_chunk = None
                not_found_retries = 0
            except ClientResponseError as e:
                if e.status == 404:
                    # Cloudflare edge answer while the hostname is unmapped
                    # (tunnel recycling). Wait a fixed interval and re-upload
                    # the same buffered chunk at the same offset.
                    if not_found_retries >= NOT_FOUND_MAX_RETRIES:
                        raise TrainMLConnectionError(
                            f"Upload endpoint returned 404 after "
                            f"{NOT_FOUND_MAX_RETRIES} retries; the endpoint "
                            f"may not be ready yet (tunnel recycling). "
                            f"Last error: {e.message}"
                        ) from e
                    not_found_retries += 1
                    logging.warning(
                        "Upload endpoint returned 404; retry %s/%s in %ss "
                        "(tunnel may be recycling)",
                        not_found_retries,
                        NOT_FOUND_MAX_RETRIES,
                        NOT_FOUND_RETRY_DELAY,
                    )
                    await asyncio.sleep(NOT_FOUND_RETRY_DELAY)
                    continue
                if e.status not in RETRY_STATUSES and e.status != 409:
                    raise
                server_offset = await get_upload_status(
                    state.session, endpoint, auth_token, upload_id
                )
                if server_offset == buffered_start:
                    continue
                if server_offset == end + 1:
                    offset = server_offset
                    buffered_chunk = None
                    continue
                raise TrainMLConnectionError(
                    f"Upload offset desync (client {buffered_start}-{end}, server expected {server_offset}). "
                    "Cannot safely resume tar stream."
                ) from e
            except (
                ServerDisconnectedError,
                ClientConnectorError,
                ClientOSError,
                ServerTimeoutError,
                ClientPayloadError,
                asyncio.TimeoutError,
            ) as exc:
                if _is_dns_name_not_known(exc):
                    # The local resolver likely cached the negative answer;
                    # rebuild on a fresh c-ares channel for this attempt.
                    await state.open(RESOLVER_MODE_CARES)
                server_offset = await get_upload_status(
                    state.session, endpoint, auth_token, upload_id
                )
                if server_offset == buffered_start:
                    continue
                if server_offset == end + 1:
                    offset = server_offset
                    buffered_chunk = None
                    continue
                raise TrainMLConnectionError(
                    f"Upload offset desync (client {buffered_start}-{end}, server expected {server_offset}). "
                    "Cannot safely resume tar stream."
                ) from exc

        _write_progress(
            offset,
            total=total_size_approx,
            desc=desc,
            last=True,
            show_progress=show_progress,
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
        final_session = state.session

        async def _finalize():
            async with final_session.post(
                f"{endpoint}/finalize",
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Upload-Id": upload_id,
                },
                json={"hash": file_hash},
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=text,
                    )
                return await response.json()

        try:
            data = await retry_request(_finalize)
        except ClientResponseError as e:
            retried = (
                f" after {MAX_RETRIES} attempts" if e.status in RETRY_STATUSES else ""
            )
            raise TrainMLConnectionError(
                f"Finalize failed (HTTP {e.status}{retried}): {e.message}"
            ) from e
        logging.debug("Upload finalized: %s", data)
    finally:
        await state.close()


async def download(
    endpoint, auth_token, target_directory, file_name=None, show_progress=True
):
    """
    Download a directory archive from the server and extract it.

    Args:
        endpoint: Server endpoint URL
        auth_token: Authentication token
        target_directory: Directory to extract files to (or save zip file)
        file_name: Optional filename override for zip archive (if ARCHIVE=true).
                   If not provided, filename is extracted from Content-Disposition header.
        show_progress: If True and stdout is a TTY, show progress bar (default True)

    Raises:
        TrainMLConnectionError: If download fails or endpoint ping fails
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
    state = _TransferSessionState(timeout=None)
    try:
        await state.open(RESOLVER_MODE_DEFAULT)

        use_archive = False
        try:

            async def _get_info():
                async with state.session.get(
                    f"{endpoint}/info",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=30,
                ) as response:
                    if response.status != 200:
                        try:
                            error_text = await response.text()
                        except (aiohttp.ClientError, UnicodeDecodeError):
                            error_text = (
                                f"Unable to read response body "
                                f"(status: {response.status})"
                            )
                        raise ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=error_text,
                        )
                    return await response.json()

            try:
                info = await retry_request(_get_info)
                use_archive = info.get("archive", False)
            except ClientConnectorError as e:
                if _is_dns_name_not_known(e):
                    # Local resolver cached the negative answer; rebuild
                    # on a fresh c-ares channel and try once more.
                    await state.open(RESOLVER_MODE_CARES)
                    info = await retry_request(_get_info)
                    use_archive = info.get("archive", False)
                else:
                    raise
        except InvalidURL as e:
            raise TrainMLConnectionError(
                f"Invalid endpoint URL: {endpoint}. "
                f"Please ensure the URL includes a protocol (http://https://). "
                f"Error: {str(e)}"
            ) from e
        except (TrainMLConnectionError, ClientResponseError) as e:
            # If /info endpoint is not available (404) or other error,
            # default to TAR stream mode and continue
            if isinstance(e, TrainMLConnectionError) and "404" in str(e):
                logging.debug(
                    "Warning: /info endpoint not available, defaulting to TAR stream mode"
                )
            elif isinstance(e, ClientResponseError) and e.status == 404:
                logging.debug(
                    "Warning: /info endpoint not available, defaulting to TAR stream mode"
                )
            else:
                # For other errors, convert ClientResponseError to TrainMLConnectionError
                # to maintain backward compatibility
                if isinstance(e, ClientResponseError):
                    error_msg = getattr(e, "message", str(e))
                    raise TrainMLConnectionError(
                        f"Failed to get server info (status {e.status}): {error_msg}"
                    ) from e
                # For TrainMLConnectionError, re-raise as-is
                raise

        # Download the archive
        # Note: Do NOT use "async with session.get() as response" - exiting the
        # context manager would release/close the connection before we read the
        # body. We need the response (and connection) to stay open for streaming.
        async def _download():
            response = await state.session.get(
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

        async def _download_once():
            try:
                return await retry_request(_download)
            except ClientConnectorError as e:
                if _is_dns_name_not_known(e):
                    # Local resolver cached the negative answer; rebuild on a
                    # fresh c-ares channel and try once more.
                    await state.open(RESOLVER_MODE_CARES)
                    return await retry_request(_download)
                raise

        not_found_retries = 0
        while True:
            try:
                response = await _download_once()
                break
            except ClientResponseError as e:
                if e.status != 404:
                    raise
                # Cloudflare edge answer while the hostname is unmapped
                # (tunnel recycling). Wait a fixed interval and retry.
                if not_found_retries >= NOT_FOUND_MAX_RETRIES:
                    raise TrainMLConnectionError(
                        f"Download endpoint returned 404 after "
                        f"{NOT_FOUND_MAX_RETRIES} retries; the endpoint "
                        f"may not be ready yet (tunnel recycling). "
                        f"Last error: {e.message}"
                    ) from e
                not_found_retries += 1
                logging.warning(
                    "Download endpoint returned 404; retry %s/%s in %ss "
                    "(tunnel may be recycling)",
                    not_found_retries,
                    NOT_FOUND_MAX_RETRIES,
                    NOT_FOUND_RETRY_DELAY,
                )
                await asyncio.sleep(NOT_FOUND_RETRY_DELAY)

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
            logging.debug("Response Content-Length: %s bytes", content_length)
        logging.debug("Response Content-Type: %s", content_type)

        total_download_size = None
        if content_length:
            try:
                total_download_size = int(content_length)
            except (TypeError, ValueError):
                pass

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
                        match = re.search(r'filename="?([^"]+)"?', content_disposition)
                        if match:
                            file_name = match.group(1)
                        else:
                            # Fallback: try without quotes
                            match = re.search(r"filename=([^;]+)", content_disposition)
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
                last_progress_time = 0.0
                async with aiofiles.open(output_path, "wb") as f:
                    # Stream the response content in chunks
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        await f.write(chunk)
                        total_bytes += len(chunk)
                        now = time.perf_counter()
                        if now - last_progress_time >= PROGRESS_THROTTLE_SEC:
                            _write_progress(
                                total_bytes,
                                total=total_download_size,
                                desc="Downloading",
                                show_progress=show_progress,
                            )
                            last_progress_time = now

                _write_progress(
                    total_bytes,
                    total=total_download_size,
                    desc="Downloading",
                    last=True,
                    show_progress=show_progress,
                )

                if total_bytes == 0:
                    raise TrainMLConnectionError(
                        "Downloaded file is empty (0 bytes). "
                        "The server may not have any files to download, or there was an error streaming the response."
                    )

                logging.info(
                    "Archive saved to: %s (%s bytes)", output_path, total_bytes
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

                total_bytes = 0
                last_progress_time = 0.0
                # Stream response to tar process
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    extract_process.stdin.write(chunk)
                    await extract_process.stdin.drain()
                    total_bytes += len(chunk)
                    now = time.perf_counter()
                    if now - last_progress_time >= PROGRESS_THROTTLE_SEC:
                        _write_progress(
                            total_bytes,
                            total=None,
                            desc="Downloading",
                            show_progress=show_progress,
                        )
                        last_progress_time = now

                _write_progress(
                    total_bytes,
                    total=None,
                    desc="Downloading",
                    last=True,
                    show_progress=show_progress,
                )

                extract_process.stdin.close()
                await extract_process.wait()

                if extract_process.returncode != 0:
                    stderr = await extract_process.stderr.read()
                    raise TrainMLException(
                        f"tar extraction failed: {stderr.decode() if stderr else 'Unknown error'}"
                    )

                logging.info("Files extracted to: %s", target_directory)
        finally:
            response.close()

        # Finalize download
        final_session = state.session

        async def _finalize():
            async with final_session.post(
                f"{endpoint}/finalize",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={},
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=text,
                    )
                return await response.json()

        try:
            data = await retry_request(_finalize)
        except ClientResponseError as e:
            retried = (
                f" after {MAX_RETRIES} attempts" if e.status in RETRY_STATUSES else ""
            )
            raise TrainMLConnectionError(
                f"Finalize failed (HTTP {e.status}{retried}): {e.message}"
            ) from e
        logging.debug("Download finalized: %s", data)
    finally:
        await state.close()
