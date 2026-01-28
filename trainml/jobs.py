import json
import asyncio
import math
import logging
import warnings
import webbrowser
from datetime import datetime

from trainml.exceptions import (
    ApiError,
    JobError,
    SpecificationError,
    TrainMLException,
)
from trainml.utils.transfer import upload, download


class Jobs(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id, **kwargs):
        resp = await self.trainml._query(f"/job/{id}", "GET", kwargs)
        return Job(self.trainml, **resp)

    async def list(self, **kwargs):
        resp = await self.trainml._query(f"/job", "GET", kwargs)
        jobs = [Job(self.trainml, **job) for job in resp]
        return jobs

    async def create(
        self,
        name,
        type,
        gpu_type=None,
        gpu_types=[],
        gpu_count=None,
        cpu_count=None,
        disk_size=10,
        max_price=10,
        worker_commands=[],
        environment=dict(),
        data=dict(),
        model=dict(),
        endpoint=dict(),
        **kwargs,
    ):
        resources = {
            k: v
            for k, v in dict(
                gpu_count=gpu_count,
                disk_size=disk_size,
                max_price=max_price,
                cpu_count=cpu_count,
            ).items()
            if v is not None
        }

        if len(gpu_types):
            resources["gpu_types"] = gpu_types
        elif gpu_type:
            resources["gpu_type_id"] = gpu_type
        else:
            raise SpecificationError(
                "resources",
                "Invalid resource specification, either 'gpu_type' or 'gpu_types' must be provided",
            )

        config = dict(
            name=name,
            type=type,
            resources=resources,
            worker_commands=worker_commands,
            workers=kwargs.get("workers"),
            environment=environment,
            data=data,
            model=model,
            endpoint=endpoint,
            source_job_uuid=kwargs.get("source_job_uuid"),
            project_uuid=kwargs.get("project_uuid")
            or self.trainml.active_project,
        )
        payload = {
            k: v
            for k, v in config.items()
            if v or (k in ["model"] and not kwargs.get("source_job_uuid"))
        }
        if (
            not payload.get("worker_commands")
            and not payload.get("workers")
            and not kwargs.get("source_job_uuid")
        ):
            payload["worker_commands"] = []
        logging.info(f"Creating Job {name}")
        job = await self.create_json(payload)
        logging.info(f"Created Job {name} with id {job.id}")
        return job

    async def create_json(self, payload):
        logging.debug(f"Job payload: {payload}")
        resp = await self.trainml._query("/job", "POST", None, payload)
        job = Job(self.trainml, **resp)
        return job

    async def remove(self, id, **kwargs):
        await self.trainml._query(
            f"/job/{id}", "DELETE", dict(**kwargs, force=True)
        )


class Job:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._job = kwargs
        self._id = self._job.get("id", self._job.get("job_uuid"))
        self._name = self._job.get("name")
        self._status = self._job.get("status")
        self._type = self._job.get("type")
        self._workers = self._job.get("workers")
        self._credits = self._job.get("credits")
        self._project_uuid = self._job.get("project_uuid")

    @property
    def dict(self) -> dict:
        return {k: v for k, v in self._job.items() if k not in ["nb_token"]}

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> str:
        return self._status

    @property
    def type(self) -> str:
        return self._type

    @property
    def workers(self) -> list:
        return self._workers

    @property
    def credits(self) -> float:
        return self._credits

    @property
    def url(self) -> str:
        return self._job.get("endpoint").get("url")

    @property
    def notebook_url(self) -> str:
        if self.type != "notebook":
            return None
        return f"{self.url}/?token={self._job.get('nb_token')}"

    def __str__(self):
        return json.dumps(self.dict)

    def __repr__(self):
        return f"Job( trainml , {self.dict.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    def get_create_json(self):
        root_keys = [
            "name",
            "type",
            "resources",
            "model",
            "data",
            "environment",
            "endpoint",
            "workers",
            "project_uuid",
        ]
        resources_keys = [
            "gpu_count",
            "gpu_types",
            "disk_size",
            "max_price",
            "preemptible",
            "cpu_count",
        ]
        model_keys = [
            "source_type",
            "source_uri",
            "project_uuid",
            "checkpoints",
        ]
        data_keys = [
            "datasets",
            "input_type",
            "input_uri",
            "input_options",
            "output_type",
            "output_uri",
            "output_options",
        ]
        environment_keys = [
            "type",
            "env",
            "custom_image",
            "worker_key_types",
            "packages",
        ]
        endpoint_keys = ["routes", "start_command", "reservation_id"]
        create_json = dict()
        for k, v in self.dict.items():
            if k in root_keys:
                if k == "resources":
                    resources = dict()
                    for k2, v2 in v.items():
                        if k2 in resources_keys:
                            resources[k2] = v2
                    create_json["resources"] = resources
                elif k == "model":
                    model = dict()
                    for k2, v2 in v.items():
                        if k2 in model_keys:
                            model[k2] = v2
                    create_json["model"] = model
                elif k == "data":
                    data = dict()
                    for k2, v2 in v.items():
                        if k2 in data_keys:
                            data[k2] = v2
                    create_json["data"] = data
                elif k == "environment":
                    environment = dict()
                    for k2, v2 in v.items():
                        if k2 in environment_keys:
                            environment[k2] = v2
                    create_json["environment"] = environment
                elif k == "endpoint":
                    endpoint = dict()
                    for k2, v2 in v.items():
                        if k2 in endpoint_keys:
                            endpoint[k2] = v2
                    create_json["endpoint"] = endpoint
                elif k == "workers":
                    workers = []
                    for worker in v:
                        workers.append(worker.get("command"))
                    create_json["workers"] = workers
                else:
                    create_json[k] = v
        return create_json

    async def start(self):
        resp = await self.trainml._query(
            f"/job/{self._id}",
            "PATCH",
            dict(project_uuid=self._project_uuid),
            dict(command="start"),
        )
        self.__init__(self.trainml, **resp)
        return self

    async def stop(self):
        resp = await self.trainml._query(
            f"/job/{self._id}",
            "PATCH",
            dict(project_uuid=self._project_uuid),
            dict(command="stop"),
        )
        self.__init__(self.trainml, **resp)
        return self

    async def update(self, data):
        if self.type != "notebook":
            raise SpecificationError(
                "type",
                "Only notebook jobs can be modified.",
            )
        resp = await self.trainml._query(
            f"/job/{self._id}",
            "PATCH",
            dict(project_uuid=self._project_uuid),
            data,
        )
        self.__init__(self.trainml, **resp)
        return self

    async def get_worker_log_url(self, job_worker_uuid):
        resp = await self.trainml._query(
            f"/job/{self._id}/worker/{job_worker_uuid}/logs",
            "GET",
            dict(project_uuid=self._project_uuid),
        )
        return resp

    async def open(self):
        if self.type != "notebook":
            raise SpecificationError(
                "type",
                "Only notebook jobs can be opened.",
            )
        webbrowser.open(self.notebook_url)

    async def connect(self):
        # Handle notebook/endpoint special cases
        if self.type == "notebook" and self.status not in [
            "new",
            "waiting for data/model download",
            "waiting for GPUs",
            "waiting for resources",
        ]:
            raise SpecificationError(
                "type",
                "Notebooks cannot be connected to after model download is complete.  Use open() instead.",
            )
        if self.type == "endpoint" and self.status not in [
            "new",
            "waiting for data/model download",
        ]:
            return self.url

        # Check for invalid statuses
        if self.status in [
            "failed",
            "finished",
            "canceled",
            "archived",
            "removed",
            "removing",
        ]:
            raise SpecificationError(
                "status",
                f"You can only connect to active jobs.",
            )

        # Only allow specific statuses for connect
        if self.status not in [
            "waiting for data/model download",
            "uploading",
            "running",
        ]:
            if self.status == "new":
                await self.wait_for("waiting for data/model download")
            else:
                raise SpecificationError(
                    "status",
                    f"You can only connect to jobs in 'waiting for data/model download', 'uploading', or 'running' status.",
                )

        # Refresh to get latest job data
        await self.refresh()

        # Re-check status after refresh (status may have changed if attach() is running in parallel)
        if self.status not in [
            "waiting for data/model download",
            "uploading",
            "running",
        ]:
            raise SpecificationError(
                "status",
                f"Job status changed to '{self.status}'. You can only connect to jobs in 'waiting for data/model download', 'uploading', or 'running' status.",
            )

        if self.status == "waiting for data/model download":
            # Upload model and/or data if local
            model = self._job.get("model", {})
            data = self._job.get("data", {})

            model_local = model.get("source_type") == "local"
            data_local = data.get("input_type") == "local"

            if not model_local and not data_local:
                raise SpecificationError(
                    "status",
                    f"Job has no local model or data to upload. Model source_type: {model.get('source_type')}, Data input_type: {data.get('input_type')}",
                )

            upload_tasks = []

            if model_local:
                model_auth_token = model.get("auth_token")
                model_hostname = model.get("hostname")
                model_source_uri = model.get("source_uri")

                if (
                    not model_auth_token
                    or not model_hostname
                    or not model_source_uri
                ):
                    raise SpecificationError(
                        "status",
                        f"Job model missing required connection properties (auth_token, hostname, source_uri).",
                    )

                upload_tasks.append(
                    upload(model_hostname, model_auth_token, model_source_uri)
                )

            if data_local:
                data_auth_token = data.get("input_auth_token")
                data_hostname = data.get("input_hostname")
                data_input_uri = data.get("input_uri")

                if (
                    not data_auth_token
                    or not data_hostname
                    or not data_input_uri
                ):
                    raise SpecificationError(
                        "status",
                        f"Job data missing required connection properties (input_auth_token, input_hostname, input_uri).",
                    )

                upload_tasks.append(
                    upload(data_hostname, data_auth_token, data_input_uri)
                )

            # Upload both in parallel if both are local
            if upload_tasks:
                await asyncio.gather(*upload_tasks)

        elif self.status in ["uploading", "running"]:
            # Download output if local
            data = self._job.get("data", {})

            if data.get("output_type") != "local":
                raise SpecificationError(
                    "status",
                    f"Job output_type is not 'local', cannot download output.",
                )

            output_uri = data.get("output_uri")
            if not output_uri:
                raise SpecificationError(
                    "status",
                    f"Job data missing output_uri for local output download.",
                )

            # Track which workers we've already started downloading
            downloading_workers = set()
            download_tasks = []

            # Poll until all workers are finished
            while True:
                # Refresh job to get latest worker statuses
                await self.refresh()

                # Get fresh workers list
                workers = self._job.get("workers", [])
                if not workers:
                    raise SpecificationError(
                        "status",
                        f"Job has no workers.",
                    )

                # Check if job is finished
                if self.status in ["finished", "canceled", "failed"]:
                    break

                # Check all workers for uploading status
                for worker in workers:
                    worker_id = worker.get("job_worker_uuid") or worker.get(
                        "id"
                    )
                    worker_status = worker.get("status")

                    # Start download for any worker that enters uploading status
                    if (
                        worker_status == "uploading"
                        and worker_id not in downloading_workers
                    ):
                        output_auth_token = worker.get("output_auth_token")
                        output_hostname = worker.get("output_hostname")

                        if not output_auth_token or not output_hostname:
                            logging.warning(
                                f"Worker {worker_id} in uploading status missing output_auth_token or output_hostname, skipping."
                            )
                            continue

                        downloading_workers.add(worker_id)
                        # Create and start download task (runs in parallel)
                        logging.info(
                            f"Starting download for worker {worker_id} from {output_hostname} to {output_uri}"
                        )
                        try:
                            download_task = asyncio.create_task(
                                download(
                                    output_hostname,
                                    output_auth_token,
                                    output_uri,
                                )
                            )
                            download_tasks.append(download_task)
                            logging.debug(
                                f"Download task created for worker {worker_id}, task: {download_task}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Failed to create download task for worker {worker_id}: {e}",
                                exc_info=True,
                            )
                            raise

                # Check if any download tasks have completed or failed
                if download_tasks:
                    completed_tasks = [
                        task for task in download_tasks if task.done()
                    ]
                    for task in completed_tasks:
                        try:
                            await task  # This will raise if the task failed
                            logging.info(
                                f"Download task completed successfully"
                            )
                        except Exception as e:
                            logging.error(
                                f"Download task failed: {e}", exc_info=True
                            )
                            raise
                    # Remove completed tasks
                    download_tasks = [
                        task for task in download_tasks if not task.done()
                    ]

                # Check if all workers are finished
                all_finished = all(
                    worker.get("status") in ["finished", "removed"]
                    for worker in workers
                )

                if all_finished:
                    break

                # If we have active download tasks, wait a bit for them to make progress
                # but don't wait the full 30 seconds - check more frequently
                if download_tasks:
                    await asyncio.sleep(5)
                else:
                    # Wait 30 seconds before next poll if no downloads in progress
                    await asyncio.sleep(30)

            # Wait for all download tasks to complete
            if download_tasks:
                logging.info(
                    f"Waiting for {len(download_tasks)} download task(s) to complete"
                )
                await asyncio.gather(*download_tasks)
                logging.info("All downloads completed")

    async def remove(self, force=False):
        await self.trainml._query(
            f"/job/{self._id}",
            "DELETE",
            dict(project_uuid=self._project_uuid, force=force),
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/job/{self.id}", "GET", dict(project_uuid=self._project_uuid)
        )
        self.__init__(self.trainml, **resp)
        return self

    def _get_msg_handler(self, msg_handler):
        worker_numbers = {
            w.get("job_worker_uuid"): ind + 1
            for ind, w in enumerate(self._workers)
        }
        worker_numbers["data_worker"] = 0

        def handler(data):
            if data.get("type") == "subscription":
                data["worker_number"] = worker_numbers.get(data.get("stream"))
                if msg_handler:
                    msg_handler(data)
                else:
                    timestamp = datetime.fromtimestamp(
                        int(data.get("time")) / 1000
                    )
                    if len(self._workers) > 1:
                        print(
                            f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: Worker {data.get('worker_number')} - {data.get('msg').rstrip()}"
                        )
                    else:
                        print(
                            f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: {data.get('msg').rstrip()}"
                        )

        return handler

    async def attach(self, msg_handler=None):
        if (
            self.type == "notebook"
            and self.status != "waiting for data/model download"
        ):
            raise SpecificationError(
                "type",
                "Notebooks cannot be attached to after model download is complete.  Use open() instead.",
            )
        await self.refresh()
        if self.status not in ["finished", "failed"]:
            await self.trainml._ws_subscribe(
                "job",
                self._project_uuid,
                self.id,
                self._get_msg_handler(msg_handler),
            )

    async def copy(self, name, **kwargs):
        logging.debug(f"copy request - name: {name} ; kwargs: {kwargs}")
        if self.type != "notebook":
            raise SpecificationError(
                "job", "Only notebook job types can be copied"
            )

        job = await self.trainml.jobs.create(
            name,
            type=kwargs.get("type") or self.type,
            gpu_type=kwargs.get("gpu_type"),
            gpu_types=kwargs.get("gpu_types")
            or self._job.get("resources").get("gpu_types"),
            gpu_count=kwargs.get("gpu_count")
            or self._job.get("resources").get("gpu_count"),
            cpu_count=kwargs.get("cpu_count")
            or self._job.get("resources").get("cpu_count"),
            disk_size=kwargs.get("disk_size")
            or self._job.get("resources").get("disk_size"),
            max_price=kwargs.get("max_price")
            or self._job.get("resources").get("max_price"),
            worker_commands=kwargs.get("worker_commands"),
            workers=kwargs.get("workers"),
            environment=kwargs.get("environment"),
            data=kwargs.get("data"),
            model=kwargs.get("model"),
            source_job_uuid=self.id,
        )
        logging.debug(f"copy result: {job}")
        return job

    async def wait_for(self, status, timeout=300):
        if self.status == status or (
            self.type == "training"
            and status == "finished"
            and self.status == "stopped"
        ):
            return
        valid_statuses = [
            "waiting for data/model download",
            "waiting for resources",
            "running",
            "stopped",
            "finished",
            "archived",
        ]
        if not status in valid_statuses:
            raise SpecificationError(
                "status",
                f"Invalid wait_for status {status}.  Valid statuses are: {valid_statuses}",
            )
        if (self.type == "training") and status == "stopped":
            warnings.warn(
                "'stopped' status is deprecated for training jobs, use 'finished' instead.",
                DeprecationWarning,
            )

        MAX_TIMEOUT = 24 * 60 * 60
        if timeout > MAX_TIMEOUT:
            raise SpecificationError(
                "timeout",
                f"timeout must be less than {MAX_TIMEOUT} seconds.",
            )

        POLL_INTERVAL_MIN = 5
        POLL_INTERVAL_MAX = 60
        POLL_INTERVAL = max(
            min(timeout / 60, POLL_INTERVAL_MAX), POLL_INTERVAL_MIN
        )
        retry_count = math.ceil(timeout / POLL_INTERVAL)
        count = 0
        while count < retry_count:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                await self.refresh()
            except ApiError as e:
                if status == "archived" and e.status == 404:
                    return
                raise e
            if (
                self.status == status
                or (
                    status
                    == "waiting for resources"  ## this status could be very short and the polling could miss it
                    and self.status not in ["new", "waiting for resources"]
                )
                or (
                    status
                    == "waiting for data/model download"  ## this status could be very short and the polling could miss it
                    and self.status
                    not in [
                        "new",
                        "waiting for resources",
                        "waiting for data/model download",
                    ]
                )
                or (
                    status
                    == "running"  ## this status could be too short for polling could miss it
                    and self.status in ["uploading", "finished"]
                )
            ):
                return self
            elif self.status == "failed":
                raise JobError(self.status, self)
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
