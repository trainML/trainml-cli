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
from trainml.connections import Connection


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
            project_uuid=kwargs.get("project_uuid") or self.trainml.active_project,
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
        await self.trainml._query(f"/job/{id}", "DELETE", dict(**kwargs, force=True))


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

    async def get_connection_utility_url(self):
        resp = await self.trainml._query(
            f"/job/{self._id}/download",
            "GET",
            dict(project_uuid=self._project_uuid),
        )
        return resp

    def get_connection_details(self):
        details = dict(
            entity_type="job",
            project_uuid=self._job.get("project_uuid"),
            cidr=self.dict.get("vpn").get("cidr"),
            ssh_port=(
                self._job.get("vpn").get("client").get("ssh_port")
                if self._job.get("vpn").get("client")
                else None
            ),
            model_path=(
                self._job.get("model").get("source_uri")
                if self._job.get("model").get("source_type") == "local"
                else None
            ),
            input_path=(
                self._job.get("data").get("input_uri")
                if self._job.get("data").get("input_type") == "local"
                else None
            ),
            output_path=(
                self._job.get("data").get("output_uri")
                if self._job.get("data").get("output_type") == "local"
                else None
            ),
        )
        return details

    async def open(self):
        if self.type != "notebook":
            raise SpecificationError(
                "type",
                "Only notebook jobs can be opened.",
            )
        webbrowser.open(self.notebook_url)

    async def connect(self):
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
        if self._job.get("vpn").get("status") == "n/a":
            logging.info("Local connection not enabled for this job.")
            return
        if self.status == "new":
            await self.wait_for("waiting for data/model download")
        connection = Connection(
            self.trainml, entity_type="job", id=self.id, entity=self
        )
        await connection.start()
        return connection.status

    async def disconnect(self):
        if self._job.get("vpn").get("status") == "n/a":
            logging.info("Local connection not enabled for this job.")
            return
        connection = Connection(
            self.trainml, entity_type="job", id=self.id, entity=self
        )
        await connection.stop()
        return connection.status

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
            w.get("job_worker_uuid"): ind + 1 for ind, w in enumerate(self._workers)
        }
        worker_numbers["data_worker"] = 0

        def handler(data):
            if data.get("type") == "subscription":
                data["worker_number"] = worker_numbers.get(data.get("stream"))
                if msg_handler:
                    msg_handler(data)
                else:
                    timestamp = datetime.fromtimestamp(int(data.get("time")) / 1000)
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
        if self.type == "notebook" and self.status != "waiting for data/model download":
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
            raise SpecificationError("job", "Only notebook job types can be copied")

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
            "waiting for GPUs",
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
        if status == "waiting for GPUs":
            warnings.warn(
                "'waiting for GPUs' status is deprecated, use 'waiting for resources' instead.",
                DeprecationWarning,
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
        POLL_INTERVAL = max(min(timeout / 60, POLL_INTERVAL_MAX), POLL_INTERVAL_MIN)
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
                    in [
                        "waiting for GPUs",
                        "waiting for resources",
                    ]  ## this status could be very short and the polling could miss it
                    and self.status
                    not in ["new", "waiting for GPUs", "waiting for resources"]
                )
                or (
                    status
                    == "waiting for data/model download"  ## this status could be very short and the polling could miss it
                    and self.status
                    not in [
                        "new",
                        "waiting for GPUs",
                        "waiting for resources",
                        "waiting for data/model download",
                    ]
                )
            ):
                return self
            elif self.status == "failed":
                raise JobError(self.status, self)
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
