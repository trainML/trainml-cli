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


def _clean_datasets_selection(
    requested_datasets=[],
):
    datasets = []
    for dataset in requested_datasets:
        if dataset.get("type") not in ["existing", "public"]:
            raise SpecificationError(
                "datasets",
                "Invalid dataset specification, 'type' must be in ['existing','public']",
            )
        if "id" in dataset.keys():
            datasets.append(
                dict(
                    id=dataset.get("id"),
                    type=dataset.get("type"),
                )
            )
        elif "dataset_uuid" in dataset.keys():
            datasets.append(dataset)
        elif "name" in dataset.keys():
            datasets.append(
                dict(
                    id=dataset.get("name"),
                    type=dataset.get("type"),
                )
            )
        else:
            raise SpecificationError(
                "datasets",
                "Invalid dataset specification, either 'id' or 'name' must be provided",
            )
    return datasets


class Jobs(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id):
        resp = await self.trainml._query(f"/job/{id}", "GET")
        return Job(self.trainml, **resp)

    async def list(self):
        resp = await self.trainml._query(f"/job", "GET")
        jobs = [Job(self.trainml, **job) for job in resp]
        return jobs

    async def create(
        self,
        name,
        type,
        gpu_type,
        gpu_count,
        disk_size,
        worker_commands=[],
        environment=dict(type="DEEPLEARNING_PY38"),
        data=dict(),
        model=dict(),
        **kwargs,
    ):

        if type in ["headless", "interactive"]:
            new_type = "notebook" if type == "interactive" else "training"
            warnings.warn(
                f"'{type}' type is deprecated, use '{new_type}' instead.",
                DeprecationWarning,
            )
        gpu_types = await self.trainml.gpu_types.list()

        selected_gpu_type = next(
            (g for g in gpu_types if g.name == gpu_type or g.id == gpu_type),
            None,
        )
        if not selected_gpu_type:
            raise SpecificationError("gpu_type", "GPU Type Not Found")

        if data and data.get("datasets"):
            datasets = _clean_datasets_selection(data.get("datasets"))
            data["datasets"] = datasets

        config = dict(
            name=name,
            type=type,
            resources=dict(
                gpu_type_id=selected_gpu_type.id,
                gpu_count=gpu_count,
                disk_size=disk_size,
            ),
            worker_commands=worker_commands,
            workers=kwargs.get("workers"),
            environment=environment,
            data=data,
            model=model,
            source_job_uuid=kwargs.get("source_job_uuid"),
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

    async def remove(self, id):
        await self.trainml._query(f"/job/{id}", "DELETE", dict(force=True))


class Job:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._job = kwargs
        self._id = self._job.get("id", self._job.get("job_uuid"))
        self._name = self._job.get("name")
        self._provider = self._job.get("provider")
        self._status = self._job.get("status")
        self._type = self._job.get("type")
        self._workers = self._job.get("workers")
        self._credits = self._job.get("credits")

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
    def provider(self) -> str:
        return self._provider

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
    def notebook_url(self) -> str:
        if self.type != "notebook":
            return None
        return f"https://notebook.{self.trainml.domain_suffix}/{self.id}/?token={self._job.get('nb_token')}"

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
            "workers",
        ]
        resources_keys = ["gpu_count", "gpu_type_id", "disk_size"]
        model_keys = ["git_uri", "model_uuid"]
        data_keys = [
            "datasets",
            "input_type",
            "input_uri",
            "output_type",
            "output_uri",
        ]
        environment_keys = ["type", "env", "worker_key_types"]
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
                elif k == "workers":
                    workers = []
                    for worker in v:
                        workers.append(worker.get("command"))
                    create_json["workers"] = workers
                else:
                    create_json[k] = v
        return create_json

    async def start(self):
        await self.trainml._query(
            f"/job/{self._id}", "PATCH", None, dict(command="start")
        )

    async def stop(self):
        await self.trainml._query(
            f"/job/{self._id}", "PATCH", None, dict(command="stop")
        )

    async def get_worker_log_url(self, job_worker_uuid):
        resp = await self.trainml._query(
            f"/job/{self._id}/worker/{job_worker_uuid}/logs", "GET"
        )
        return resp

    async def get_connection_utility_url(self):
        resp = await self.trainml._query(f"/job/{self._id}/download", "GET")
        return resp

    def get_connection_details(self):
        details = dict(
            cidr=self.dict.get("vpn").get("cidr"),
            ssh_port=self._job.get("vpn").get("client").get("ssh_port")
            if self._job.get("vpn").get("client")
            else None,
            model_path=self._job.get("model").get("source_uri")
            if self._job.get("model").get("source_type") == "local"
            else None,
            input_path=self._job.get("data").get("input_uri")
            if self._job.get("data").get("input_type") == "local"
            else None,
            output_path=self._job.get("data").get("output_uri")
            if self._job.get("data").get("output_type") == "local"
            else None,
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
        if (
            self.type == "notebook"
            and self.status != "waiting for data/model download"
        ):
            raise SpecificationError(
                "type",
                "Notebooks cannot be connected to after model download is complete.  Use open() instead.",
            )
        connection = Connection(
            self.trainml, entity_type="job", id=self.id, entity=self
        )
        await connection.start()
        return connection.status

    async def disconnect(self):
        connection = Connection(
            self.trainml, entity_type="job", id=self.id, entity=self
        )
        await connection.stop()
        return connection.status

    async def remove(self):
        await self.trainml._query(f"/job/{self._id}", "DELETE")

    async def refresh(self):
        resp = await self.trainml._query(f"/job/{self.id}", "GET")
        self.__init__(self.trainml, **resp)
        return self

    def _get_msg_handler(self, msg_handler):
        worker_numbers = {
            w.get("job_worker_uuid"): ind + 1
            for ind, w in enumerate(self._workers)
        }

        def handler(msg):
            data = json.loads(msg.data)
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
                "job", self.id, self._get_msg_handler(msg_handler)
            )

    async def copy(self, name, **kwargs):
        logging.debug(f"copy request - name: {name} ; kwargs: {kwargs}")
        if self.type not in ["interactive", "notebook"]:
            raise SpecificationError(
                "job", "Only notebook job types can be copied"
            )

        job = await self.trainml.jobs.create(
            name,
            type=kwargs.get("type") or self.type,
            gpu_type=kwargs.get("gpu_type")
            or self._job.get("resources").get("gpu_type_id"),
            gpu_count=kwargs.get("gpu_count")
            or self._job.get("resources").get("gpu_count"),
            disk_size=kwargs.get("disk_size")
            or self._job.get("resources").get("disk_size"),
            worker_commands=kwargs.get("worker_commands"),
            workers=kwargs.get("workers"),
            environment=kwargs.get("environment"),
            data=kwargs.get("data"),
            source_job_uuid=self.id,
        )
        logging.debug(f"copy result: {job}")
        return job

    async def wait_for(self, status, timeout=300):
        valid_statuses = [
            "waiting for data/model download",
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
        if (
            self.type == "headless" or self.type == "training"
        ) and status == "stopped":
            warnings.warn(
                "'stopped' status is deprecated for training jobs, use 'finished' instead.",
                DeprecationWarning,
            )
        if self.status == status or (
            (self.type == "headless" or self.type == "training")
            and status == "finished"
            and self.status == "stopped"
        ):
            return
        POLL_INTERVAL = 5
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
            if self.status == status or (
                (self.type == "headless" or self.type == "training")
                and status == "finished"
                and self.status == "stopped"
            ):
                return self
            elif self.status == "failed":
                raise JobError(self.status, self)
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
