import json
import asyncio
import math
import logging
from datetime import datetime

from .exceptions import JobError


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
        worker_count=1,
        worker_commands=[],
        environment=dict(type="DEEPLEARNING_PY37"),
        data=dict(datasets=[]),
        model=dict(),
        vpn=dict(net_prefix_type_id=1),
        **kwargs,
    ):
        gpu_type_task = asyncio.create_task(self.trainml.gpu_types.get())
        my_datasets_task = asyncio.create_task(self.trainml.datasets.list())
        public_datasets_task = asyncio.create_task(self.trainml.datasets.list_public())

        gpu_types, my_datasets, public_datasets = await asyncio.gather(
            gpu_type_task, my_datasets_task, public_datasets_task
        )

        selected_gpu_type = next(
            (g for g in gpu_types if g.name == gpu_type or g.id == gpu_type),
            None,
        )
        if not selected_gpu_type:
            raise ValueError("GPU Type Not Found")

        datasets = []
        for dataset in data.get("datasets"):
            if "id" in dataset.keys():
                datasets.append(
                    dict(dataset_uuid=dataset.get("id"), type=dataset.get("type"))
                )
            elif "name" in dataset.keys():
                if dataset.get("type") == "existing":
                    selected_dataset = next(
                        (d for d in my_datasets if d.name == dataset.get("name")),
                        None,
                    )
                    if not selected_dataset:
                        raise ValueError(f"Dataset {dataset} Not Found")
                    datasets.append(
                        dict(dataset_uuid=selected_dataset.id, type=dataset.get("type"))
                    )
                elif dataset.get("type") == "public":
                    selected_dataset = next(
                        (d for d in public_datasets if d.name == dataset.get("name")),
                        None,
                    )
                    if not selected_dataset:
                        raise ValueError(f"Dataset {dataset} Not Found")
                        datasets.append(
                            dict(
                                dataset_uuid=selected_dataset.id,
                                type=dataset.get("type"),
                            )
                        )
                else:
                    raise ValueError(
                        "Invalid dataset specification, 'type' must be in ['existing','public']"
                    )
            else:
                raise ValueError(
                    "Invalid dataset specification, either 'id' or 'name' must be provided"
                )

        config = dict(
            name=name,
            type=type,
            resources=dict(
                gpu_type_id=selected_gpu_type.id,
                gpu_count=gpu_count,
                disk_size=disk_size,
            ),
            worker_count=worker_count,
            worker_commands=worker_commands,
            environment=environment,
            data=data,
            model=model,
            vpn=vpn,
            source_job_uuid=kwargs.get("source_job_uuid"),
        )
        payload = {
            k: v for k, v in config.items() if v or k in ["worker_commands", "model"]
        }
        logging.info(f"Creating Job {name}")
        logging.debug(f"Job payload: {payload}")
        resp = await self.trainml._query("/job", "POST", None, payload)
        job = Job(self.trainml, **resp)
        logging.info(f"Created Job {name} with id {job.id}")
        if kwargs.get("wait"):
            await job.attach()
            job = await self.get(job.id)
        return job

    def remove(self, id):
        self.trainml._query(f"/job/{id}", "DELETE", dict(force=True))


class Job:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._job = kwargs
        self._id = self._job.get("id", self._job.get("job_uuid"))
        self._name = self._job.get("name")
        self._status = self._job.get("status")
        self._type = self._job.get("type")
        self._workers = self._job.get("workers")

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

    def __str__(self):
        return json.dumps({k: v for k, v in self._job.items()})

    def __repr__(self):
        return f"Job( trainml , {self._job.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def start(self):
        await self.trainml._query(
            f"/job/{self._id}", "PATCH", None, dict(command="start")
        )

    async def stop(self):
        await self.trainml._query(
            f"/job/{self._id}", "PATCH", None, dict(command="stop")
        )

    async def get_connection_utility_url(self):
        resp = await self.trainml._query(f"/job/{self._id}/download", "GET")
        return resp

    async def remove(self):
        await self.trainml._query(f"/job/{self._id}", "DELETE")

    async def refresh(self):
        resp = await self.trainml._query(f"/job/{self.id}", "GET")
        self.__init__(self.trainml, **resp)
        return self

    async def attach(self):
        worker_numbers = {
            w.get("job_worker_uuid"): ind + 1 for ind, w in enumerate(self._workers)
        }

        def msg_handler(msg):
            data = json.loads(msg.data)
            if data.get("type") == "subscription":
                timestamp = datetime.fromtimestamp(int(data.get("time")) / 1000)
                if len(self._workers) > 1:
                    print(
                        f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: Worker {worker_numbers.get(data.get('stream'))} - {data.get('msg').rstrip()}"
                    )
                else:
                    print(
                        f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: {data.get('msg').rstrip()}"
                    )

        await self.trainml._ws_subscribe("job", self.id, msg_handler)

    async def copy(self, name, **kwargs):
        logging.debug(f"copy request - name: {name} ; kwargs: {kwargs}")
        if self.type != "interactive":
            raise TypeError("Only interactive job types can be copied")

        job = await self.trainml.jobs.create(
            name,
            type=kwargs.get("type") or self.type,
            gpu_type=kwargs.get("gpu_type")
            or self._job.get("resources").get("gpu_type_id"),
            gpu_count=kwargs.get("gpu_count")
            or self._job.get("resources").get("gpu_count"),
            disk_size=kwargs.get("disk_size")
            or self._job.get("resources").get("disk_size"),
            worker_count=kwargs.get("worker_count") or len(self._workers),
            worker_commands=kwargs.get("worker_commands"),
            environment=kwargs.get("environment") or self._job.get("environment"),
            data=kwargs.get("data") or self._job.get("data"),
            vpn=kwargs.get("vpn") or self._job.get("vpn"),
            source_job_uuid=self.id,
            wait=kwargs.get("wait"),
        )
        logging.debug(f"copy result: {job}")
        return job

    async def waitFor(self, status, timeout=300):
        valid_statuses = ["running", "stopped", "archived"]
        if not status in valid_statuses:
            raise ValueError(
                f"Invalid waitFor status {status}.  Valid statuses are: {valid_statuses}"
            )
        if self.status == status:
            return
        POLL_INTERVAL = 5
        retry_count = math.ceil(timeout / POLL_INTERVAL)
        count = 0
        while count < retry_count:
            await asyncio.sleep(POLL_INTERVAL)
            await self.refresh()
            if self.status == status:
                return self
            elif not self and status == "archived":
                return
            elif self.status == "failed":
                raise JobError(self.status, self)
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TimeoutError(f"Timeout waiting for {status}")
