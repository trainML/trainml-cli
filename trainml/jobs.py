import json
from datetime import datetime


class Jobs(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id):
        resp = await self.trainml._query(f"/job/{id}", "GET")
        return Job(self.trainml, **resp)

    async def create(
        self,
        name,
        type,
        gpu_type_id,
        gpu_count,
        disk_size,
        worker_count=1,
        worker_commands=[],
        environment=dict(type="DEEPLEARNING_PY37"),
        data=dict(datasets=[]),
        model=None,
        vpn=dict(net_prefix_type_id=1),
        **kwargs,
    ):
        payload = dict(
            name=name,
            type=type,
            resources=dict(
                gpu_type_id=gpu_type_id, gpu_count=gpu_count, disk_size=disk_size
            ),
            worker_count=worker_count,
            worker_commands=worker_commands,
            environment=environment,
            data=data,
            model=model,
            vpn=vpn,
        )
        resp = await self.trainml._query("/job", "POST", None, payload)
        job = Job(self.trainml, **resp)
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
        self._status = self._job.get("status")
        self._workers = self._job.get("workers")

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> str:
        return self._status

    async def start(self):
        await self.trainml._query(
            f"/job/{self._id}", "PATCH", None, dict(command="start")
        )

    async def stop(self):
        await self.trainml._query(
            f"/job/{self._id}", "PATCH", None, dict(command="stop")
        )

    async def destroy(self):
        await self.trainml._query(f"/job/{self._id}", "DELETE")

    async def attach(self):
        worker_numbers = {
            w.get("job_worker_uuid"): ind + 1 for ind, w in enumerate(self._workers)
        }
        print(worker_numbers)

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