class Jobs(object):
    def __init__(self, trainml):
        self.trainml = trainml

    def create(
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
        resp = self.trainml._query("/job", "POST", None, payload)
        job = Job(self.trainml, **resp)
        return job

    def remove(self, id):
        self.trainml._query(f"/job/{id}", "DELETE", dict(force=True))


class Job:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._job = kwargs
        self._id = self._job.get("id", self._job.get("job_uuid"))
        self._status = self._job.get("status")

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> str:
        return self._status

    def start(self):
        self.trainml._query(f"/job/{self._id}", "PATCH", None, dict(command="start"))

    def stop(self):
        self.trainml._query(f"/job/{self._id}", "PATCH", None, dict(command="stop"))

    def destroy(self):
        self.trainml._query(f"/job/{self._id}", "DELETE")