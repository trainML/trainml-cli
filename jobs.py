class Jobs(object):
    def __init__(self, trainml):
        self.trainml = trainml


class Job:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._job = kwargs
        self._id = self._job.get("id", self._job.get("job_uuid"))

    @property
    def id(self) -> str:
        return self._id