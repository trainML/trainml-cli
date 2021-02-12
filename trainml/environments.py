import json
from datetime import datetime


class Environments(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def list(self):
        resp = await self.trainml._query(f"/job/environments", "GET")
        environments = [
            Environment(self.trainml, **environment) for environment in resp
        ]
        return environments


class Environment:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._environment = kwargs
        self._id = self._environment.get("id")
        self._name = self._environment.get("name")
        self._py_version = self._environment.get("py_version")
        self._framework = self._environment.get("framework")
        self._version = self._environment.get("version")
        self._cuda_version = self._environment.get("cuda_version")

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def py_version(self) -> str:
        return self._py_version

    @property
    def framework(self) -> str:
        return self._framework

    @property
    def version(self) -> str:
        return self._version

    @property
    def cuda_version(self) -> str:
        return self._cuda_version

    def __str__(self):
        return json.dumps(
            {k: v for k, v in self._environment.items() if k not in ["image"]}
        )

    def __repr__(self):
        return f"Environment( trainml , **{self._environment.__repr__()})"
