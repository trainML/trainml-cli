import json
import logging
from datetime import datetime


class Projects(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id):
        resp = await self.trainml._query(f"/project/{id}", "GET")
        return Project(self.trainml, **resp)

    async def list(self):
        resp = await self.trainml._query(f"/project", "GET")
        projects = [Project(self.trainml, **project) for project in resp]
        return projects

    async def create(self, name, copy_keys=False, **kwargs):
        data = dict(
            name=name,
            copy_keys=copy_keys,
        )
        payload = {k: v for k, v in data.items() if v or k in ["copy_keys"]}
        logging.info(f"Creating Project {name}")
        resp = await self.trainml._query("/project", "POST", None, payload)
        project = Project(self.trainml, **resp)
        logging.info(f"Created Project {name} with id {project.id}")

        return project

    async def remove(self, id):
        await self.trainml._query(f"/project/{id}", "DELETE")


class Project:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._project = kwargs
        self._id = self._project.get("id")
        self._name = self._project.get("name")
        self._is_owner = self._project.get("owner")
        self._owner_name = self._project.get("owner_name")

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_owner(self) -> bool:
        return self._is_owner

    @property
    def owner_name(self) -> str:
        return self._owner_name

    def __str__(self):
        return json.dumps({k: v for k, v in self._project.items()})

    def __repr__(self):
        return f"Project( trainml , **{self._project.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(f"/project/{self._id}", "DELETE")
