import json
import logging
from .datastores import ProjectDatastores
from .data_connectors import ProjectDataConnectors
from .services import ProjectServices
from .credentials import ProjectCredentials
from .secrets import ProjectSecrets


class Projects(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id, **kwargs):
        resp = await self.trainml._query(f"/project/{id}", "GET", kwargs)
        return Project(self.trainml, **resp)

    async def get_current(self, **kwargs):
        resp = await self.trainml._query(
            f"/project/{self.trainml.project}", "GET", kwargs
        )
        return Project(self.trainml, **resp)

    async def list(self, **kwargs):
        resp = await self.trainml._query(f"/project", "GET", kwargs)
        projects = [Project(self.trainml, **project) for project in resp]
        return projects

    async def create(self, name, copy_credentials=False, **kwargs):
        data = dict(
            name=name,
            copy_credentials=copy_credentials,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        logging.info(f"Creating Project {name}")
        resp = await self.trainml._query("/project", "POST", None, payload)
        project = Project(self.trainml, **resp)
        logging.info(f"Created Project {name} with id {project.id}")

        return project

    async def remove(self, id, **kwargs):
        await self.trainml._query(f"/project/{id}", "DELETE", kwargs)


class Project:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._entity = kwargs
        self._id = self._entity.get("id")
        self._name = self._entity.get("name")
        self._is_owner = self._entity.get("owner")
        self._owner_name = self._entity.get("owner_name")
        self.datastores = ProjectDatastores(self.trainml, self._id)
        self.data_connectors = ProjectDataConnectors(self.trainml, self._id)
        self.services = ProjectServices(self.trainml, self._id)
        self.credentials = ProjectCredentials(self.trainml, self._id)
        self.secrets = ProjectSecrets(self.trainml, self._id)

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
        return json.dumps({k: v for k, v in self._entity.items()})

    def __repr__(self):
        return f"Project( trainml , **{self._entity.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(f"/project/{self._id}", "DELETE")
