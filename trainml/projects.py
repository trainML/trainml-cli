import json
import logging


class Projects(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id, **kwargs):
        resp = await self.trainml._query(f"/project/{id}", "GET", kwargs)
        return Project(self.trainml, **resp)

    async def list(self, **kwargs):
        resp = await self.trainml._query(f"/project", "GET", kwargs)
        projects = [Project(self.trainml, **project) for project in resp]
        return projects

    async def create(self, name, copy_keys=False, **kwargs):
        data = dict(
            name=name,
            copy_keys=copy_keys,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        logging.info(f"Creating Project {name}")
        resp = await self.trainml._query("/project", "POST", None, payload)
        project = Project(self.trainml, **resp)
        logging.info(f"Created Project {name} with id {project.id}")

        return project

    async def remove(self, id, **kwargs):
        await self.trainml._query(f"/project/{id}", "DELETE", kwargs)


class ProjectDatastore:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._datastore = kwargs
        self._id = self._datastore.get("id")
        self._project_uuid = self._datastore.get("project_uuid")
        self._name = self._datastore.get("name")
        self._type = self._datastore.get("type")
        self._region_uuid = self._datastore.get("region_uuid")

    @property
    def id(self) -> str:
        return self._id

    @property
    def project_uuid(self) -> str:
        return self._project_uuid

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def region_uuid(self) -> str:
        return self._region_uuid

    def __str__(self):
        return json.dumps({k: v for k, v in self._datastore.items()})

    def __repr__(self):
        return f"ProjectDatastore( trainml , **{self._datastore.__repr__()})"

    def __bool__(self):
        return bool(self._id)


class ProjectReservation:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._reservation = kwargs
        self._id = self._reservation.get("id")
        self._project_uuid = self._reservation.get("project_uuid")
        self._name = self._reservation.get("name")
        self._type = self._reservation.get("type")
        self._hostname = self._reservation.get("hostname")
        self._resource = self._reservation.get("resource")
        self._region_uuid = self._reservation.get("region_uuid")

    @property
    def id(self) -> str:
        return self._id

    @property
    def project_uuid(self) -> str:
        return self._project_uuid

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @property
    def hostname(self) -> str:
        return self._hostname

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def region_uuid(self) -> str:
        return self._region_uuid

    def __str__(self):
        return json.dumps({k: v for k, v in self._reservation.items()})

    def __repr__(self):
        return (
            f"ProjectReservation( trainml , **{self._reservation.__repr__()})"
        )

    def __bool__(self):
        return bool(self._id)


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

    async def list_datastores(self):
        resp = await self.trainml._query(
            f"/project/{self._id}/datastores", "GET"
        )
        datastores = [
            ProjectDatastore(self.trainml, **datastore) for datastore in resp
        ]
        return datastores

    async def list_reservations(self):
        resp = await self.trainml._query(
            f"/project/{self._id}/reservations", "GET"
        )
        reservations = [
            ProjectReservation(self.trainml, **reservation)
            for reservation in resp
        ]
        return reservations

    async def refresh_datastores(self):
        await self.trainml._query(f"/project/{self._id}/datastores", "PATCH")

    async def refresh_reservations(self):
        await self.trainml._query(f"/project/{self._id}/reservations", "PATCH")
