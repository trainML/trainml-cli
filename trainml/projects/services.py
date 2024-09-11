import json
import logging


class ProjectServices(object):
    def __init__(self, trainml, project_id):
        self.trainml = trainml
        self.project_id = project_id

    async def get(self, id, **kwargs):
        resp = await self.trainml._query(
            f"/project/{self.project_id}/services/{id}", "GET", kwargs
        )
        return ProjectService(self.trainml, **resp)

    async def list(self, **kwargs):
        resp = await self.trainml._query(
            f"/project/{self.project_id}/services", "GET", kwargs
        )
        services = [ProjectService(self.trainml, **service) for service in resp]
        return services

    async def refresh(self):
        await self.trainml._query(f"/project/{self.project_id}/services", "PATCH")


class ProjectService:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._entity = kwargs
        self._id = self._entity.get("id")
        self._project_uuid = self._entity.get("project_uuid")
        self._name = self._entity.get("name")
        self._hostname = self._entity.get("hostname")
        self._public = self._entity.get("public")
        self._region_uuid = self._entity.get("region_uuid")

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
    def hostname(self) -> str:
        return self._hostname

    @property
    def public(self) -> bool:
        return self._public

    @property
    def region_uuid(self) -> str:
        return self._region_uuid

    def __str__(self):
        return json.dumps({k: v for k, v in self._entity.items()})

    def __repr__(self):
        return f"ProjectService( trainml , **{self._entity.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def enable(self):
        await self.trainml._query(
            f"/project/{self._project_uuid}/services/{self._id}/enable", "PATCH"
        )

    async def disable(self):
        await self.trainml._query(
            f"/project/{self._project_uuid}/services/{self._id}/disable", "PATCH"
        )
