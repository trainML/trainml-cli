import json
import logging
from datetime import datetime
from dateutil import parser, tz


class ProjectSecrets(object):
    def __init__(self, trainml, project_id):
        self.trainml = trainml
        self.project_id = project_id

    async def list(self, **kwargs):
        resp = await self.trainml._query(
            f"/project/{self.project_id}/secrets", "GET", kwargs
        )
        secrets = [ProjectSecret(self.trainml, **service) for service in resp]
        return secrets

    async def put(self, name, value, **kwargs):
        data = dict(value=value)
        payload = {k: v for k, v in data.items() if v is not None}
        logging.info(f"Creating Project Secret {name}")
        resp = await self.trainml._query(
            f"/project/{self.project_id}/secret/{name}", "PUT", None, payload
        )
        secret = ProjectSecret(self.trainml, **resp)
        logging.info(f"Created Project Secret {name} in project {self.project_id}")
        return secret

    async def remove(self, name, **kwargs):
        await self.trainml._query(
            f"/project/{self.project_id}/secret/{name}", "DELETE", kwargs
        )


class ProjectSecret:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._entity = kwargs
        self._name = self._entity.get("name")
        self._project_uuid = self._entity.get("project_uuid")
        self._created_by = self._entity.get("created_by")
        self._updated_at = self._entity.get("updatedAt")

    @property
    def name(self) -> str:
        return self._name

    @property
    def project_uuid(self) -> str:
        return self._project_uuid

    @property
    def created_by(self) -> str:
        return self._created_by

    @property
    def updated_at(self) -> datetime:
        timestamp = parser.isoparse(self._updated_at)
        timezone = tz.tzlocal()
        return timestamp.astimezone(timezone)

    def __str__(self):
        return json.dumps({k: v for k, v in self._entity.items()})

    def __repr__(self):
        return f"ProjectSecret( trainml , **{self._entity.__repr__()})"

    def __bool__(self):
        return bool(self._name)
