import json
import logging
from datetime import datetime
from dateutil import parser, tz
from typing import Literal


class ProjectAuthorizers(object):
    def __init__(self, trainml, project_id):
        self.trainml = trainml
        self.project_id = project_id

    async def get(self, name_or_id, **kwargs):
        resp = await self.trainml._query(
            f"/project/{self.project_id}/authorizers/{name_or_id}", "GET", kwargs
        )
        return ProjectAuthorizer(self.trainml, **resp)

    async def list(self, **kwargs):
        resp = await self.trainml._query(
            f"/project/{self.project_id}/authorizers", "GET", kwargs
        )
        authorizers = [
            ProjectAuthorizer(self.trainml, **authorizer) for authorizer in resp
        ]
        return authorizers

    async def create(
        self, name, type: Literal["api_key", "oidc"], keys=None, oidc=None, **kwargs
    ):
        data = dict(
            name=name,
            type=type,
            keys=keys,
            oidc=oidc,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        logging.info(f"Creating Project Authorizer {name}")
        resp = await self.trainml._query(
            f"/project/{self.project_id}/authorizers", "POST", kwargs, payload
        )
        authorizer = ProjectAuthorizer(self.trainml, **resp)
        logging.info(
            f"Created Project Authorizer {name} with id {authorizer.id} in project {self.project_id}"
        )
        # Note: for api_key authorizers the returned keys are the encrypted
        # values as stored, not the plaintext keys that were submitted.
        return authorizer

    async def remove(self, authorizer_id, **kwargs):
        await self.trainml._query(
            f"/project/{self.project_id}/authorizers/{authorizer_id}", "DELETE", kwargs
        )


class ProjectAuthorizer:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._entity = kwargs
        self._id = self._entity.get("id")
        self._project_uuid = self._entity.get("project_uuid")
        self._name = self._entity.get("name")
        self._type = self._entity.get("type")
        self._keys = self._entity.get("keys")
        self._oidc = self._entity.get("oidc")
        self._created_at = self._entity.get("createdAt")
        self._updated_at = self._entity.get("updatedAt")

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
    def keys(self) -> list:
        return self._keys

    @property
    def oidc(self) -> dict:
        return self._oidc

    @staticmethod
    def _to_local_datetime(value) -> datetime:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            timestamp = datetime.fromtimestamp(value / 1000)
        else:
            timestamp = parser.isoparse(str(value))
        timezone = tz.tzlocal()
        return timestamp.astimezone(timezone)

    @property
    def created_at(self) -> datetime:
        return self._to_local_datetime(self._created_at)

    @property
    def updated_at(self) -> datetime:
        return self._to_local_datetime(self._updated_at)

    async def edit(self, name=None, keys=None, oidc=None, **kwargs):
        data = dict(
            name=name,
            keys=keys,
            oidc=oidc,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        logging.info(f"Editing Project Authorizer {self._name}")
        resp = await self.trainml._query(
            f"/project/{self._project_uuid}/authorizers/{self._id}",
            "PATCH",
            kwargs,
            payload,
        )
        updated = ProjectAuthorizer(self.trainml, **resp)
        logging.info(f"Edited Project Authorizer {updated.name} with id {updated.id}")
        return updated

    def __str__(self):
        return json.dumps({k: v for k, v in self._entity.items()})

    def __repr__(self):
        return f"ProjectAuthorizer( trainml , **{self._entity.__repr__()})"

    def __bool__(self):
        return bool(self._id)
