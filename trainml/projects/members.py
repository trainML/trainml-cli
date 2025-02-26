import json
import logging
from typing import Literal

class ProjectMembers(object):
    def __init__(self, trainml, project_id):
        self.trainml = trainml
        self.project_id = project_id

    async def list(self, **kwargs):
        resp = await self.trainml._query(
            f"/project/{self.project_id}/access", "GET", kwargs
        )
        members = [ProjectMember(self.trainml, **member) for member in resp]
        return members
    
    async def add(self, email: str, job: Literal["all", "read"], dataset: Literal["all", "read"], model: Literal["all", "read"], checkpoint: Literal["all", "read"], volume: Literal["all", "read"],  **kwargs):
        data = dict(
            email=email,
            job=job,
            dataset=dataset,
            model=model,
            checkpoint=checkpoint,
            volume=volume,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/project/{self.project_id}/access", "POST",kwargs, payload)
        member = ProjectMember(self.trainml, **resp)
        logging.info(f"Added Project Member {email} to project {self.project_id}")
        return member

    
    async def remove(self, email, **kwargs):
        await self.trainml._query(
            f"/project/{self.project_id}/access", "DELETE", dict(**kwargs, email=email)
        )


class ProjectMember:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._entity = kwargs
        self._id = self._entity.get("email")
        self._project_uuid = self._entity.get("project_uuid")
        self._owner = self._entity.get("owner")
        self._job = self._entity.get("job")
        self._dataset = self._entity.get("dataset")
        self._model = self._entity.get("model")
        self._checkpoint = self._entity.get("checkpoint")
        self._volume = self._entity.get("volume")

    @property
    def id(self) -> str:
        return self._id

    @property
    def project_uuid(self) -> str:
        return self._project_uuid
    
    @property
    def email(self) -> str:
        return self._id

    @property
    def owner(self) -> bool:
        return self._owner

    @property
    def job(self) -> str:
        return self._job

    @property
    def dataset(self) -> str:
        return self._dataset
    
    @property
    def model(self) -> str:
        return self._model
    
    @property
    def checkpoint(self) -> str:
        return self._checkpoint
    
    @property
    def volume(self) -> str:
        return self._volume

    def __str__(self):
        return json.dumps({k: v for k, v in self._entity.items()})

    def __repr__(self):
        return f"ProjectMember( trainml , **{self._entity.__repr__()})"

    def __bool__(self):
        return bool(self._id)


