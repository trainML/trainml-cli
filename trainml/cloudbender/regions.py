import json
import logging


class Regions(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, provider_uuid, id, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{id}", "GET", kwargs
        )
        return Region(self.trainml, **resp)

    async def list(self, provider_uuid, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region", "GET", kwargs
        )
        regions = [Region(self.trainml, **region) for region in resp]
        return regions

    async def create(self, provider_uuid, name, public, storage, **kwargs):
        logging.info(f"Creating Region {name}")
        data = dict(name=name, public=public, storage=storage, **kwargs)
        payload = payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region", "POST", None, payload
        )
        region = Region(self.trainml, **resp)
        logging.info(f"Created Region {name} with id {region.id}")
        return region

    async def remove(self, provider_uuid, id, **kwargs):
        await self.trainml._query(
            f"/provider/{provider_uuid}/region/{id}", "DELETE", kwargs
        )


class Region:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._region = kwargs
        self._id = self._region.get("region_uuid")
        self._provider_uuid = self._region.get("provider_uuid")
        self._type = self._region.get("provider_type")
        self._name = self._region.get("name")
        self._status = self._region.get("status")

    @property
    def id(self) -> str:
        return self._id

    @property
    def provider_uuid(self) -> str:
        return self._provider_uuid

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> str:
        return self._status

    def __str__(self):
        return json.dumps({k: v for k, v in self._region.items()})

    def __repr__(self):
        return f"Region( trainml , **{self._region.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._id}", "DELETE"
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self

    async def add_dataset(self, project_uuid, dataset_uuid, **kwargs):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._id}/dataset",
            "POST",
            None,
            dict(project_uuid=project_uuid, dataset_uuid=dataset_uuid),
        )

    async def add_model(self, project_uuid, model_uuid, **kwargs):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._id}/model",
            "POST",
            None,
            dict(project_uuid=project_uuid, model_uuid=model_uuid),
        )

    async def add_checkpoint(self, project_uuid, checkpoint_uuid, **kwargs):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._id}/checkpoint",
            "POST",
            None,
            dict(project_uuid=project_uuid, checkpoint_uuid=checkpoint_uuid),
        )
