import json
import logging
import asyncio
import math

from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
    RegionError,
)


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

    async def wait_for(self, status, timeout=300):
        if self.status == status:
            return
        valid_statuses = ["healthy", "offline", "archived"]
        if not status in valid_statuses:
            raise SpecificationError(
                "status",
                f"Invalid wait_for status {status}.  Valid statuses are: {valid_statuses}",
            )
        MAX_TIMEOUT = 24 * 60 * 60
        if timeout > MAX_TIMEOUT:
            raise SpecificationError(
                "timeout",
                f"timeout must be less than {MAX_TIMEOUT} seconds.",
            )

        POLL_INTERVAL_MIN = 5
        POLL_INTERVAL_MAX = 60
        POLL_INTERVAL = max(min(timeout / 60, POLL_INTERVAL_MAX), POLL_INTERVAL_MIN)
        retry_count = math.ceil(timeout / POLL_INTERVAL)
        count = 0
        while count < retry_count:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                await self.refresh()
            except ApiError as e:
                if status == "archived" and e.status == 404:
                    return
                raise e
            if self.status in ["errored", "failed"]:
                raise RegionError(self.status, self)
            if self.status == status:
                return self
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
