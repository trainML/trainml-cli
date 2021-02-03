import json
import logging
import math
import asyncio
from datetime import datetime

from .exceptions import DatasetError, ApiError


class Datasets(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id):
        resp = await self.trainml._query(f"/dataset/pub/{id}", "GET")
        return Dataset(self.trainml, **resp)

    async def list(self):
        resp = await self.trainml._query(f"/dataset/pub", "GET")
        datasets = [Dataset(self.trainml, **dataset) for dataset in resp]
        return datasets

    async def list_public(self):
        resp = await self.trainml._query(f"/dataset/public", "GET")
        datasets = [Dataset(self.trainml, **dataset) for dataset in resp]
        return datasets

    async def create(self, name, source_type, source_uri, **kwargs):
        if kwargs.get("provider") and kwargs.get("provider") != "trainml":
            if not kwargs.get("disk_size"):
                raise AttributeError(
                    "'disk_size' attribute required for non-trainML providers"
                )
        data = dict(
            name=name,
            source_type=source_type,
            source_uri=source_uri,
            source_options=kwargs.get("source_options"),
            provider=kwargs.get("provider"),
            disk_size=kwargs.get("disk_size"),
        )
        payload = {k: v for k, v in data.items() if v}
        logging.info(f"Creating Dataset {name}")
        resp = await self.trainml._query("/dataset/pub", "POST", None, payload)
        dataset = Dataset(self.trainml, **resp)
        logging.info(f"Created Dataset {name} with id {dataset.id}")

        if kwargs.get("wait"):
            await dataset.attach()
            dataset = await self.get(dataset.id)
        return dataset

    async def remove(self, id):
        await self.trainml._query(f"/dataset/pub/{id}", "DELETE")


class Dataset:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._dataset = kwargs
        self._id = self._dataset.get("id", self._dataset.get("dataset_uuid"))
        self._status = self._dataset.get("status")
        self._name = self._dataset.get("name")

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> str:
        return self._status

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return json.dumps({k: v for k, v in self._dataset.items()})

    def __repr__(self):
        return f"Dataset( trainml , {self._dataset.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def get_connection_utility_url(self):
        resp = await self.trainml._query(f"/dataset/pub/{self._id}/download", "GET")
        return resp

    async def remove(self):
        await self.trainml._query(f"/dataset/pub/{self._id}", "DELETE")

    async def attach(self):
        def msg_handler(msg):
            data = json.loads(msg.data)
            if data.get("type") == "subscription":
                timestamp = datetime.fromtimestamp(int(data.get("time")) / 1000)
                print(
                    f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: {data.get('msg').rstrip()}"
                )

        await self.trainml._ws_subscribe("dataset", self.id, msg_handler)

    async def refresh(self):
        resp = await self.trainml._query(f"/dataset/pub/{self.id}", "GET")
        self.__init__(self.trainml, **resp)
        return self

    async def waitFor(self, status, timeout=300):
        valid_statuses = ["ready", "archived"]
        if not status in valid_statuses:
            raise ValueError(
                f"Invalid waitFor status {status}.  Valid statuses are: {valid_statuses}"
            )
        if self.status == status:
            return
        POLL_INTERVAL = 5
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
            if self.status == status:
                return self
            elif self.status == "failed":
                raise DatasetError(self.status, self)
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TimeoutError(f"Timeout waiting for {status}")