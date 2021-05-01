import json
import logging
import math
import asyncio
from datetime import datetime

from .exceptions import (
    ModelError,
    ApiError,
    SpecificationError,
    TrainMLException,
)
from .connections import Connection


class Models(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id):
        resp = await self.trainml._query(f"/model/pub/{id}", "GET")
        return Model(self.trainml, **resp)

    async def list(self):
        resp = await self.trainml._query(f"/model/pub", "GET")
        models = [Model(self.trainml, **model) for model in resp]
        return models

    async def create(self, name, source_type, source_uri, **kwargs):
        if kwargs.get("provider") and kwargs.get("provider") != "trainml":
            if not kwargs.get("disk_size"):
                raise SpecificationError(
                    "disk_size",
                    "'disk_size' attribute required for non-trainML providers",
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
        logging.info(f"Creating Model {name}")
        resp = await self.trainml._query("/model/pub", "POST", None, payload)
        model = Model(self.trainml, **resp)
        logging.info(f"Created Model {name} with id {model.id}")

        return model

    async def remove(self, id):
        await self.trainml._query(f"/model/pub/{id}", "DELETE")


class Model:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._model = kwargs
        self._id = self._model.get("id", self._model.get("model_uuid"))
        self._status = self._model.get("status")
        self._provider = self._model.get("provider")
        self._name = self._model.get("name")
        self._size = self._model.get("size")

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> str:
        return self._status

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    def __str__(self):
        return json.dumps({k: v for k, v in self._model.items()})

    def __repr__(self):
        return f"Model( trainml , **{self._model.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def get_log_url(self):
        resp = await self.trainml._query(f"/model/pub/{self._id}/logs", "GET")
        return resp

    async def get_details(self):
        resp = await self.trainml._query(
            f"/model/pub/{self._id}/details", "GET"
        )
        return resp

    async def get_connection_utility_url(self):
        resp = await self.trainml._query(
            f"/model/pub/{self._id}/download", "GET"
        )
        return resp

    def get_connection_details(self):
        if self._model.get("vpn"):
            details = dict(
                cidr=self._model.get("vpn").get("cidr"),
                ssh_port=self._model.get("vpn").get("client").get("ssh_port"),
                input_path=self._model.get("source_uri"),
                output_path=None,
            )
        else:
            details = dict()
        return details

    async def connect(self):
        connection = Connection(
            self.trainml, entity_type="model", id=self.id, entity=self
        )
        await connection.start()
        return connection.status

    async def disconnect(self):
        connection = Connection(
            self.trainml, entity_type="model", id=self.id, entity=self
        )
        await connection.stop()
        return connection.status

    async def remove(self):
        await self.trainml._query(f"/model/pub/{self._id}", "DELETE")

    def _get_msg_handler(self, msg_handler):
        def handler(msg):
            data = json.loads(msg.data)
            if data.get("type") == "subscription":
                if msg_handler:
                    msg_handler(data)
                else:
                    timestamp = datetime.fromtimestamp(
                        int(data.get("time")) / 1000
                    )
                    print(
                        f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: {data.get('msg').rstrip()}"
                    )

        return handler

    async def attach(self, msg_handler=None):
        await self.refresh()
        if self.status not in ["ready", "failed"]:
            await self.trainml._ws_subscribe(
                "model", self.id, self._get_msg_handler(msg_handler)
            )

    async def refresh(self):
        resp = await self.trainml._query(f"/model/pub/{self.id}", "GET")
        self.__init__(self.trainml, **resp)
        return self

    async def wait_for(self, status, timeout=300):
        valid_statuses = ["ready", "archived"]
        if not status in valid_statuses:
            raise SpecificationError(
                "status",
                f"Invalid wait_for status {status}.  Valid statuses are: {valid_statuses}",
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
                raise ModelError(self.status, self)
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
