import json
import logging


class DataConnectors(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, provider_uuid, region_uuid, id, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/data_connector/{id}",
            "GET",
            kwargs,
        )
        return DataConnector(self.trainml, **resp)

    async def list(self, provider_uuid, region_uuid, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/data_connector",
            "GET",
            kwargs,
        )
        data_connectors = [
            DataConnector(self.trainml, **data_connector) for data_connector in resp
        ]
        return data_connectors

    async def create(
        self,
        provider_uuid,
        region_uuid,
        name,
        type,
        **kwargs,
    ):
        logging.info(f"Creating Data Connector {name}")
        data = dict(
            name=name,
            type=type,
            **kwargs,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/data_connector",
            "POST",
            None,
            payload,
        )
        data_connector = DataConnector(self.trainml, **resp)
        logging.info(f"Created Data Connector {name} with id {data_connector.id}")
        return data_connector

    async def remove(self, provider_uuid, region_uuid, id, **kwargs):
        await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/data_connector/{id}",
            "DELETE",
            kwargs,
        )


class DataConnector:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._data_connector = kwargs
        self._id = self._data_connector.get("connector_id")
        self._provider_uuid = self._data_connector.get("provider_uuid")
        self._region_uuid = self._data_connector.get("region_uuid")
        self._type = self._data_connector.get("type")
        self._name = self._data_connector.get("name")

    @property
    def id(self) -> str:
        return self._id

    @property
    def provider_uuid(self) -> str:
        return self._provider_uuid

    @property
    def region_uuid(self) -> str:
        return self._region_uuid

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return json.dumps({k: v for k, v in self._data_connector.items()})

    def __repr__(self):
        return f"DataConnector( trainml , **{self._data_connector.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/data_connector/{self._id}",
            "DELETE",
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/data_connector/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self
