import json
import logging
import asyncio
import math

from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)


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

    async def edit(self, provider_uuid, region_uuid, id, **kwargs):
        logging.info(f"Editing Data Connector {id}")
        payload = {k: v for k, v in kwargs.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/data_connector/{id}",
            "PATCH",
            None,
            payload,
        )
        return DataConnector(self.trainml, **resp)

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
        self._auth_type = self._data_connector.get("auth_type")
        self._host = self._data_connector.get("host")
        self._port = self._data_connector.get("port")
        self._protocol = self._data_connector.get("protocol")
        self._port_range = self._data_connector.get("port_range")
        self._cidr = self._data_connector.get("cidr")
        self._database = self._data_connector.get("database")
        self._client_id = self._data_connector.get("client_id")
        self._token_url = self._data_connector.get("token_url")
        self._scopes = self._data_connector.get("scopes")
        self._ca_certificate = self._data_connector.get("ca_certificate")
        self._client_certificate = self._data_connector.get("client_certificate")
        self._tls = self._data_connector.get("tls")
        self._authorization_header = self._data_connector.get("authorization_header")
        self._authorization_prefix = self._data_connector.get("authorization_prefix")

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

    @property
    def auth_type(self) -> str:
        return self._auth_type

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self):
        return self._port

    @property
    def protocol(self) -> str:
        return self._protocol

    @property
    def port_range(self) -> str:
        return self._port_range

    @property
    def cidr(self) -> str:
        return self._cidr

    @property
    def database(self) -> str:
        return self._database

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def token_url(self) -> str:
        return self._token_url

    @property
    def scopes(self) -> str:
        return self._scopes

    @property
    def ca_certificate(self) -> str:
        return self._ca_certificate

    @property
    def client_certificate(self) -> str:
        return self._client_certificate

    @property
    def tls(self) -> bool:
        return self._tls

    @property
    def authorization_header(self) -> str:
        return self._authorization_header

    @property
    def authorization_prefix(self) -> str:
        return self._authorization_prefix

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

    async def edit(self, **kwargs):
        logging.info(f"Editing Data Connector {self._id}")
        payload = {k: v for k, v in kwargs.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/data_connector/{self._id}",
            "PATCH",
            None,
            payload,
        )
        self.__init__(self.trainml, **resp)
        return self
