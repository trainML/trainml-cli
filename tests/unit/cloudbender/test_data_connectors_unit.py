import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.data_connectors as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.data_connectors]


@fixture
def data_connectors(mock_trainml):
    yield specimen.DataConnectors(mock_trainml)


@fixture
def data_connector(mock_trainml):
    yield specimen.DataConnector(
        mock_trainml,
        provider_uuid="1",
        region_uuid="a",
        connector_id="x",
        name="On-Prem Data Connector",
        type="custom",
        cidr="192.168.0.50/32",
        port="443",
        protocol="tcp",
    )


class RegionsTests:
    @mark.asyncio
    async def test_get_data_connector(
        self,
        data_connectors,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await data_connectors.get("1234", "5687", "91011")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/data_connector/91011", "GET", {}
        )

    @mark.asyncio
    async def test_list_data_connectors(
        self,
        data_connectors,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await data_connectors.list("1234", "5687")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/data_connector", "GET", {}
        )

    @mark.asyncio
    async def test_remove_data_connector(
        self,
        data_connectors,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await data_connectors.remove("1234", "4567", "8910")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/4567/data_connector/8910", "DELETE", {}
        )

    @mark.asyncio
    async def test_create_data_connector(self, data_connectors, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            region_uuid="region-id-1",
            name="On-Prem DataConnector",
            type="custom",
            cidr="192.168.0.50/32",
            port="443",
            protocol="tcp",
        )
        expected_payload = dict(
            name="On-Prem DataConnector",
            type="custom",
            cidr="192.168.0.50/32",
            port="443",
            protocol="tcp",
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "connector_id": "connector-id-1",
            "name": "On-Prem DataConnector",
            "type": "custom",
            "cidr": "192.168.0.50/32",
            "port": "443",
            "protocol": "tcp",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await data_connectors.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/data_connector",
            "POST",
            None,
            expected_payload,
        )
        assert response.id == "connector-id-1"


class DataConnectorTests:
    def test_data_connector_properties(self, data_connector):
        assert isinstance(data_connector.id, str)
        assert isinstance(data_connector.provider_uuid, str)
        assert isinstance(data_connector.region_uuid, str)
        assert isinstance(data_connector.type, str)
        assert isinstance(data_connector.name, str)

    def test_data_connector_str(self, data_connector):
        string = str(data_connector)
        regex = r"^{.*\"connector_id\": \"" + data_connector.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_data_connector_repr(self, data_connector):
        string = repr(data_connector)
        regex = (
            r"^DataConnector\( trainml , \*\*{.*'connector_id': '"
            + data_connector.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_data_connector_bool(self, data_connector, mock_trainml):
        empty_data_connector = specimen.DataConnector(mock_trainml)
        assert bool(data_connector)
        assert not bool(empty_data_connector)

    @mark.asyncio
    async def test_data_connector_remove(self, data_connector, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await data_connector.remove()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/data_connector/x", "DELETE"
        )

    @mark.asyncio
    async def test_data_connector_refresh(self, data_connector, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "connector_id": "connector-id-1",
            "name": "On-Prem DataConnector",
            "type": "custom",
            "cidr": "192.168.0.50/32",
            "port": "443",
            "protocol": "tcp",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await data_connector.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/provider/1/region/a/data_connector/x", "GET"
        )
        assert data_connector.id == "connector-id-1"
        assert response.id == "connector-id-1"
