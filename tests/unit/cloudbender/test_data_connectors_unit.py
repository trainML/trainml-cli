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
        query_mock = AsyncMock(return_value=api_response)
        with patch.object(mock_trainml, "_query", query_mock):
            await data_connectors.get("1234", "5687", "91011")
        query_mock.assert_called_once_with(
            "/provider/1234/region/5687/data_connector/91011", "GET", {}
        )

    @mark.asyncio
    async def test_list_data_connectors(
        self,
        data_connectors,
        mock_trainml,
    ):
        recorded = []

        async def fake_query(path, method, params=None, **_kwargs):
            recorded.append((path, method, params))
            return []

        with patch.object(mock_trainml, "_query", fake_query):
            await data_connectors.list("1234", "5687")
        assert recorded == [
            ("/provider/1234/region/5687/data_connector", "GET", {}),
        ]

    @mark.asyncio
    async def test_remove_data_connector(
        self,
        data_connectors,
        mock_trainml,
    ):
        api_response = dict()
        query_mock = AsyncMock(return_value=api_response)
        with patch.object(mock_trainml, "_query", query_mock):
            await data_connectors.remove("1234", "4567", "8910")
        query_mock.assert_called_once_with(
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

        query_mock = AsyncMock(return_value=api_response)
        with patch.object(mock_trainml, "_query", query_mock):
            response = await data_connectors.create(**requested_config)
        query_mock.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/data_connector",
            "POST",
            None,
            expected_payload,
        )
        assert response.id == "connector-id-1"

    @mark.asyncio
    async def test_edit_data_connector(self, data_connectors, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            region_uuid="region-id-1",
            id="connector-id-1",
            host="10.0.0.20",
            port=9443,
        )
        expected_payload = dict(
            host="10.0.0.20",
            port=9443,
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "connector_id": "connector-id-1",
            "name": "On-Prem DataConnector",
            "type": "custom",
            "host": "10.0.0.20",
            "port": 9443,
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        query_mock = AsyncMock(return_value=api_response)
        with patch.object(mock_trainml, "_query", query_mock):
            response = await data_connectors.edit(**requested_config)
        query_mock.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/"
            "data_connector/connector-id-1",
            "PATCH",
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

    def test_data_connector_connection_properties(self, mock_trainml):
        data_connector = specimen.DataConnector(
            mock_trainml,
            provider_uuid="1",
            region_uuid="a",
            connector_id="x",
            name="HTTP Data Connector",
            type="http",
            host="10.0.0.10",
            port=8443,
            auth_type="api_key",
            client_id="client-1",
            token_url="https://10.0.0.10:8080/oauth/token",
            scopes="read write",
            database="ml_data",
            protocol="tcp",
            port_range="8000:8099",
            cidr="192.168.0.0/24",
            ca_certificate="CA_CERT_PEM",
            client_certificate="CLIENT_CERT_PEM",
            tls=False,
            authorization_header="X-Api-Key",
            authorization_prefix="Token ",
        )
        assert data_connector.host == "10.0.0.10"
        assert data_connector.port == 8443
        assert data_connector.auth_type == "api_key"
        assert data_connector.client_id == "client-1"
        assert data_connector.token_url == ("https://10.0.0.10:8080/oauth/token")
        assert data_connector.scopes == "read write"
        assert data_connector.database == "ml_data"
        assert data_connector.protocol == "tcp"
        assert data_connector.port_range == "8000:8099"
        assert data_connector.cidr == "192.168.0.0/24"
        assert data_connector.ca_certificate == "CA_CERT_PEM"
        assert data_connector.client_certificate == "CLIENT_CERT_PEM"
        assert data_connector.tls is False
        assert data_connector.authorization_header == "X-Api-Key"
        assert data_connector.authorization_prefix == "Token "

    def test_data_connector_properties_absent(self, mock_trainml):
        data_connector = specimen.DataConnector(mock_trainml)
        assert data_connector.auth_type is None
        assert data_connector.host is None
        assert data_connector.port is None
        assert data_connector.protocol is None
        assert data_connector.port_range is None
        assert data_connector.cidr is None
        assert data_connector.database is None
        assert data_connector.client_id is None
        assert data_connector.token_url is None
        assert data_connector.scopes is None
        assert data_connector.ca_certificate is None
        assert data_connector.client_certificate is None
        assert data_connector.tls is None
        assert data_connector.authorization_header is None
        assert data_connector.authorization_prefix is None

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
        query_mock = AsyncMock(return_value=api_response)
        with patch.object(mock_trainml, "_query", query_mock):
            await data_connector.remove()
        query_mock.assert_called_once_with(
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
        query_mock = AsyncMock(return_value=api_response)
        with patch.object(mock_trainml, "_query", query_mock):
            response = await data_connector.refresh()
        query_mock.assert_called_once_with(
            f"/provider/1/region/a/data_connector/x", "GET"
        )
        assert data_connector.id == "connector-id-1"
        assert response.id == "connector-id-1"

    @mark.asyncio
    async def test_data_connector_edit(self, data_connector, mock_trainml):
        api_response = {
            "provider_uuid": "1",
            "region_uuid": "a",
            "connector_id": "x",
            "name": "On-Prem Data Connector",
            "type": "custom",
            "cidr": "192.168.0.50/32",
            "port": "443",
            "protocol": "tcp",
            "port_range": "8000:8099",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        query_mock = AsyncMock(return_value=api_response)
        with patch.object(mock_trainml, "_query", query_mock):
            response = await data_connector.edit(port_range="8000:8099")
        query_mock.assert_called_once_with(
            "/provider/1/region/a/data_connector/x",
            "PATCH",
            None,
            dict(port_range="8000:8099"),
        )
        assert data_connector.id == "x"
        assert response.id == "x"
        assert data_connector.port_range == "8000:8099"
