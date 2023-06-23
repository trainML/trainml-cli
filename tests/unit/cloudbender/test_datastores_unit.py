import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.datastores as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.datastores]


@fixture
def datastores(mock_trainml):
    yield specimen.Datastores(mock_trainml)


@fixture
def datastore(mock_trainml):
    yield specimen.Datastore(
        mock_trainml,
        provider_uuid="1",
        region_uuid="a",
        store_id="x",
        name="On-Prem Datastore",
        type="nfs",
        uri="192.168.0.50",
        root="/exports",
    )


class RegionsTests:
    @mark.asyncio
    async def test_get_datastore(
        self,
        datastores,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datastores.get("1234", "5687", "91011")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/datastore/91011", "GET", {}
        )

    @mark.asyncio
    async def test_list_datastores(
        self,
        datastores,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datastores.list("1234", "5687")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/datastore", "GET", {}
        )

    @mark.asyncio
    async def test_remove_datastore(
        self,
        datastores,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datastores.remove("1234", "4567", "8910")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/4567/datastore/8910", "DELETE", {}
        )

    @mark.asyncio
    async def test_create_datastore(self, datastores, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            region_uuid="region-id-1",
            name="On-Prem Datastore",
            type="nfs",
            uri="192.168.0.50",
            root="/exports",
        )
        expected_payload = dict(
            name="On-Prem Datastore",
            type="nfs",
            uri="192.168.0.50",
            root="/exports",
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "store_id": "store-id-1",
            "name": "On-Prem Datastore",
            "type": "nfs",
            "uri": "192.168.0.50",
            "root": "/exports",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await datastores.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/datastore",
            "POST",
            None,
            expected_payload,
        )
        assert response.id == "store-id-1"


class datastoreTests:
    def test_datastore_properties(self, datastore):
        assert isinstance(datastore.id, str)
        assert isinstance(datastore.provider_uuid, str)
        assert isinstance(datastore.region_uuid, str)
        assert isinstance(datastore.type, str)
        assert isinstance(datastore.name, str)
        assert isinstance(datastore.uri, str)
        assert isinstance(datastore.root, str)

    def test_datastore_str(self, datastore):
        string = str(datastore)
        regex = r"^{.*\"store_id\": \"" + datastore.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_datastore_repr(self, datastore):
        string = repr(datastore)
        regex = (
            r"^Datastore\( trainml , \*\*{.*'store_id': '"
            + datastore.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_datastore_bool(self, datastore, mock_trainml):
        empty_datastore = specimen.Datastore(mock_trainml)
        assert bool(datastore)
        assert not bool(empty_datastore)

    @mark.asyncio
    async def test_datastore_remove(self, datastore, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datastore.remove()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/datastore/x", "DELETE"
        )

    @mark.asyncio
    async def test_datastore_refresh(self, datastore, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "store_id": "store-id-1",
            "name": "On-Prem Datastore",
            "type": "nfs",
            "uri": "192.168.0.50",
            "root": "/exports",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await datastore.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/provider/1/region/a/datastore/x", "GET"
        )
        assert datastore.id == "store-id-1"
        assert response.id == "store-id-1"
