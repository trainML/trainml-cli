import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.services as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.services]


@fixture
def services(mock_trainml):
    yield specimen.Services(mock_trainml)


@fixture
def service(mock_trainml):
    yield specimen.Service(
        mock_trainml,
        provider_uuid="1",
        region_uuid="a",
        service_id="x",
        name="On-Prem Service",
        type="https",
        public=False,
        hostname="app1.proximl.cloud",
    )


class RegionsTests:
    @mark.asyncio
    async def test_get_service(
        self,
        services,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await services.get("1234", "5687", "91011")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/service/91011", "GET", {}
        )

    @mark.asyncio
    async def test_list_services(
        self,
        services,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await services.list("1234", "5687")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/service", "GET", {}
        )

    @mark.asyncio
    async def test_remove_service(
        self,
        services,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await services.remove("1234", "4567", "8910")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/4567/service/8910", "DELETE", {}
        )

    @mark.asyncio
    async def test_create_service(self, services, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            region_uuid="region-id-1",
            name="On-Prem Service",
            type="https",
            public=False,
        )
        expected_payload = dict(
            name="On-Prem Service",
            type="https",
            public=False,
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "service_id": "service-id-1",
            "name": "On-Prem Service",
            "type": "https",
            "public": False,
            "hostname": "app1.proximl.cloud",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await services.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/service",
            "POST",
            None,
            expected_payload,
        )
        assert response.id == "service-id-1"


class serviceTests:
    def test_service_properties(self, service):
        assert isinstance(service.id, str)
        assert isinstance(service.provider_uuid, str)
        assert isinstance(service.region_uuid, str)
        assert isinstance(service.public, bool)
        assert isinstance(service.name, str)
        assert isinstance(service.hostname, str)
        assert isinstance(service.type, str)

    def test_service_str(self, service):
        string = str(service)
        regex = r"^{.*\"service_id\": \"" + service.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_service_repr(self, service):
        string = repr(service)
        regex = r"^Service\( trainml , \*\*{.*'service_id': '" + service.id + r"'.*}\)$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_service_bool(self, service, mock_trainml):
        empty_service = specimen.Service(mock_trainml)
        assert bool(service)
        assert not bool(empty_service)

    @mark.asyncio
    async def test_service_remove(self, service, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await service.remove()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/service/x", "DELETE"
        )

    @mark.asyncio
    async def test_service_refresh(self, service, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "service_id": "service-id-1",
            "name": "On-Prem Service",
            "type": "https",
            "public": False,
            "hostname": "app1.proximl.cloud",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await service.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/provider/1/region/a/service/x", "GET"
        )
        assert service.id == "service-id-1"
        assert response.id == "service-id-1"
