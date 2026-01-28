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

    def test_service_status_property(self, service):
        """Test service status property."""
        service._status = "active"
        assert service.status == "active"

    def test_service_port_property(self, service):
        """Test service port property."""
        service._port = "443"
        assert service.port == "443"

    @mark.asyncio
    async def test_service_wait_for_already_at_status(self, service):
        """Test wait_for returns immediately if already at target status."""
        service._status = "active"
        result = await service.wait_for("active")
        assert result is None

    @mark.asyncio
    async def test_service_wait_for_invalid_status(self, service):
        """Test wait_for raises error for invalid status."""
        with raises(SpecificationError) as exc_info:
            await service.wait_for("invalid_status")
        assert "Invalid wait_for status" in str(exc_info.value.message)

    @mark.asyncio
    async def test_service_wait_for_timeout_validation(self, service):
        """Test wait_for validates timeout."""
        with raises(SpecificationError) as exc_info:
            await service.wait_for("active", timeout=25 * 60 * 60)
        assert "timeout must be less than" in str(exc_info.value.message)

    @mark.asyncio
    async def test_service_wait_for_success(self, service, mock_trainml):
        """Test wait_for succeeds when status matches."""
        service._status = "new"
        api_response_new = dict(
            provider_uuid="1",
            region_uuid="a",
            service_id="x",
            status="new",
        )
        api_response_active = dict(
            provider_uuid="1",
            region_uuid="a",
            service_id="x",
            status="active",
        )
        mock_trainml._query = AsyncMock(
            side_effect=[api_response_new, api_response_active]
        )
        with patch("trainml.cloudbender.services.asyncio.sleep", new_callable=AsyncMock):
            result = await service.wait_for("active", timeout=10)
        assert result == service
        assert service.status == "active"

    @mark.asyncio
    async def test_service_wait_for_archived_404(self, service, mock_trainml):
        """Test wait_for handles 404 for archived status."""
        service._status = "active"
        api_error = ApiError(404, {"errorMessage": "Not found"})
        mock_trainml._query = AsyncMock(side_effect=api_error)
        with patch("trainml.cloudbender.services.asyncio.sleep", new_callable=AsyncMock):
            await service.wait_for("archived", timeout=10)

    @mark.asyncio
    async def test_service_wait_for_timeout(self, service, mock_trainml):
        """Test wait_for raises timeout exception."""
        service._status = "new"
        api_response_new = dict(
            provider_uuid="1",
            region_uuid="a",
            service_id="x",
            status="new",
        )
        mock_trainml._query = AsyncMock(return_value=api_response_new)
        with patch("trainml.cloudbender.services.asyncio.sleep", new_callable=AsyncMock):
            with raises(TrainMLException) as exc_info:
                await service.wait_for("active", timeout=0.1)
        assert "Timeout waiting for" in str(exc_info.value.message)

    @mark.asyncio
    async def test_service_wait_for_api_error_non_404(self, service, mock_trainml):
        """Test wait_for raises ApiError when not 404 for archived (line 181)."""
        service._status = "active"
        api_error = ApiError(500, {"errorMessage": "Server Error"})
        mock_trainml._query = AsyncMock(side_effect=api_error)
        with patch("trainml.cloudbender.services.asyncio.sleep", new_callable=AsyncMock):
            with raises(ApiError):
                await service.wait_for("archived", timeout=10)

    @mark.asyncio
    async def test_service_generate_certificate(self, service, mock_trainml):
        """Test generate_certificate method."""
        api_response = {
            "provider_uuid": "1",
            "region_uuid": "a",
            "service_id": "x",
            "certificate": "cert-data",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        result = await service.generate_certificate()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/service/x/certificate",
            "POST",
            {},
            dict(algorithm="ed25519"),
        )
        assert result == service

    @mark.asyncio
    async def test_service_generate_certificate_custom_algorithm(self, service, mock_trainml):
        """Test generate_certificate with custom algorithm."""
        api_response = {
            "provider_uuid": "1",
            "region_uuid": "a",
            "service_id": "x",
            "certificate": "cert-data",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        result = await service.generate_certificate(algorithm="rsa")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/service/x/certificate",
            "POST",
            {},
            dict(algorithm="rsa"),
        )

    @mark.asyncio
    async def test_service_sign_client_certificate(self, service, mock_trainml):
        """Test sign_client_certificate method."""
        api_response = {"certificate": "signed-cert-data"}
        mock_trainml._query = AsyncMock(return_value=api_response)
        result = await service.sign_client_certificate("csr-data")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/service/x/certificate/sign",
            "POST",
            {},
            dict(csr="csr-data"),
        )
        assert result == api_response
