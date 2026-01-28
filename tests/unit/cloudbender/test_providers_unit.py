import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.providers as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.providers]


@fixture
def providers(mock_trainml):
    yield specimen.Providers(mock_trainml)


@fixture
def provider(mock_trainml):
    yield specimen.Provider(
        mock_trainml,
        customer_uuid="a",
        provider_uuid="1",
        type="physical",
        payment_mode="credits",
        createdAt="2020-12-31T23:59:59.000Z",
        credits=0.0,
    )


class ProvidersTests:
    @mark.asyncio
    async def test_get_provider(
        self,
        providers,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await providers.get("1234")
        mock_trainml._query.assert_called_once_with("/provider/1234", "GET")

    @mark.asyncio
    async def test_list_providers(
        self,
        providers,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await providers.list()
        mock_trainml._query.assert_called_once_with("/provider", "GET")

    @mark.asyncio
    async def test_remove_provider(
        self,
        providers,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await providers.remove("4567")
        mock_trainml._query.assert_called_once_with("/provider/4567", "DELETE")

    @mark.asyncio
    async def test_enable_provider_simple(self, providers, mock_trainml):
        requested_config = dict(
            type="physical",
        )
        expected_payload = dict(type="physical")
        api_response = {
            "customer_uuid": "cust-id-1",
            "provider_uuid": "provider-id-1",
            "type": "new provider",
            "credits": 0.0,
            "payment_mode": "credits",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await providers.enable(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider", "POST", None, expected_payload
        )
        assert response.id == "provider-id-1"


class providerTests:
    def test_provider_properties(self, provider):
        assert isinstance(provider.id, str)
        assert isinstance(provider.type, str)
        assert isinstance(provider.credits, float)

    def test_provider_str(self, provider):
        string = str(provider)
        regex = r"^{.*\"provider_uuid\": \"" + provider.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_provider_repr(self, provider):
        string = repr(provider)
        regex = (
            r"^Provider\( trainml , \*\*{.*'provider_uuid': '"
            + provider.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_provider_bool(self, provider, mock_trainml):
        empty_provider = specimen.Provider(mock_trainml)
        assert bool(provider)
        assert not bool(empty_provider)

    @mark.asyncio
    async def test_provider_remove(self, provider, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await provider.remove()
        mock_trainml._query.assert_called_once_with("/provider/1", "DELETE")

    @mark.asyncio
    async def test_provider_refresh(self, provider, mock_trainml):
        api_response = {
            "customer_uuid": "cust-id-1",
            "provider_uuid": "provider-id-1",
            "type": "new provider",
            "credits": 0.0,
            "payment_mode": "credits",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await provider.refresh()
        mock_trainml._query.assert_called_once_with(f"/provider/1", "GET")
        assert provider.id == "provider-id-1"
        assert response.id == "provider-id-1"

    def test_provider_status_property(self, provider):
        """Test provider status property."""
        provider._status = "ready"
        assert provider.status == "ready"

    @mark.asyncio
    async def test_provider_wait_for_already_at_status(self, provider):
        """Test wait_for returns immediately if already at target status."""
        provider._status = "ready"
        result = await provider.wait_for("ready")
        assert result is None

    @mark.asyncio
    async def test_provider_wait_for_invalid_status(self, provider):
        """Test wait_for raises error for invalid status."""
        with raises(SpecificationError) as exc_info:
            await provider.wait_for("invalid_status")
        assert "Invalid wait_for status" in str(exc_info.value.message)

    @mark.asyncio
    async def test_provider_wait_for_timeout_validation(self, provider):
        """Test wait_for validates timeout."""
        with raises(SpecificationError) as exc_info:
            await provider.wait_for("ready", timeout=25 * 60 * 60)
        assert "timeout must be less than" in str(exc_info.value.message)

    @mark.asyncio
    async def test_provider_wait_for_success(self, provider, mock_trainml):
        """Test wait_for succeeds when status matches."""
        provider._status = "new"
        api_response_new = dict(
            customer_uuid="a",
            provider_uuid="1",
            status="new",
        )
        api_response_ready = dict(
            customer_uuid="a",
            provider_uuid="1",
            status="ready",
        )
        mock_trainml._query = AsyncMock(
            side_effect=[api_response_new, api_response_ready]
        )
        with patch("trainml.cloudbender.providers.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.wait_for("ready", timeout=10)
        assert result == provider
        assert provider.status == "ready"

    @mark.asyncio
    async def test_provider_wait_for_archived_404(self, provider, mock_trainml):
        """Test wait_for handles 404 for archived status."""
        provider._status = "ready"
        api_error = ApiError(404, {"errorMessage": "Not found"})
        mock_trainml._query = AsyncMock(side_effect=api_error)
        with patch("trainml.cloudbender.providers.asyncio.sleep", new_callable=AsyncMock):
            await provider.wait_for("archived", timeout=10)

    @mark.asyncio
    async def test_provider_wait_for_error_status(self, provider, mock_trainml):
        """Test wait_for raises error for errored/failed status."""
        provider._status = "new"
        api_response_errored = dict(
            customer_uuid="a",
            provider_uuid="1",
            status="errored",
        )
        mock_trainml._query = AsyncMock(return_value=api_response_errored)
        with patch("trainml.cloudbender.providers.asyncio.sleep", new_callable=AsyncMock):
            with raises(specimen.ProviderError):
                await provider.wait_for("ready", timeout=10)

    @mark.asyncio
    async def test_provider_wait_for_timeout(self, provider, mock_trainml):
        """Test wait_for raises timeout exception."""
        provider._status = "new"
        api_response_new = dict(
            customer_uuid="a",
            provider_uuid="1",
            status="new",
        )
        mock_trainml._query = AsyncMock(return_value=api_response_new)
        with patch("trainml.cloudbender.providers.asyncio.sleep", new_callable=AsyncMock):
            with raises(TrainMLException) as exc_info:
                await provider.wait_for("ready", timeout=0.1)
        assert "Timeout waiting for" in str(exc_info.value.message)

    @mark.asyncio
    async def test_provider_wait_for_api_error_non_404(self, provider, mock_trainml):
        """Test wait_for raises ApiError when not 404 for archived (line 115)."""
        provider._status = "ready"
        api_error = ApiError(500, {"errorMessage": "Server Error"})
        mock_trainml._query = AsyncMock(side_effect=api_error)
        with patch("trainml.cloudbender.providers.asyncio.sleep", new_callable=AsyncMock):
            with raises(ApiError):
                await provider.wait_for("archived", timeout=10)
