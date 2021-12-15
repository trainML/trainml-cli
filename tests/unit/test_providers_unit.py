import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.providers as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.providers]


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
        regions=[],
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
            "regions": [],
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
        assert isinstance(provider.regions, list)

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
