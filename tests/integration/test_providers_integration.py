import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.providers]


@mark.create
@mark.asyncio
class GetProvidersTests:
    @fixture(scope="class")
    async def provider(self, trainml):
        provider = await trainml.providers.enable(type="test")
        yield provider
        await provider.remove()

    async def test_get_providers(self, trainml):
        providers = await trainml.providers.list()
        assert len(providers) > 0

    async def test_get_provider(self, trainml, provider):
        response = await trainml.providers.get(provider.id)
        assert response.id == provider.id

    async def test_provider_properties(self, provider):
        assert isinstance(provider.id, str)
        assert isinstance(provider.type, str)
        assert isinstance(provider.regions, list)
        assert provider.type == "test"
        assert len(provider.regions) == 0
        assert provider.credits == 0

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
