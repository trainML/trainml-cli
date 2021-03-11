import re
from unittest.mock import AsyncMock
from pytest import mark, fixture

import trainml.gpu_types as specimen

pytestmark = [mark.sdk, mark.unit, mark.gpu_types]


@fixture()
def gpu_types(mock_trainml):
    yield specimen.GpuTypes(mock_trainml)


@fixture()
def gpu_type(mock_trainml):
    yield specimen.GpuType(
        mock_trainml,
        **{
            "available": 4,
            "credits_per_hour": 0.1,
            "name": "GTX 1060",
            "provider": "trainml",
            "id": "1060-id",
        },
    )


class GpuTypesTests:
    @mark.asyncio
    async def test_list_gpu_types(
        self,
        gpu_types,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await gpu_types.list()
        mock_trainml._query.assert_called_once_with(f"/gpu/pub/types", "GET")


class GpuTypeTests:
    def test_gpu_type_properties(self, gpu_type):
        assert isinstance(gpu_type.id, str)
        assert isinstance(gpu_type.name, str)
        assert isinstance(gpu_type.provider, str)
        assert isinstance(gpu_type.available, int)
        assert isinstance(gpu_type.credits_per_hour, float)

    @mark.asyncio
    async def test_gpu_type_refresh(
        self,
        gpu_type,
        mock_trainml,
    ):
        api_response = {
            "available": 2,
            "credits_per_hour": 0.1,
            "name": "GTX 1060",
            "provider": "trainml",
            "id": "1060-id",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        await gpu_type.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/gpu/pub/types/{gpu_type.id}", "GET"
        )
        assert gpu_type.available == 2

    def test_gpu_type_str(self, gpu_type):
        string = str(gpu_type)
        regex = r"^{.*\"id\": \"" + gpu_type.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_gpu_type_repr(self, gpu_type):
        string = repr(gpu_type)
        regex = (
            r"^GpuType\( trainml , \*\*{.*'id': '" + gpu_type.id + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
