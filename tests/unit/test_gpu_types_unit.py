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
            "price": {"min": 0.1, "max": 0.1},
            "name": "GTX 1060",
            "id": "1060-id",
            "abbrv": "gtx1060",
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
        mock_trainml._query.assert_called_once_with(
            f"/project/proj-id-a/gputypes", "GET"
        )


class GpuTypeTests:
    def test_gpu_type_properties(self, gpu_type):
        assert isinstance(gpu_type.id, str)
        assert isinstance(gpu_type.name, str)
        assert isinstance(gpu_type.abbrv, str)
        assert isinstance(gpu_type.credits_per_hour_min, float)
        assert isinstance(gpu_type.credits_per_hour_max, float)

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
