import re
from pytest import mark, fixture


pytestmark = [mark.sdk, mark.integration, mark.gpu_types]


@fixture(scope="module")
async def gpu_types(trainml):
    gpu_types = await trainml.gpu_types.list()
    yield gpu_types


@fixture(scope="module")
async def gpu_type(gpu_types):
    gpu_type = next(
        (gpu_type for gpu_type in gpu_types if gpu_type.name == "GTX 1060"),
        None,
    )
    yield gpu_type


@mark.asyncio
async def test_get_gpu_types(gpu_types):
    assert len(gpu_types) > 0


@mark.asyncio
async def test_gpu_type_properties(gpu_type):
    assert isinstance(gpu_type.id, str)
    assert isinstance(gpu_type.name, str)
    assert isinstance(gpu_type.abbrv, str)
    assert isinstance(gpu_type.credits_per_hour_min, float)
    assert isinstance(gpu_type.credits_per_hour_max, float)


@mark.asyncio
async def test_gpu_type_str(gpu_type):
    string = str(gpu_type)
    regex = r"^{.*\"id\": \"" + gpu_type.id + r"\".*}$"
    assert isinstance(string, str)
    assert re.match(regex, string)


@mark.asyncio
async def test_gpu_type_repr(gpu_type):
    string = repr(gpu_type)
    regex = r"^GpuType\( trainml , \*\*{.*'id': '" + gpu_type.id + r"'.*}\)$"
    assert isinstance(string, str)
    assert re.match(regex, string)
