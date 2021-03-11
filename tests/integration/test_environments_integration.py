import re
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.environments]


@fixture(scope="module")
async def environments(trainml):
    environments = await trainml.environments.list()
    yield environments


@fixture(scope="module")
async def environment(environments):
    environment = next(
        (environment for environment in environments if environment.version),
        None,
    )
    yield environment


@mark.asyncio
async def test_get_environments(environments):
    assert len(environments) > 0


@mark.asyncio
async def test_environment_properties(environment):
    assert isinstance(environment.id, str)
    assert isinstance(environment.name, str)
    assert isinstance(environment.py_version, str)
    assert isinstance(environment.framework, str)
    assert isinstance(environment.version, str)
    assert isinstance(environment.cuda_version, str)


@mark.asyncio
def test_environment_str(environment):
    string = str(environment)
    regex = r"^{.*\"id\": \"" + environment.id + r"\".*}$"
    assert isinstance(string, str)
    assert re.match(regex, string)


@mark.asyncio
def test_environment_repr(environment):
    string = repr(environment)
    regex = (
        r"^Environment\( trainml , \*\*{.*'id': '"
        + environment.id
        + r"'.*}\)$"
    )
    assert isinstance(string, str)
    assert re.match(regex, string)
