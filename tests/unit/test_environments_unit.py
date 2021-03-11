import re
from unittest.mock import AsyncMock
from pytest import mark, fixture

import trainml.environments as specimen

pytestmark = [mark.sdk, mark.unit, mark.environments]


@fixture
def environments(mock_trainml):
    yield specimen.Environments(mock_trainml)


@fixture
def environment(mock_trainml):
    yield specimen.Environment(
        mock_trainml,
        **{
            "id": "PYTORCH_PY38_17",
            "framework": "PyTorch",
            "py_version": "3.8",
            "version": "1.7",
            "cuda_version": "11.1",
            "name": "PyTorch 1.7 - Python 3.8",
        },
    )


class EnvironmentsTests:
    @mark.asyncio
    async def test_list_environments(self, environments, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await environments.list()
        mock_trainml._query.assert_called_once_with(
            f"/job/environments", "GET"
        )


class EnvironmentTests:
    def test_environment_properties(self, environment):
        assert isinstance(environment.id, str)
        assert isinstance(environment.name, str)
        assert isinstance(environment.py_version, str)
        assert isinstance(environment.framework, str)
        assert isinstance(environment.version, str)
        assert isinstance(environment.cuda_version, str)

    def test_environment_str(self, environment):
        string = str(environment)
        regex = r"^{.*\"id\": \"" + environment.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_environment_repr(self, environment):
        string = repr(environment)
        regex = (
            r"^Environment\( trainml , \*\*{.*'id': '"
            + environment.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
