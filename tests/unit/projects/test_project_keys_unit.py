import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.projects.keys as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.projects]


@fixture
def project_keys(mock_trainml):
    yield specimen.ProjectKeys(mock_trainml, project_id="1")


@fixture
def project_key(mock_trainml):
    yield specimen.ProjectKey(
        mock_trainml, project_uuid="proj-id-1", type="aws", key_id="AIYHGFSDLK"
    )


class ProjectKeysTests:
    @mark.asyncio
    async def test_project_keys_list(self, project_keys, mock_trainml):
        api_response = [
            {"project_uuid": "proj-id-1", "type": "aws", "key_id": "AIYHGFSDLK"},
            {"project_uuid": "proj-id-1", "type": "gcp", "key_id": "credentials.json"},
        ]
        mock_trainml._query = AsyncMock(return_value=api_response)
        resp = await project_keys.list()
        mock_trainml._query.assert_called_once_with("/project/1/keys", "GET", dict())
        assert len(resp) == 2

    @mark.asyncio
    async def test_remove_project_key(
        self,
        project_keys,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project_keys.remove("aws")
        mock_trainml._query.assert_called_once_with(
            "/project/1/key/aws", "DELETE", dict()
        )

    @mark.asyncio
    async def test_put_project_key(self, project_keys, mock_trainml):
        requested_config = dict(type="aws", key_id="AIUDHADA", secret="ASKHJSLKF")
        expected_payload = dict(key_id="AIUDHADA", secret="ASKHJSLKF")
        api_response = {
            "project_uuid": "project-id-1",
            "type": "aws",
            "key_id": "AIUDHADA",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await project_keys.put(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/project/1/key/aws", "PUT", None, expected_payload
        )
        assert response.type == "aws"


class ProjectKeyTests:
    def test_project_key_properties(self, project_key):
        assert isinstance(project_key.type, str)
        assert isinstance(project_key.key_id, str)
        assert isinstance(project_key.project_uuid, str)

    def test_project_key_str(self, project_key):
        string = str(project_key)
        regex = r"^{.*\"type\": \"" + project_key.type + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_key_repr(self, project_key):
        string = repr(project_key)
        regex = (
            r"^ProjectKey\( trainml , \*\*{.*'type': '" + project_key.type + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_key_bool(self, project_key, mock_trainml):
        empty_project_key = specimen.ProjectKey(mock_trainml)
        assert bool(project_key)
        assert not bool(empty_project_key)
