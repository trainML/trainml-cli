import re
import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.projects.authorizers as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.projects]


@fixture
def project_authorizers(mock_trainml):
    yield specimen.ProjectAuthorizers(mock_trainml, project_id="1")


@fixture
def project_authorizer(mock_trainml):
    yield specimen.ProjectAuthorizer(
        mock_trainml,
        id="auth-id-1",
        project_uuid="proj-id-1",
        name="api_key_authorizer",
        type="api_key",
        keys=[{"client_id": "key-1", "key": "secret-key", "created_at": 1685733760084}],
        createdAt="2023-06-02T21:22:40.084Z",
        updatedAt="2023-06-02T21:22:40.084Z",
    )


class ProjectAuthorizersTests:
    @mark.asyncio
    async def test_get_project_authorizer(self, project_authorizers, mock_trainml):
        api_response = {
            "id": "auth-id-1",
            "project_uuid": "proj-id-1",
            "name": "api_key_authorizer",
            "type": "api_key",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        resp = await project_authorizers.get("api_key_authorizer")
        mock_trainml._query.assert_called_once_with(
            "/project/1/authorizers/api_key_authorizer", "GET", dict()
        )
        assert resp.name == "api_key_authorizer"

    @mark.asyncio
    async def test_project_authorizers_list(self, project_authorizers, mock_trainml):
        api_response = [
            {
                "id": "auth-id-1",
                "project_uuid": "proj-id-1",
                "name": "api_key_authorizer",
                "type": "api_key",
            },
            {
                "id": "auth-id-2",
                "project_uuid": "proj-id-1",
                "name": "oidc_authorizer",
                "type": "oidc",
            },
        ]
        mock_trainml._query = AsyncMock(return_value=api_response)
        resp = await project_authorizers.list()
        mock_trainml._query.assert_called_once_with(
            "/project/1/authorizers", "GET", dict()
        )
        assert len(resp) == 2

    @mark.asyncio
    async def test_remove_project_authorizer(
        self,
        project_authorizers,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project_authorizers.remove("auth-id-1")
        mock_trainml._query.assert_called_once_with(
            "/project/1/authorizers/auth-id-1", "DELETE", dict()
        )

    @mark.asyncio
    async def test_create_project_authorizer_api_key(
        self, project_authorizers, mock_trainml
    ):
        requested_config = dict(
            name="api_key_authorizer",
            type="api_key",
            keys=[{"client_id": "key-1", "key": "secret-key"}],
        )
        expected_payload = dict(
            name="api_key_authorizer",
            type="api_key",
            keys=[{"client_id": "key-1", "key": "secret-key"}],
        )
        api_response = {
            "id": "auth-id-1",
            "project_uuid": "project-id-1",
            "name": "api_key_authorizer",
            "type": "api_key",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await project_authorizers.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/project/1/authorizers", "POST", dict(), expected_payload
        )
        assert response.name == "api_key_authorizer"

    @mark.asyncio
    async def test_create_project_authorizer_oidc(
        self, project_authorizers, mock_trainml
    ):
        requested_config = dict(
            name="oidc_authorizer",
            type="oidc",
            oidc={
                "issuer": "https://auth.example.com",
                "audience": "trainml-test",
                "algorithms": ["RS256"],
            },
        )
        expected_payload = dict(
            name="oidc_authorizer",
            type="oidc",
            oidc={
                "issuer": "https://auth.example.com",
                "audience": "trainml-test",
                "algorithms": ["RS256"],
            },
        )
        api_response = {
            "id": "auth-id-2",
            "project_uuid": "project-id-1",
            "name": "oidc_authorizer",
            "type": "oidc",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await project_authorizers.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/project/1/authorizers", "POST", dict(), expected_payload
        )
        assert "keys" not in expected_payload
        assert response.name == "oidc_authorizer"


class ProjectAuthorizerTests:
    @mark.asyncio
    async def test_edit_project_authorizer(self, project_authorizer, mock_trainml):
        expected_payload = dict(
            keys=[
                {"client_id": "key-1", "key": "secret-key"},
                {"client_id": "key-2", "key": "secret-key-2"},
            ]
        )
        api_response = {
            "id": "auth-id-1",
            "project_uuid": "proj-id-1",
            "name": "api_key_authorizer",
            "type": "api_key",
            "keys": [
                {
                    "client_id": "key-1",
                    "key": "secret-key",
                    "created_at": 1685733760084,
                },
                {
                    "client_id": "key-2",
                    "key": "secret-key-2",
                    "created_at": 1685733760084,
                },
            ],
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await project_authorizer.edit(keys=expected_payload["keys"])
        mock_trainml._query.assert_called_once_with(
            "/project/proj-id-1/authorizers/auth-id-1",
            "PATCH",
            dict(),
            expected_payload,
        )
        assert len(response.keys) == 2

    def test_project_authorizer_properties(self, project_authorizer):
        assert isinstance(project_authorizer.id, str)
        assert isinstance(project_authorizer.project_uuid, str)
        assert isinstance(project_authorizer.name, str)
        assert isinstance(project_authorizer.type, str)
        assert isinstance(project_authorizer.keys, list)
        assert isinstance(project_authorizer.created_at, datetime)
        assert isinstance(project_authorizer.updated_at, datetime)
        assert project_authorizer.oidc is None

    def test_project_authorizer_timestamps_epoch_ms(self, mock_trainml):
        # The api-project dynamoose model stores createdAt/updatedAt as Date,
        # which serializes as an epoch-millisecond integer.
        authorizer = specimen.ProjectAuthorizer(
            mock_trainml,
            id="auth-id-1",
            project_uuid="proj-id-1",
            name="api_key_authorizer",
            type="api_key",
            createdAt=1685733760084,
            updatedAt=1685733760084,
        )
        assert isinstance(authorizer.created_at, datetime)
        assert isinstance(authorizer.updated_at, datetime)
        assert authorizer.created_at.astimezone().timestamp() * 1000 == 1685733760084
        assert authorizer.updated_at.astimezone().timestamp() * 1000 == 1685733760084

    def test_project_authorizer_timestamps_iso(self, mock_trainml):
        authorizer = specimen.ProjectAuthorizer(
            mock_trainml,
            id="auth-id-1",
            project_uuid="proj-id-1",
            name="api_key_authorizer",
            type="api_key",
            createdAt="2023-06-02T21:22:40.084Z",
            updatedAt="2023-06-02T21:22:40.084Z",
        )
        assert isinstance(authorizer.created_at, datetime)
        assert isinstance(authorizer.updated_at, datetime)

    def test_project_authorizer_timestamps_none(self, mock_trainml):
        authorizer = specimen.ProjectAuthorizer(
            mock_trainml,
            id="auth-id-1",
            project_uuid="proj-id-1",
            name="api_key_authorizer",
            type="api_key",
        )
        assert authorizer.created_at is None
        assert authorizer.updated_at is None

    def test_project_authorizer_str(self, project_authorizer):
        string = str(project_authorizer)
        regex = r"^{.*\"name\": \"" + project_authorizer.name + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_authorizer_repr(self, project_authorizer):
        string = repr(project_authorizer)
        regex = (
            r"^ProjectAuthorizer\( trainml , \*\*{.*'name': '"
            + project_authorizer.name
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_authorizer_bool(self, project_authorizer, mock_trainml):
        empty_project_authorizer = specimen.ProjectAuthorizer(mock_trainml)
        assert bool(project_authorizer)
        assert not bool(empty_project_authorizer)
