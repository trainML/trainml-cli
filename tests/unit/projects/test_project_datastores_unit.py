import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.projects.datastores as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.projects]


@fixture
def project_datastores(mock_trainml):
    yield specimen.ProjectDatastores(mock_trainml, project_id="1")


@fixture
def project_datastore(mock_trainml):
    yield specimen.ProjectDatastore(
        mock_trainml,
        id="ds-id-1",
        name="datastore 1",
        project_uuid="proj-id-1",
        type="nfs",
        region_uuid="reg-id-1",
    )


class ProjectDatastoresTests:
    @mark.asyncio
    async def test_project_datastores_refresh(self, project_datastores, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project_datastores.refresh()
        mock_trainml._query.assert_called_once_with("/project/1/datastores", "PATCH")

    @mark.asyncio
    async def test_project_datastores_list(self, project_datastores, mock_trainml):
        api_response = [
            {
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-1",
                "id": "store-id-1",
                "type": "nfs",
                "name": "On-prem NFS",
            },
            {
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-2",
                "id": "store-id-2",
                "type": "smb",
                "name": "GCP Samba",
            },
        ]
        mock_trainml._query = AsyncMock(return_value=api_response)
        resp = await project_datastores.list()
        mock_trainml._query.assert_called_once_with(
            "/project/1/datastores", "GET", dict()
        )
        assert len(resp) == 2


class ProjectDatastoreTests:
    def test_project_datastore_properties(self, project_datastore):
        assert isinstance(project_datastore.id, str)
        assert isinstance(project_datastore.name, str)
        assert isinstance(project_datastore.project_uuid, str)
        assert isinstance(project_datastore.type, str)
        assert isinstance(project_datastore.region_uuid, str)

    def test_project_datastore_str(self, project_datastore):
        string = str(project_datastore)
        regex = r"^{.*\"id\": \"" + project_datastore.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_datastore_repr(self, project_datastore):
        string = repr(project_datastore)
        regex = (
            r"^ProjectDatastore\( trainml , \*\*{.*'id': '"
            + project_datastore.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_datastore_bool(self, project_datastore, mock_trainml):
        empty_project_datastore = specimen.ProjectDatastore(mock_trainml)
        assert bool(project_datastore)
        assert not bool(empty_project_datastore)
