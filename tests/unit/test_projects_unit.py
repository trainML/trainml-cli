import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.projects as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.projects]


@fixture
def projects(mock_trainml):
    yield specimen.Projects(mock_trainml)


@fixture
def project(mock_trainml):
    yield specimen.Project(
        mock_trainml,
        id="1",
        name="My Mock Project",
        owner=True,
        owner_name="Me",
        created_name="Me",
        job_all=True,
        dataset_all=True,
        model_all=True,
        createdAt="2020-12-31T23:59:59.000Z",
    )


class ProjectsTests:
    @mark.asyncio
    async def test_get_project(
        self,
        projects,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await projects.get("1234")
        mock_trainml._query.assert_called_once_with("/project/1234", "GET")

    @mark.asyncio
    async def test_list_projects(
        self,
        projects,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await projects.list()
        mock_trainml._query.assert_called_once_with("/project", "GET")

    @mark.asyncio
    async def test_remove_project(
        self,
        projects,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await projects.remove("4567")
        mock_trainml._query.assert_called_once_with("/project/4567", "DELETE")

    @mark.asyncio
    async def test_create_project_simple(self, projects, mock_trainml):
        requested_config = dict(
            name="new project",
        )
        expected_payload = dict(name="new project", copy_keys=False)
        api_response = {
            "id": "project-id-1",
            "name": "new project",
            "owner": True,
            "owner_name": "Me",
            "created_name": "Me",
            "job_all": True,
            "dataset_all": True,
            "model_all": True,
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await projects.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/project", "POST", None, expected_payload
        )
        assert response.id == "project-id-1"


class projectTests:
    def test_project_properties(self, project):
        assert isinstance(project.id, str)
        assert isinstance(project.name, str)
        assert isinstance(project.owner_name, str)
        assert isinstance(project.is_owner, bool)

    def test_project_str(self, project):
        string = str(project)
        regex = r"^{.*\"id\": \"" + project.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_repr(self, project):
        string = repr(project)
        regex = (
            r"^Project\( trainml , \*\*{.*'id': '" + project.id + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_bool(self, project, mock_trainml):
        empty_project = specimen.Project(mock_trainml)
        assert bool(project)
        assert not bool(empty_project)

    @mark.asyncio
    async def test_project_remove(self, project, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project.remove()
        mock_trainml._query.assert_called_once_with("/project/1", "DELETE")
