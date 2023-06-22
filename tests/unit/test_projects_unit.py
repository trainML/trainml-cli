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


@fixture
def project_reservation(mock_trainml):
    yield specimen.ProjectReservation(
        mock_trainml,
        id="res-id-1",
        name="reservation 1",
        project_uuid="proj-id-1",
        region_uuid="reg-id-1",
        type="port",
        resource="8001",
        hostname="service.local",
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
        mock_trainml._query.assert_called_once_with(
            "/project/1234", "GET", dict()
        )

    @mark.asyncio
    async def test_list_projects(
        self,
        projects,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await projects.list()
        mock_trainml._query.assert_called_once_with("/project", "GET", dict())

    @mark.asyncio
    async def test_remove_project(
        self,
        projects,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await projects.remove("4567")
        mock_trainml._query.assert_called_once_with(
            "/project/4567", "DELETE", dict()
        )

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


class ProjectReservationTests:
    def test_project_reservation_properties(self, project_reservation):
        assert isinstance(project_reservation.id, str)
        assert isinstance(project_reservation.name, str)
        assert isinstance(project_reservation.project_uuid, str)
        assert isinstance(project_reservation.type, str)
        assert isinstance(project_reservation.hostname, str)
        assert isinstance(project_reservation.resource, str)
        assert isinstance(project_reservation.region_uuid, str)

    def test_project_reservation_str(self, project_reservation):
        string = str(project_reservation)
        regex = r"^{.*\"id\": \"" + project_reservation.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_reservation_repr(self, project_reservation):
        string = repr(project_reservation)
        regex = (
            r"^ProjectReservation\( trainml , \*\*{.*'id': '"
            + project_reservation.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_reservation_bool(self, project_reservation, mock_trainml):
        empty_project_reservation = specimen.ProjectReservation(mock_trainml)
        assert bool(project_reservation)
        assert not bool(empty_project_reservation)


class ProjectTests:
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

    @mark.asyncio
    async def test_project_refresh_datastores(self, project, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project.refresh_datastores()
        mock_trainml._query.assert_called_once_with(
            "/project/1/datastores", "PATCH"
        )

    @mark.asyncio
    async def test_project_refresh_reservations(self, project, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project.refresh_reservations()
        mock_trainml._query.assert_called_once_with(
            "/project/1/reservations", "PATCH"
        )

    @mark.asyncio
    async def test_project_list_datastores(self, project, mock_trainml):
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
        resp = await project.list_datastores()
        mock_trainml._query.assert_called_once_with(
            "/project/1/datastores", "GET"
        )
        assert len(resp) == 2

    @mark.asyncio
    async def test_project_list_reservations(self, project, mock_trainml):
        api_response = [
            {
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-1",
                "id": "res-id-1",
                "type": "port",
                "name": "On-Prem Service A",
                "resource": "8001",
                "hostname": "service-a.local",
            },
            {
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-2",
                "id": "res-id-2",
                "type": "port",
                "name": "Cloud Service B",
                "resource": "8001",
                "hostname": "service-b.local",
            },
        ]
        mock_trainml._query = AsyncMock(return_value=api_response)
        resp = await project.list_reservations()
        mock_trainml._query.assert_called_once_with(
            "/project/1/reservations", "GET"
        )
        assert len(resp) == 2
