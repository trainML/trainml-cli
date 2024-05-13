import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.projects.services as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.projects]


@fixture
def project_services(mock_trainml):
    yield specimen.ProjectServices(mock_trainml, project_id="1")


@fixture
def project_service(mock_trainml):
    yield specimen.ProjectService(
        mock_trainml,
        id="res-id-1",
        name="service 1",
        project_uuid="proj-id-1",
        region_uuid="reg-id-1",
        public=False,
        hostname="asdf.proximl.cloud",
    )


class ProjectServicesTests:
    @mark.asyncio
    async def test_project_services_refresh(self, project_services, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project_services.refresh()
        mock_trainml._query.assert_called_once_with("/project/1/services", "PATCH")

    @mark.asyncio
    async def test_project_services_list(self, project_services, mock_trainml):
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
        resp = await project_services.list()
        mock_trainml._query.assert_called_once_with(
            "/project/1/services", "GET", dict()
        )
        assert len(resp) == 2


class ProjectServiceTests:
    def test_project_service_properties(self, project_service):
        assert isinstance(project_service.id, str)
        assert isinstance(project_service.name, str)
        assert isinstance(project_service.project_uuid, str)
        assert isinstance(project_service.hostname, str)
        assert isinstance(project_service.public, bool)
        assert isinstance(project_service.region_uuid, str)

    def test_project_service_str(self, project_service):
        string = str(project_service)
        regex = r"^{.*\"id\": \"" + project_service.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_service_repr(self, project_service):
        string = repr(project_service)
        regex = (
            r"^ProjectService\( trainml , \*\*{.*'id': '"
            + project_service.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_service_bool(self, project_service, mock_trainml):
        empty_project_service = specimen.ProjectService(mock_trainml)
        assert bool(project_service)
        assert not bool(empty_project_service)
