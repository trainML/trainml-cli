import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.projects.data_connectors as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.projects]


@fixture
def project_data_connectors(mock_trainml):
    yield specimen.ProjectDataConnectors(mock_trainml, project_id="1")


@fixture
def project_data_connector(mock_trainml):
    yield specimen.ProjectDataConnector(
        mock_trainml,
        id="ds-id-1",
        name="connector 1",
        project_uuid="proj-id-1",
        type="custom",
        region_uuid="reg-id-1",
    )


class ProjectDataConnectorsTests:
    @mark.asyncio
    async def test_project_data_connectors_refresh(
        self, project_data_connectors, mock_trainml
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project_data_connectors.refresh()
        mock_trainml._query.assert_called_once_with(
            "/project/1/data_connectors", "PATCH"
        )

    @mark.asyncio
    async def test_project_data_connectors_list(
        self, project_data_connectors, mock_trainml
    ):
        api_response = [
            {
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-1",
                "id": "store-id-1",
                "type": "custom",
                "name": "On-Prem Connection A",
            },
            {
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-2",
                "id": "store-id-2",
                "type": "custom",
                "name": "Cloud Connection B",
            },
        ]
        mock_trainml._query = AsyncMock(return_value=api_response)
        resp = await project_data_connectors.list()
        mock_trainml._query.assert_called_once_with(
            "/project/1/data_connectors", "GET", dict()
        )
        assert len(resp) == 2


class ProjectDataConnectorTests:
    def test_project_data_connector_properties(self, project_data_connector):
        assert isinstance(project_data_connector.id, str)
        assert isinstance(project_data_connector.name, str)
        assert isinstance(project_data_connector.project_uuid, str)
        assert isinstance(project_data_connector.type, str)
        assert isinstance(project_data_connector.region_uuid, str)

    def test_project_data_connector_str(self, project_data_connector):
        string = str(project_data_connector)
        regex = r"^{.*\"id\": \"" + project_data_connector.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_data_connector_repr(self, project_data_connector):
        string = repr(project_data_connector)
        regex = (
            r"^ProjectDataConnector\( trainml , \*\*{.*'id': '"
            + project_data_connector.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_data_connector_bool(self, project_data_connector, mock_trainml):
        empty_project_data_connector = specimen.ProjectDataConnector(mock_trainml)
        assert bool(project_data_connector)
        assert not bool(empty_project_data_connector)
