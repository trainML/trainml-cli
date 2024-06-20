import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
class ProjectDataConnectorsTests:
    @fixture(scope="class")
    async def project_data_connector(self, project):
        data_connectors = await project.data_connectors.list()
        yield data_connectors[0]

    async def test_list_project_data_connectors(self, project, project_data_connector):
        data_connectors = await project.data_connectors.list()
        assert len(data_connectors) > 0

    async def test_project_data_connector_properties(
        self, project, project_data_connector
    ):
        assert isinstance(project_data_connector.project_uuid, str)
        assert isinstance(project_data_connector.type, str)
        assert isinstance(project_data_connector.name, str)
        assert isinstance(project_data_connector.region_uuid, str)
        assert project.id == project_data_connector.project_uuid

    async def test_project_data_connector_str(self, project_data_connector):
        string = str(project_data_connector)
        regex = r"^{.*\"name\": \"" + project_data_connector.name + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_data_connector_repr(self, project_data_connector):
        string = repr(project_data_connector)
        regex = (
            r"^ProjectDataConnector\( trainml , \*\*{.*'name': '"
            + project_data_connector.name
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
