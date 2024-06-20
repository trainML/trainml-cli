import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
class ProjectDatastoresTests:
    @fixture(scope="class")
    async def project_datastore(self, project):
        datastores = await project.datastores.list()
        yield datastores[0]

    async def test_list_project_datastores(self, project):
        datastores = await project.datastores.list()
        assert len(datastores) > 0

    async def test_project_datastore_properties(self, project, project_datastore):
        assert isinstance(project_datastore.project_uuid, str)
        assert isinstance(project_datastore.type, str)
        assert isinstance(project_datastore.name, str)
        assert isinstance(project_datastore.region_uuid, str)
        assert project.id == project_datastore.project_uuid

    async def test_project_datastore_str(self, project_datastore):
        string = str(project_datastore)
        regex = r"^{.*\"name\": \"" + project_datastore.name + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_datastore_repr(self, project_datastore):
        string = repr(project_datastore)
        regex = (
            r"^ProjectDatastore\( trainml , \*\*{.*'name': '"
            + project_datastore.name
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
