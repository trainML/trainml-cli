import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
class ProjectServicesTests:
    @fixture(scope="class")
    async def project_service(self, project):
        services = await project.services.list()
        yield services[0]

    async def test_list_project_services(self, project):
        services = await project.services.list()
        assert len(services) > 0

    async def test_project_service_properties(self, project, project_service):
        assert isinstance(project_service.project_uuid, str)
        assert isinstance(project_service.name, str)
        assert isinstance(project_service.id, str)
        assert isinstance(project_service.hostname, str)
        assert isinstance(project_service.region_uuid, str)
        assert isinstance(project_service.public, bool)
        assert project.id == project_service.project_uuid

    async def test_project_service_str(self, project_service):
        string = str(project_service)
        regex = r"^{.*\"name\": \"" + project_service.name + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_service_repr(self, project_service):
        string = repr(project_service)
        regex = (
            r"^ProjectService\( trainml , \*\*{.*'name': '"
            + project_service.name
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
