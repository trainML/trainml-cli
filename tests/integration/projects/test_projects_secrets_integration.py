import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
class ProjectSecretsTests:
    @fixture(scope="class")
    async def project_secret(self, project):
        project_secret = await project.secrets.put(
            name="secret_value", value="IUHKLHKAHF"
        )
        yield project_secret
        await project.secrets.remove(name="secret_value")

    async def test_list_project_secrets(self, project, project_secret):
        secrets = await project.secrets.list()
        assert len(secrets) > 0

    async def test_project_secret_properties(self, project, project_secret):
        assert isinstance(project_secret.project_uuid, str)
        assert isinstance(project_secret.name, str)
        assert project_secret.name == "secret_value"
        assert project.id == project_secret.project_uuid

    async def test_project_secret_str(self, project_secret):
        string = str(project_secret)
        regex = r"^{.*\"name\": \"" + project_secret.name + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_secret_repr(self, project_secret):
        string = repr(project_secret)
        regex = (
            r"^ProjectSecret\( trainml , \*\*{.*'name': '"
            + project_secret.name
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
