import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
class ProjectCredentialsTests:
    @fixture(scope="class")
    async def project_credential(self, project):
        project_credential = await project.credentials.put(
            type="aws", key_id="ASFHALKF", secret="IUHKLHKAHF"
        )
        yield project_credential
        await project.credentials.remove(type="aws")

    async def test_list_project_credentials(self, project, project_credential):
        credentials = await project.credentials.list()
        assert len(credentials) > 0

    async def test_project_credential_properties(self, project, project_credential):
        assert isinstance(project_credential.project_uuid, str)
        assert isinstance(project_credential.type, str)
        assert isinstance(project_credential.key_id, str)
        assert project_credential.type == "aws"
        assert project.id == project_credential.project_uuid

    async def test_project_credential_str(self, project_credential):
        string = str(project_credential)
        regex = r"^{.*\"type\": \"" + project_credential.type + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_credential_repr(self, project_credential):
        string = repr(project_credential)
        regex = (
            r"^ProjectCredential\( trainml , \*\*{.*'type': '"
            + project_credential.type
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
