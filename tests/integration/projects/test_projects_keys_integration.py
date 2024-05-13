import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
class ProjectKeysTests:
    @fixture(scope="class")
    async def project_key(self, project):
        project_key = await project.keys.put(
            type="aws", key_id="ASFHALKF", secret="IUHKLHKAHF"
        )
        yield project_key
        await project.keys.remove(type="aws")

    async def test_list_project_keys(self, project, project_key):
        keys = await project.keys.list()
        assert len(keys) > 0

    async def test_project_key_properties(self, project, project_key):
        assert isinstance(project_key.project_uuid, str)
        assert isinstance(project_key.type, str)
        assert isinstance(project_key.key_id, str)
        assert project_key.type == "aws"
        assert project.id == project_key.project_uuid

    async def test_project_key_str(self, project_key):
        string = str(project_key)
        regex = r"^{.*\"type\": \"" + project_key.type + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_key_repr(self, project_key):
        string = repr(project_key)
        regex = (
            r"^ProjectKey\( trainml , \*\*{.*'type': '" + project_key.type + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
