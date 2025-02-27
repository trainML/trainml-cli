import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
@mark.xdist_group("project_resources")
class ProjectMembersTests:
    @fixture(scope="class")
    async def project_member(self, project):
        member = await project.members.add("test.account@proximl.ai","read","read","read","read","read")
        yield member
        await project.members.remove("test.account@proximl.ai")

    async def test_list_project_members(self, project):
        members = await project.members.list()
        assert len(members) > 0

    async def test_project_member_properties(self, project, project_member):
        assert isinstance(project_member.id, str)
        assert isinstance(project_member.email, str)
        assert isinstance(project_member.project_uuid, str)
        assert isinstance(project_member.owner, bool)
        assert isinstance(project_member.job, str)
        assert isinstance(project_member.dataset, str)
        assert isinstance(project_member.model, str)
        assert isinstance(project_member.checkpoint, str)
        assert isinstance(project_member.volume, str)
        assert project.id == project_member.project_uuid
        assert project_member.id == "test.account@proximl.ai"

    async def test_project_member_str(self, project_member):
        string = str(project_member)
        regex = r"^{.*\"email\": \"" + project_member.email + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_member_repr(self, project_member):
        string = repr(project_member)
        regex = (
            r"^ProjectMember\( trainml , \*\*{.*'email': '"
            + project_member.email
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
