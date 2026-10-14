import re
import sys
import asyncio
from datetime import datetime
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.projects]


@mark.create
@mark.asyncio
@mark.xdist_group("project_resources")
class ProjectAuthorizersTests:
    @fixture(scope="class")
    async def api_key_authorizer(self, project):
        authorizer = await project.authorizers.create(
            name="test_authorizer",
            type="api_key",
            keys=[dict(client_id="key-1", key="secret-key")],
        )
        yield authorizer
        await project.authorizers.remove(authorizer.id)

    @fixture(scope="class")
    async def oidc_authorizer(self, project):
        authorizer = await project.authorizers.create(
            name="test_authorizer_oidc",
            type="oidc",
            oidc=dict(
                issuer="https://auth.example.com",
                audience="trainml-test",
                algorithms=["RS256"],
            ),
        )
        yield authorizer
        await project.authorizers.remove(authorizer.id)

    async def test_list_project_authorizers(
        self, project, api_key_authorizer, oidc_authorizer
    ):
        authorizers = await project.authorizers.list()
        names = [authorizer.name for authorizer in authorizers]
        assert api_key_authorizer.name in names
        assert oidc_authorizer.name in names

    async def test_get_project_authorizer_by_name(self, project, api_key_authorizer):
        authorizer = await project.authorizers.get(api_key_authorizer.name)
        assert authorizer.id == api_key_authorizer.id
        assert authorizer.project_uuid == project.id

    async def test_get_project_authorizer_by_id(self, project, api_key_authorizer):
        authorizer = await project.authorizers.get(api_key_authorizer.id)
        assert authorizer.name == api_key_authorizer.name
        assert authorizer.project_uuid == project.id

    async def test_project_authorizer_properties(self, project, api_key_authorizer):
        assert isinstance(api_key_authorizer.id, str)
        assert isinstance(api_key_authorizer.name, str)
        assert isinstance(api_key_authorizer.type, str)
        assert isinstance(api_key_authorizer.project_uuid, str)
        assert isinstance(api_key_authorizer.keys, list)
        assert isinstance(api_key_authorizer.created_at, datetime)
        assert isinstance(api_key_authorizer.updated_at, datetime)
        assert api_key_authorizer.type == "api_key"
        assert project.id == api_key_authorizer.project_uuid

    async def test_project_authorizer_str(self, api_key_authorizer):
        string = str(api_key_authorizer)
        regex = r"^{.*\"name\": \"" + api_key_authorizer.name + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_project_authorizer_repr(self, api_key_authorizer):
        string = repr(api_key_authorizer)
        regex = (
            r"^ProjectAuthorizer\( trainml , \*\*{.*'name': '"
            + api_key_authorizer.name
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_oidc_authorizer_properties(self, oidc_authorizer):
        assert oidc_authorizer.type == "oidc"
        assert oidc_authorizer.oidc["issuer"] == "https://auth.example.com"
        assert oidc_authorizer.oidc["audience"] == "trainml-test"
        assert oidc_authorizer.oidc["algorithms"] == ["RS256"]
        assert oidc_authorizer.keys is None

    async def test_remove_project_authorizer(self, project):
        authorizer = await project.authorizers.create(
            name="test_authorizer_remove",
            type="api_key",
            keys=[dict(client_id="key-1", key="secret-key")],
        )
        await project.authorizers.remove(authorizer.id)
        authorizers = await project.authorizers.list()
        names = [a.name for a in authorizers]
        assert "test_authorizer_remove" not in names

    async def test_edit_project_authorizer(self, project, api_key_authorizer):
        original_keys = api_key_authorizer.keys
        updated_keys = original_keys + [dict(client_id="key-2", key="secret-key-2")]
        updated = await api_key_authorizer.edit(keys=updated_keys)
        assert len(updated.keys) == len(original_keys) + 1
        await updated.edit(keys=original_keys)
