import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.projects.members as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.projects]


@fixture
def project_members(mock_trainml):
    yield specimen.ProjectMembers(mock_trainml, project_id="1")


@fixture
def project_member(mock_trainml):
    yield specimen.ProjectMember(
        mock_trainml,
        id="owner@gmail.com",
        email="owner@gmail.com",
        project_uuid="proj-id-1",
       owner= True,
                job= "all",
                dataset= "all",
                model= "all",
                checkpoint="all",
                volume= "all"
    )


class ProjectMembersTests:
    @mark.asyncio
    async def test_project_members_add(self, project_members, mock_trainml):
        """Test add method (lines 18-31)."""
        api_response = {
            "project_uuid": "proj-id-1",
            "email": "newuser@gmail.com",
            "owner": False,
            "job": "all",
            "dataset": "read",
            "model": "all",
            "checkpoint": "read",
            "volume": "all",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        result = await project_members.add(
            email="newuser@gmail.com",
            job="all",
            dataset="read",
            model="all",
            checkpoint="read",
            volume="all",
            param1="value1",
        )
        expected_payload = dict(
            email="newuser@gmail.com",
            job="all",
            dataset="read",
            model="all",
            checkpoint="read",
            volume="all",
        )
        mock_trainml._query.assert_called_once_with(
            "/project/1/access", "POST", dict(param1="value1"), expected_payload
        )
        assert result.email == "newuser@gmail.com"

    @mark.asyncio
    async def test_project_members_list(self, project_members, mock_trainml):
        api_response = [
            {
                "project_uuid": "proj-id-1",
                "email": "owner@gmail.com",
                "createdAt": "2024-09-04T00:42:39.529Z",
                "updatedAt": "2024-09-04T00:42:39.529Z",
                "owner": True,
                "job": "all",
                "dataset": "all",
                "model": "all",
                "checkpoint": "all",
                "volume": "all"
            },
            {
                "project_uuid": "proj-id-1",
                "email": "non-owner@gmail.com",
                "createdAt": "2024-09-04T00:42:39.529Z",
                "updatedAt": "2024-09-04T00:42:39.529Z",
                "owner": False,
                "job": "all",
                "dataset": "all",
                "model": "all",
                "checkpoint": "read",
                "volume": "read"
            },
        ]
        mock_trainml._query = AsyncMock(return_value=api_response)
        resp = await project_members.list()
        mock_trainml._query.assert_called_once_with(
            "/project/1/access", "GET", dict()
        )
        assert len(resp) == 2

    @mark.asyncio
    async def test_project_members_remove(self, project_members, mock_trainml):
        """Test remove method (line 35)."""
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await project_members.remove("user@gmail.com", param1="value1")
        mock_trainml._query.assert_called_once_with(
            "/project/1/access", "DELETE", dict(param1="value1", email="user@gmail.com")
        )


class ProjectMemberTests:
    def test_project_member_properties(self, project_member):
        assert isinstance(project_member.id, str)
        assert isinstance(project_member.email, str)
        assert isinstance(project_member.project_uuid, str)
        assert isinstance(project_member.owner, bool)
        assert isinstance(project_member.job, str)
        assert isinstance(project_member.dataset, str)
        assert isinstance(project_member.model, str)
        assert isinstance(project_member.checkpoint, str)
        assert isinstance(project_member.volume, str)

    def test_project_member_str(self, project_member):
        string = str(project_member)
        regex = r"^{.*\"id\": \"" + project_member.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_member_repr(self, project_member):
        string = repr(project_member)
        regex = (
            r"^ProjectMember\( trainml , \*\*{.*'id': '"
            + project_member.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_project_member_bool(self, project_member, mock_trainml):
        empty_project_member = specimen.ProjectMember(mock_trainml)
        assert bool(project_member)
        assert not bool(empty_project_member)
