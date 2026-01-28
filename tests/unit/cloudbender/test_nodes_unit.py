import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.nodes as specimen
from trainml.exceptions import (
    ApiError,
    NodeError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.nodes]


@fixture
def nodes(mock_trainml):
    yield specimen.Nodes(mock_trainml)


@fixture
def node(mock_trainml):
    yield specimen.Node(
        mock_trainml,
        provider_uuid="1",
        region_uuid="a",
        rig_uuid="x",
        type="permanent",
        service="compute",
        friendly_name="hq-a100-01",
        hostname="hq-a100-01",
        status="active",
        online=True,
        maintenance_mode=False,
    )


class RegionsTests:
    @mark.asyncio
    async def test_get_node(
        self,
        nodes,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await nodes.get("1234", "5687", "91011")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/node/91011", "GET", {}
        )

    @mark.asyncio
    async def test_list_nodes(
        self,
        nodes,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await nodes.list("1234", "5687")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/node", "GET", {}
        )

    @mark.asyncio
    async def test_remove_node(
        self,
        nodes,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await nodes.remove("1234", "4567", "8910")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/4567/node/8910", "DELETE", {}
        )

    @mark.asyncio
    async def test_create_node(self, nodes, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            region_uuid="region-id-1",
            friendly_name="phys-node",
            hostname="phys-node",
            minion_id="asdf",
        )
        expected_payload = dict(
            friendly_name="phys-node",
            hostname="phys-node",
            minion_id="asdf",
            type="permanent",
            service="compute",
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "rig_uuid": "rig-id-1",
            "name": "phys-node",
            "type": "permanent",
            "service": "compute",
            "status": "new",
            "online": False,
            "maintenance_mode": True,
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await nodes.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/node",
            "POST",
            None,
            expected_payload,
        )
        assert response.id == "rig-id-1"


class nodeTests:
    def test_node_properties(self, node):
        assert isinstance(node.id, str)
        assert isinstance(node.provider_uuid, str)
        assert isinstance(node.region_uuid, str)
        assert isinstance(node.type, str)
        assert isinstance(node.service, str)
        assert isinstance(node.name, str)
        assert isinstance(node.hostname, str)
        assert isinstance(node.status, str)
        assert isinstance(node.online, bool)
        assert isinstance(node.maintenance_mode, bool)

    def test_node_str(self, node):
        string = str(node)
        regex = r"^{.*\"rig_uuid\": \"" + node.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_node_repr(self, node):
        string = repr(node)
        regex = (
            r"^Node\( trainml , \*\*{.*'rig_uuid': '" + node.id + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_node_bool(self, node, mock_trainml):
        empty_node = specimen.Node(mock_trainml)
        assert bool(node)
        assert not bool(empty_node)

    @mark.asyncio
    async def test_node_remove(self, node, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await node.remove()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/node/x", "DELETE"
        )

    @mark.asyncio
    async def test_node_refresh(self, node, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "rig_uuid": "rig-id-1",
            "name": "phys-node",
            "type": "permanent",
            "service": "compute",
            "status": "new",
            "online": False,
            "maintenance_mode": True,
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await node.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/provider/1/region/a/node/x", "GET"
        )
        assert node.id == "rig-id-1"
        assert response.id == "rig-id-1"

    @mark.asyncio
    async def test_node_toggle_maintenance(self, node, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await node.toggle_maintenance()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/node/x/maintenance", "PATCH"
        )

    @mark.asyncio
    async def test_node_run_action(self, node, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await node.run_action(command="report")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/node/x/action",
            "POST",
            None,
            dict(command="report"),
        )

    @mark.asyncio
    async def test_node_wait_for_already_at_status(self, node):
        """Test wait_for returns immediately if already at target status."""
        node._status = "active"
        result = await node.wait_for("active")
        assert result is None

    @mark.asyncio
    async def test_node_wait_for_invalid_status(self, node):
        """Test wait_for raises error for invalid status."""
        with raises(SpecificationError) as exc_info:
            await node.wait_for("invalid_status")
        assert "Invalid wait_for status" in str(exc_info.value.message)

    @mark.asyncio
    async def test_node_wait_for_timeout_validation(self, node):
        """Test wait_for validates timeout (line 172)."""
        node._status = "new"  # Set to different status so timeout check runs
        with raises(SpecificationError) as exc_info:
            await node.wait_for("active", timeout=25 * 60 * 60)
        assert "timeout must be less than" in str(exc_info.value.message)

    @mark.asyncio
    async def test_node_wait_for_success(self, node, mock_trainml):
        """Test wait_for succeeds when status matches."""
        node._status = "new"
        api_response_new = dict(
            provider_uuid="1",
            region_uuid="a",
            rig_uuid="x",
            status="new",
        )
        api_response_active = dict(
            provider_uuid="1",
            region_uuid="a",
            rig_uuid="x",
            status="active",
        )
        mock_trainml._query = AsyncMock(
            side_effect=[api_response_new, api_response_active]
        )
        with patch("trainml.cloudbender.nodes.asyncio.sleep", new_callable=AsyncMock):
            result = await node.wait_for("active", timeout=10)
        assert result == node
        assert node.status == "active"

    @mark.asyncio
    async def test_node_wait_for_archived_404(self, node, mock_trainml):
        """Test wait_for handles 404 for archived status."""
        node._status = "active"
        api_error = ApiError(404, {"errorMessage": "Not found"})
        mock_trainml._query = AsyncMock(side_effect=api_error)
        with patch("trainml.cloudbender.nodes.asyncio.sleep", new_callable=AsyncMock):
            await node.wait_for("archived", timeout=10)

    @mark.asyncio
    async def test_node_wait_for_error_status(self, node, mock_trainml):
        """Test wait_for raises error for errored/failed status."""
        node._status = "new"
        api_response_errored = dict(
            provider_uuid="1",
            region_uuid="a",
            rig_uuid="x",
            status="errored",
        )
        mock_trainml._query = AsyncMock(return_value=api_response_errored)
        with patch("trainml.cloudbender.nodes.asyncio.sleep", new_callable=AsyncMock):
            with raises(NodeError):
                await node.wait_for("active", timeout=10)

    @mark.asyncio
    async def test_node_wait_for_timeout(self, node, mock_trainml):
        """Test wait_for raises timeout exception."""
        node._status = "new"
        api_response_new = dict(
            provider_uuid="1",
            region_uuid="a",
            rig_uuid="x",
            status="new",
        )
        mock_trainml._query = AsyncMock(return_value=api_response_new)
        with patch("trainml.cloudbender.nodes.asyncio.sleep", new_callable=AsyncMock):
            with raises(TrainMLException) as exc_info:
                await node.wait_for("active", timeout=0.1)
        assert "Timeout waiting for" in str(exc_info.value.message)

    @mark.asyncio
    async def test_node_wait_for_api_error_non_404(self, node, mock_trainml):
        """Test wait_for raises ApiError when not 404 for archived (line 189)."""
        node._status = "active"
        api_error = ApiError(500, {"errorMessage": "Server Error"})
        mock_trainml._query = AsyncMock(side_effect=api_error)
        with patch("trainml.cloudbender.nodes.asyncio.sleep", new_callable=AsyncMock):
            with raises(ApiError):
                await node.wait_for("archived", timeout=10)

    @mark.asyncio
    async def test_node_wait_for_failed_status(self, node, mock_trainml):
        """Test wait_for raises error for failed status (line 191)."""
        node._status = "new"
        api_response_failed = dict(
            provider_uuid="1",
            region_uuid="a",
            rig_uuid="x",
            status="failed",
        )
        mock_trainml._query = AsyncMock(return_value=api_response_failed)
        with patch("trainml.cloudbender.nodes.asyncio.sleep", new_callable=AsyncMock):
            with raises(NodeError):
                await node.wait_for("active", timeout=10)
