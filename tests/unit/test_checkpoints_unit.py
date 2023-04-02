import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.checkpoints as specimen
from trainml.exceptions import (
    ApiError,
    CheckpointError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.checkpoints]


@fixture
def checkpoints(mock_trainml):
    yield specimen.Checkpoints(mock_trainml)


@fixture
def checkpoint(mock_trainml):
    yield specimen.Checkpoint(
        mock_trainml,
        checkpoint_uuid="1",
        project_uuid="proj-id-1",
        name="first one",
        status="downloading",
        size=100000,
        createdAt="2020-12-31T23:59:59.000Z",
    )


class CheckpointsTests:
    @mark.asyncio
    async def test_get_checkpoint(
        self,
        checkpoints,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await checkpoints.get("1234")
        mock_trainml._query.assert_called_once_with(
            "/checkpoint/1234", "GET", dict()
        )

    @mark.asyncio
    async def test_list_checkpoints(
        self,
        checkpoints,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await checkpoints.list()
        mock_trainml._query.assert_called_once_with(
            "/checkpoint", "GET", dict()
        )

    @mark.asyncio
    async def test_remove_checkpoint(
        self,
        checkpoints,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await checkpoints.remove("4567")
        mock_trainml._query.assert_called_once_with(
            "/checkpoint/4567", "DELETE", dict(force=True)
        )

    @mark.asyncio
    async def test_create_checkpoint_simple(self, checkpoints, mock_trainml):
        requested_config = dict(
            name="new checkpoint",
            source_type="aws",
            source_uri="s3://trainml-examples/checkpoints/resnet50",
        )
        expected_payload = dict(
            project_uuid="proj-id-1",
            name="new checkpoint",
            source_type="aws",
            source_uri="s3://trainml-examples/checkpoints/resnet50",
        )
        api_response = {
            "project_uuid": "cus-id-1",
            "checkpoint_uuid": "checkpoint-id-1",
            "name": "new checkpoint",
            "status": "new",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/checkpoints/resnet50",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await checkpoints.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/checkpoint", "POST", None, expected_payload
        )
        assert response.id == "checkpoint-id-1"


class CheckpointTests:
    def test_checkpoint_properties(self, checkpoint):
        assert isinstance(checkpoint.id, str)
        assert isinstance(checkpoint.status, str)
        assert isinstance(checkpoint.name, str)
        assert isinstance(checkpoint.size, int)

    def test_checkpoint_str(self, checkpoint):
        string = str(checkpoint)
        regex = r"^{.*\"checkpoint_uuid\": \"" + checkpoint.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_checkpoint_repr(self, checkpoint):
        string = repr(checkpoint)
        regex = (
            r"^Checkpoint\( trainml , \*\*{.*'checkpoint_uuid': '"
            + checkpoint.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_checkpoint_bool(self, checkpoint, mock_trainml):
        empty_checkpoint = specimen.Checkpoint(mock_trainml)
        assert bool(checkpoint)
        assert not bool(empty_checkpoint)

    @mark.asyncio
    async def test_checkpoint_get_log_url(self, checkpoint, mock_trainml):
        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/logs/first_one.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await checkpoint.get_log_url()
        mock_trainml._query.assert_called_once_with(
            "/checkpoint/1/logs", "GET", dict(project_uuid="proj-id-1")
        )
        assert response == api_response

    @mark.asyncio
    async def test_checkpoint_get_details(self, checkpoint, mock_trainml):
        api_response = {
            "type": "directory",
            "name": "/",
            "count": "8",
            "size": "177M",
            "contents": [],
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await checkpoint.get_details()
        mock_trainml._query.assert_called_once_with(
            "/checkpoint/1/details", "GET", dict(project_uuid="proj-id-1")
        )
        assert response == api_response

    @mark.asyncio
    async def test_checkpoint_get_connection_utility_url(
        self, checkpoint, mock_trainml
    ):
        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/vpn/first_one.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await checkpoint.get_connection_utility_url()
        mock_trainml._query.assert_called_once_with(
            "/checkpoint/1/download", "GET", dict(project_uuid="proj-id-1")
        )
        assert response == api_response

    def test_checkpoint_get_connection_details_no_vpn(self, checkpoint):
        details = checkpoint.get_connection_details()
        expected_details = dict()
        assert details == expected_details

    def test_checkpoint_get_connection_details_local_data(self, mock_trainml):
        checkpoint = specimen.Checkpoint(
            mock_trainml,
            checkpoint_uuid="1",
            project_uuid="a",
            name="first one",
            status="new",
            size=100000,
            createdAt="2020-12-31T23:59:59.000Z",
            source_type="local",
            source_uri="~/tensorflow-example",
            vpn={
                "status": "new",
                "cidr": "10.106.171.0/24",
                "client": {
                    "port": "36017",
                    "id": "cus-id-1",
                    "address": "10.106.171.253",
                    "ssh_port": 46600,
                },
                "net_prefix_type_id": 1,
            },
        )
        details = checkpoint.get_connection_details()
        expected_details = dict(
            project_uuid="a",
            entity_type="checkpoint",
            cidr="10.106.171.0/24",
            ssh_port=46600,
            input_path="~/tensorflow-example",
            output_path=None,
        )
        assert details == expected_details

    @mark.asyncio
    async def test_checkpoint_connect(self, checkpoint, mock_trainml):
        with patch(
            "trainml.checkpoints.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "connected"
            resp = await checkpoint.connect()
            connection.start.assert_called_once()
            assert resp == "connected"

    @mark.asyncio
    async def test_checkpoint_disconnect(self, checkpoint, mock_trainml):
        with patch(
            "trainml.checkpoints.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "removed"
            resp = await checkpoint.disconnect()
            connection.stop.assert_called_once()
            assert resp == "removed"

    @mark.asyncio
    async def test_checkpoint_remove(self, checkpoint, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await checkpoint.remove()
        mock_trainml._query.assert_called_once_with(
            "/checkpoint/1",
            "DELETE",
            dict(project_uuid="proj-id-1", force=False),
        )

    def test_checkpoint_default_ws_msg_handler(self, checkpoint, capsys):
        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = checkpoint._get_msg_handler(None)
        handler(data)
        captured = capsys.readouterr()
        assert (
            captured.out
            == "02/11/2021, 15:35:45: download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n"
        )

    def test_checkpoint_custom_ws_msg_handler(self, checkpoint, capsys):
        def custom_handler(msg):
            print(msg.get("stream"))

        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = checkpoint._get_msg_handler(custom_handler)
        handler(data)
        captured = capsys.readouterr()
        assert captured.out == "worker-id-1\n"

    @mark.asyncio
    async def test_checkpoint_attach(self, checkpoint, mock_trainml):
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "checkpoint_uuid": "data-id-1",
            "name": "new checkpoint",
            "status": "downloading",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        checkpoint.refresh = AsyncMock(return_value=refresh_response)
        await checkpoint.attach()
        mock_trainml._ws_subscribe.assert_called_once()

    @mark.asyncio
    async def test_checkpoint_attach_immediate_return(self, mock_trainml):
        checkpoint = specimen.Checkpoint(
            mock_trainml,
            checkpoint_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "checkpoint_uuid": "1",
            "name": "new checkpoint",
            "status": "ready",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        checkpoint.refresh = AsyncMock(return_value=refresh_response)
        await checkpoint.attach()
        mock_trainml._ws_subscribe.assert_not_called()

    @mark.asyncio
    async def test_checkpoint_refresh(self, checkpoint, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "checkpoint_uuid": "data-id-1",
            "name": "new checkpoint",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await checkpoint.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/checkpoint/1", "GET", dict(project_uuid="proj-id-1")
        )
        assert checkpoint.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_checkpoint_wait_for_successful(
        self, checkpoint, mock_trainml
    ):
        api_response = {
            "customer_uuid": "cus-id-1",
            "checkpoint_uuid": "data-id-1",
            "name": "new checkpoint",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await checkpoint.wait_for("ready")
        mock_trainml._query.assert_called_once_with(
            f"/checkpoint/1", "GET", dict(project_uuid="proj-id-1")
        )
        assert checkpoint.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_checkpoint_wait_for_current_status(self, mock_trainml):
        checkpoint = specimen.Checkpoint(
            mock_trainml,
            checkpoint_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await checkpoint.wait_for("ready")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_checkpoint_wait_for_incorrect_status(
        self, checkpoint, mock_trainml
    ):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(SpecificationError):
            await checkpoint.wait_for("stopped")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_checkpoint_wait_for_with_delay(
        self, checkpoint, mock_trainml
    ):
        api_response_initial = dict(
            checkpoint_uuid="1",
            name="first one",
            status="new",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response_final = dict(
            checkpoint_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock()
        mock_trainml._query.side_effect = [
            api_response_initial,
            api_response_initial,
            api_response_final,
        ]
        response = await checkpoint.wait_for("ready")
        assert checkpoint.status == "ready"
        assert response.status == "ready"

    @mark.asyncio
    async def test_checkpoint_wait_for_timeout(self, checkpoint, mock_trainml):
        api_response = dict(
            checkpoint_uuid="1",
            name="first one",
            status="downloading",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(TrainMLException):
            await checkpoint.wait_for("ready", 10)
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_checkpoint_wait_for_checkpoint_failed(
        self, checkpoint, mock_trainml
    ):
        api_response = dict(
            checkpoint_uuid="1",
            name="first one",
            status="failed",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(CheckpointError):
            await checkpoint.wait_for("ready")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_checkpoint_wait_for_archived_succeeded(
        self, checkpoint, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(
                404, dict(errorMessage="Checkpoint Not Found")
            )
        )
        await checkpoint.wait_for("archived")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_checkpoint_wait_for_unexpected_api_error(
        self, checkpoint, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(
                404, dict(errorMessage="Checkpoint Not Found")
            )
        )
        with raises(ApiError):
            await checkpoint.wait_for("ready")
        mock_trainml._query.assert_called()
