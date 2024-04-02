import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.volumes as specimen
from trainml.exceptions import (
    ApiError,
    VolumeError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.volumes]


@fixture
def volumes(mock_trainml):
    yield specimen.Volumes(mock_trainml)


@fixture
def volume(mock_trainml):
    yield specimen.Volume(
        mock_trainml,
        id="1",
        project_uuid="proj-id-1",
        name="first one",
        status="downloading",
        capacity="10G",
        used_size=100000000,
        billed_size=100000000,
        createdAt="2020-12-31T23:59:59.000Z",
    )


class VolumesTests:
    @mark.asyncio
    async def test_get_volume(
        self,
        volumes,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await volumes.get("1234")
        mock_trainml._query.assert_called_once_with("/volume/1234", "GET", dict())

    @mark.asyncio
    async def test_list_volumes(
        self,
        volumes,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await volumes.list()
        mock_trainml._query.assert_called_once_with("/volume", "GET", dict())

    @mark.asyncio
    async def test_remove_volume(
        self,
        volumes,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await volumes.remove("4567")
        mock_trainml._query.assert_called_once_with(
            "/volume/4567", "DELETE", dict(force=True)
        )

    @mark.asyncio
    async def test_create_volume_simple(self, volumes, mock_trainml):
        requested_config = dict(
            name="new volume",
            source_type="aws",
            source_uri="s3://trainml-examples/volumes/resnet50",
            capacity="10G",
        )
        expected_payload = dict(
            project_uuid="proj-id-1",
            name="new volume",
            source_type="aws",
            source_uri="s3://trainml-examples/volumes/resnet50",
            capacity="10G",
        )
        api_response = {
            "project_uuid": "cus-id-1",
            "id": "volume-id-1",
            "name": "new volume",
            "status": "new",
            "source_type": "aws",
            "capacity": "10G",
            "source_uri": "s3://trainml-examples/volumes/resnet50",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await volumes.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/volume", "POST", None, expected_payload
        )
        assert response.id == "volume-id-1"


class VolumeTests:
    def test_volume_properties(self, volume):
        assert isinstance(volume.id, str)
        assert isinstance(volume.status, str)
        assert isinstance(volume.name, str)
        assert isinstance(volume.capacity, str)
        assert isinstance(volume.used_size, int)
        assert isinstance(volume.billed_size, int)

    def test_volume_str(self, volume):
        string = str(volume)
        regex = r"^{.*\"id\": \"" + volume.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_volume_repr(self, volume):
        string = repr(volume)
        regex = r"^Volume\( trainml , \*\*{.*'id': '" + volume.id + r"'.*}\)$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_volume_bool(self, volume, mock_trainml):
        empty_volume = specimen.Volume(mock_trainml)
        assert bool(volume)
        assert not bool(empty_volume)

    @mark.asyncio
    async def test_volume_get_log_url(self, volume, mock_trainml):
        api_response = (
            "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/logs/first_one.zip"
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await volume.get_log_url()
        mock_trainml._query.assert_called_once_with(
            "/volume/1/logs", "GET", dict(project_uuid="proj-id-1")
        )
        assert response == api_response

    @mark.asyncio
    async def test_volume_get_details(self, volume, mock_trainml):
        api_response = {
            "type": "directory",
            "name": "/",
            "count": "8",
            "used_size": "177M",
            "contents": [],
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await volume.get_details()
        mock_trainml._query.assert_called_once_with(
            "/volume/1/details", "GET", dict(project_uuid="proj-id-1")
        )
        assert response == api_response

    @mark.asyncio
    async def test_volume_get_connection_utility_url(self, volume, mock_trainml):
        api_response = (
            "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/vpn/first_one.zip"
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await volume.get_connection_utility_url()
        mock_trainml._query.assert_called_once_with(
            "/volume/1/download", "GET", dict(project_uuid="proj-id-1")
        )
        assert response == api_response

    def test_volume_get_connection_details_no_vpn(self, volume):
        details = volume.get_connection_details()
        expected_details = dict()
        assert details == expected_details

    def test_volume_get_connection_details_local_data(self, mock_trainml):
        volume = specimen.Volume(
            mock_trainml,
            id="1",
            project_uuid="a",
            name="first one",
            status="new",
            capacity="10G",
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
        details = volume.get_connection_details()
        expected_details = dict(
            project_uuid="a",
            entity_type="volume",
            cidr="10.106.171.0/24",
            ssh_port=46600,
            input_path="~/tensorflow-example",
            output_path=None,
        )
        assert details == expected_details

    @mark.asyncio
    async def test_volume_connect(self, volume, mock_trainml):
        with patch(
            "trainml.volumes.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "connected"
            resp = await volume.connect()
            connection.start.assert_called_once()
            assert resp == "connected"

    @mark.asyncio
    async def test_volume_disconnect(self, volume, mock_trainml):
        with patch(
            "trainml.volumes.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "removed"
            resp = await volume.disconnect()
            connection.stop.assert_called_once()
            assert resp == "removed"

    @mark.asyncio
    async def test_volume_remove(self, volume, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await volume.remove()
        mock_trainml._query.assert_called_once_with(
            "/volume/1",
            "DELETE",
            dict(project_uuid="proj-id-1", force=False),
        )

    def test_volume_default_ws_msg_handler(self, volume, capsys):
        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = volume._get_msg_handler(None)
        handler(data)
        captured = capsys.readouterr()
        assert (
            captured.out
            == "02/11/2021, 15:35:45: download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n"
        )

    def test_volume_custom_ws_msg_handler(self, volume, capsys):
        def custom_handler(msg):
            print(msg.get("stream"))

        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = volume._get_msg_handler(custom_handler)
        handler(data)
        captured = capsys.readouterr()
        assert captured.out == "worker-id-1\n"

    @mark.asyncio
    async def test_volume_attach(self, volume, mock_trainml):
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "id": "data-id-1",
            "name": "new volume",
            "status": "downloading",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        volume.refresh = AsyncMock(return_value=refresh_response)
        await volume.attach()
        mock_trainml._ws_subscribe.assert_called_once()

    @mark.asyncio
    async def test_volume_attach_immediate_return(self, mock_trainml):
        volume = specimen.Volume(
            mock_trainml,
            id="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "id": "1",
            "name": "new volume",
            "status": "ready",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        volume.refresh = AsyncMock(return_value=refresh_response)
        await volume.attach()
        mock_trainml._ws_subscribe.assert_not_called()

    @mark.asyncio
    async def test_volume_refresh(self, volume, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "id": "data-id-1",
            "name": "new volume",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await volume.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/volume/1", "GET", dict(project_uuid="proj-id-1")
        )
        assert volume.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_volume_wait_for_successful(self, volume, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "id": "data-id-1",
            "name": "new volume",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await volume.wait_for("ready")
        mock_trainml._query.assert_called_once_with(
            f"/volume/1", "GET", dict(project_uuid="proj-id-1")
        )
        assert volume.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_volume_wait_for_current_status(self, mock_trainml):
        volume = specimen.Volume(
            mock_trainml,
            id="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await volume.wait_for("ready")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_volume_wait_for_incorrect_status(self, volume, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(SpecificationError):
            await volume.wait_for("stopped")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_volume_wait_for_with_delay(self, volume, mock_trainml):
        api_response_initial = dict(
            id="1",
            name="first one",
            status="new",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response_final = dict(
            id="1",
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
        response = await volume.wait_for("ready")
        assert volume.status == "ready"
        assert response.status == "ready"

    @mark.asyncio
    async def test_volume_wait_for_timeout(self, volume, mock_trainml):
        api_response = dict(
            id="1",
            name="first one",
            status="downloading",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(TrainMLException):
            await volume.wait_for("ready", 10)
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_volume_wait_for_volume_failed(self, volume, mock_trainml):
        api_response = dict(
            id="1",
            name="first one",
            status="failed",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(VolumeError):
            await volume.wait_for("ready")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_volume_wait_for_archived_succeeded(self, volume, mock_trainml):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Volume Not Found"))
        )
        await volume.wait_for("archived")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_volume_wait_for_unexpected_api_error(self, volume, mock_trainml):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Volume Not Found"))
        )
        with raises(ApiError):
            await volume.wait_for("ready")
        mock_trainml._query.assert_called()
