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
        type="evefs",
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
        mock_trainml._query.assert_called_once_with(
            "/volume/1234", "GET", dict()
        )

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
            type="evefs",
        )
        api_response = {
            "project_uuid": "cus-id-1",
            "id": "volume-id-1",
            "name": "new volume",
            "status": "new",
            "type": "evefs",
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
        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/logs/first_one.zip"
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
    async def test_volume_connect_downloading_status(self, mock_trainml):
        volume = specimen.Volume(
            mock_trainml,
            id="1",
            project_uuid="proj-id-1",
            name="test volume",
            status="downloading",
            auth_token="test-token",
            hostname="example.com",
            source_uri="/path/to/source",
        )

        with patch(
            "trainml.volumes.Volume.refresh", new_callable=AsyncMock
        ) as mock_refresh:
            with patch(
                "trainml.volumes.upload", new_callable=AsyncMock
            ) as mock_upload:
                await volume.connect()
                mock_refresh.assert_called_once()
                mock_upload.assert_called_once_with(
                    "example.com", "test-token", "/path/to/source"
                )

    @mark.asyncio
    async def test_volume_connect_exporting_status(
        self, mock_trainml, tmp_path
    ):
        output_dir = str(tmp_path / "output")
        volume = specimen.Volume(
            mock_trainml,
            id="1",
            project_uuid="proj-id-1",
            name="test volume",
            status="exporting",
            auth_token="test-token",
            hostname="example.com",
            output_uri=output_dir,
        )

        with patch(
            "trainml.volumes.Volume.refresh", new_callable=AsyncMock
        ) as mock_refresh:
            with patch(
                "trainml.volumes.download", new_callable=AsyncMock
            ) as mock_download:
                await volume.connect()
                mock_refresh.assert_called_once()
                mock_download.assert_called_once_with(
                    "example.com", "test-token", output_dir
                )

    @mark.asyncio
    async def test_volume_connect_invalid_status(self, mock_trainml):
        volume = specimen.Volume(
            mock_trainml,
            id="1",
            project_uuid="proj-id-1",
            name="test volume",
            status="ready",
        )

        with raises(
            SpecificationError,
            match="You can only connect to downloading or exporting volumes",
        ):
            await volume.connect()

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
    async def test_volume_wait_for_incorrect_status(
        self, volume, mock_trainml
    ):
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
    async def test_volume_wait_for_archived_succeeded(
        self, volume, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Volume Not Found"))
        )
        await volume.wait_for("archived")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_volume_wait_for_unexpected_api_error(
        self, volume, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Volume Not Found"))
        )
        with raises(ApiError):
            await volume.wait_for("ready")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_volume_rename(self, volume, mock_trainml):
        api_response = dict(
            id="1",
            name="renamed volume",
            project_uuid="proj-id-1",
            status="ready",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        result = await volume.rename("renamed volume")
        mock_trainml._query.assert_called_once_with(
            "/volume/1",
            "PATCH",
            dict(project_uuid="proj-id-1"),
            dict(name="renamed volume"),
        )
        assert result == volume
        assert volume.name == "renamed volume"

    @mark.asyncio
    async def test_volume_export(self, volume, mock_trainml):
        api_response = dict(
            id="1",
            name="first one",
            project_uuid="proj-id-1",
            status="exporting",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        result = await volume.export("aws", "s3://bucket/path", dict(key="value"))
        mock_trainml._query.assert_called_once_with(
            "/volume/1/export",
            "POST",
            dict(project_uuid="proj-id-1"),
            dict(
                output_type="aws",
                output_uri="s3://bucket/path",
                output_options=dict(key="value"),
            ),
        )
        assert result == volume
        assert volume.status == "exporting"

    @mark.asyncio
    async def test_volume_export_default_options(self, volume, mock_trainml):
        api_response = dict(
            id="1",
            name="first one",
            project_uuid="proj-id-1",
            status="exporting",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        result = await volume.export("aws", "s3://bucket/path")
        mock_trainml._query.assert_called_once_with(
            "/volume/1/export",
            "POST",
            dict(project_uuid="proj-id-1"),
            dict(
                output_type="aws",
                output_uri="s3://bucket/path",
                output_options=dict(),
            ),
        )
        assert result == volume

    @mark.asyncio
    async def test_volume_wait_for_timeout_validation(
        self, volume, mock_trainml
    ):
        with raises(SpecificationError) as exc_info:
            await volume.wait_for("ready", timeout=25 * 60 * 60)  # > 24 hours
        assert "timeout" in str(exc_info.value.attribute).lower()
        assert "less than" in str(exc_info.value.message).lower()

    @mark.asyncio
    async def test_volume_connect_new_status_waits_for_downloading(
        self, volume, mock_trainml
    ):
        """Test that connect() waits for downloading status when status is 'new'."""
        volume._volume["status"] = "new"
        volume._status = "new"
        api_response_new = dict(
            id="1",
            name="first one",
            status="new",
        )
        api_response_downloading = dict(
            id="1",
            name="first one",
            status="downloading",
            auth_token="token",
            hostname="host",
            source_uri="s3://bucket/path",
        )
        # wait_for calls refresh multiple times, then connect calls refresh again
        mock_trainml._query = AsyncMock(
            side_effect=[
                api_response_new,  # wait_for refresh 1
                api_response_downloading,  # wait_for refresh 2 (status matches, wait_for returns)
                api_response_downloading,  # connect refresh
            ]
        )
        with patch("trainml.volumes.upload", new_callable=AsyncMock) as mock_upload:
            await volume.connect()
        # After refresh, status should be downloading
        assert volume.status == "downloading"
        mock_upload.assert_called_once()

    @mark.asyncio
    async def test_volume_connect_downloading_missing_properties(
        self, volume, mock_trainml
    ):
        """Test connect() raises error when downloading status missing properties."""
        volume._volume["status"] = "downloading"
        api_response = dict(
            id="1",
            name="first one",
            status="downloading",
            # Missing auth_token, hostname, or source_uri
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(SpecificationError) as exc_info:
            await volume.connect()
        assert "missing required connection properties" in str(exc_info.value.message).lower()

    @mark.asyncio
    async def test_volume_connect_exporting_missing_properties(
        self, volume, mock_trainml
    ):
        """Test connect() raises error when exporting status missing properties."""
        volume._volume["status"] = "exporting"
        api_response = dict(
            id="1",
            name="first one",
            status="exporting",
            # Missing auth_token, hostname, or output_uri
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(SpecificationError) as exc_info:
            await volume.connect()
        assert "missing required connection properties" in str(exc_info.value.message).lower()
