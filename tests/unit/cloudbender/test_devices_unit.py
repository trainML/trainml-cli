import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.devices as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.devices]


@fixture
def devices(mock_trainml):
    yield specimen.Devices(mock_trainml)


@fixture
def device(mock_trainml):
    yield specimen.Device(
        mock_trainml,
        provider_uuid="1",
        region_uuid="a",
        device_id="x",
        type="device",
        service="compute",
        friendly_name="hq-orin-01",
        hostname="hq-orin-01",
        status="active",
        online=True,
        maintenance_mode=False,
        job_status="stopped",
        job_last_deployed="2023-06-02T21:22:40.084Z",
        job_config_id="job-id-1",
        job_config_revision="1685740490096",
        device_config_id="conf-id-2",
    )


class RegionsTests:
    @mark.asyncio
    async def test_get_device(
        self,
        devices,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await devices.get("1234", "5687", "91011")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/device/91011", "GET", {}
        )

    @mark.asyncio
    async def test_list_devices(
        self,
        devices,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await devices.list("1234", "5687")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/device", "GET", {}
        )

    @mark.asyncio
    async def test_remove_device(
        self,
        devices,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await devices.remove("1234", "4567", "8910")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/4567/device/8910", "DELETE", {}
        )

    @mark.asyncio
    async def test_create_device(self, devices, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            region_uuid="region-id-1",
            friendly_name="phys-device",
            hostname="phys-device",
            minion_id="asdf",
        )
        expected_payload = dict(
            friendly_name="phys-device",
            hostname="phys-device",
            minion_id="asdf",
            type="device",
            service="compute",
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "device_id": "rig-id-1",
            "name": "phys-device",
            "type": "device",
            "service": "compute",
            "status": "new",
            "online": False,
            "maintenance_mode": True,
            "job_status": "stopped",
            "job_last_deployed": "2023-06-02T21:22:40.084Z",
            "job_config_id": "job-id-1",
            "job_config_revision": "1685740490096",
            "device_config_id": "conf-id-1",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await devices.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/device",
            "POST",
            None,
            expected_payload,
        )
        assert response.id == "rig-id-1"


class deviceTests:
    def test_device_properties(self, device):
        assert isinstance(device.id, str)
        assert isinstance(device.provider_uuid, str)
        assert isinstance(device.region_uuid, str)
        assert isinstance(device.name, str)
        assert isinstance(device.hostname, str)
        assert isinstance(device.status, str)
        assert isinstance(device.online, bool)
        assert isinstance(device.maintenance_mode, bool)
        assert isinstance(device.device_config_id, str)
        assert isinstance(device.job_status, str)
        assert isinstance(device.job_last_deployed, str)
        assert isinstance(device.job_config_id, str)
        assert isinstance(device.job_config_revision, str)

    def test_device_str(self, device):
        string = str(device)
        regex = r"^{.*\"device_id\": \"" + device.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_device_repr(self, device):
        string = repr(device)
        regex = (
            r"^Device\( trainml , \*\*{.*'device_id': '"
            + device.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_device_bool(self, device, mock_trainml):
        empty_device = specimen.Device(mock_trainml)
        assert bool(device)
        assert not bool(empty_device)

    @mark.asyncio
    async def test_device_remove(self, device, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device.remove()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/device/x", "DELETE"
        )

    @mark.asyncio
    async def test_device_refresh(self, device, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "device_id": "device-id-1",
            "name": "phys-device",
            "type": "device",
            "service": "compute",
            "status": "new",
            "online": False,
            "maintenance_mode": True,
            "job_status": "stopped",
            "job_last_deployed": "2023-06-02T21:22:40.084Z",
            "job_config_id": "job-id-1",
            "job_config_revision": "1685740490096",
            "device_config_id": "conf-id-1",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await device.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/provider/1/region/a/device/x", "GET"
        )
        assert device.id == "device-id-1"
        assert response.id == "device-id-1"

    @mark.asyncio
    async def test_device_toggle_maintenance(self, device, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device.toggle_maintenance()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/device/x/maintenance", "PATCH"
        )

    @mark.asyncio
    async def test_device_run_action(self, device, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device.run_action(command="report")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/device/x/action",
            "POST",
            None,
            dict(command="report"),
        )

    @mark.asyncio
    async def test_device_set_config(self, device, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "device_id": "device-id-1",
            "name": "phys-device",
            "type": "device",
            "service": "compute",
            "status": "new",
            "online": False,
            "maintenance_mode": True,
            "job_status": "stopped",
            "job_last_deployed": "2023-06-02T21:22:40.084Z",
            "job_config_id": "job-id-1",
            "job_config_revision": "1685740490096",
            "device_config_id": "config-id-1",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await device.set_config(device_config_id="config-id-1")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/device/x",
            "PATCH",
            None,
            dict(device_config_id="config-id-1"),
        )
        assert device.id == "device-id-1"
        assert response.id == "device-id-1"

    @mark.asyncio
    async def test_device_deploy_endpoint(self, device, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device.deploy_endpoint()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/device/x/deploy", "PUT"
        )

    @mark.asyncio
    async def test_device_stop_endpoint(self, device, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device.stop_endpoint()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/device/x/stop", "PUT"
        )
