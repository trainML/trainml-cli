import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.device_configs as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.device_configs]


@fixture
def device_configs(mock_trainml):
    yield specimen.DeviceConfigs(mock_trainml)


@fixture
def device_config(mock_trainml):
    yield specimen.DeviceConfig(
        mock_trainml,
        provider_uuid="1",
        region_uuid="a",
        config_id="x",
        name="On-Prem DeviceConfig",
        model_uuid="model-id-1",
        image="nvidia/cuda",
        command="python run.py",
    )


class RegionsTests:
    @mark.asyncio
    async def test_get_device_config(
        self,
        device_configs,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device_configs.get("1234", "5687", "91011")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/device/config/91011", "GET", {}
        )

    @mark.asyncio
    async def test_list_device_configs(
        self,
        device_configs,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device_configs.list("1234", "5687")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687/device/config", "GET", {}
        )

    @mark.asyncio
    async def test_remove_device_config(
        self,
        device_configs,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device_configs.remove("1234", "4567", "8910")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/4567/device/config/8910", "DELETE", {}
        )

    @mark.asyncio
    async def test_create_device_config(self, device_configs, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            region_uuid="region-id-1",
            name="IoT 1",
            model_uuid="model-id-1",
            image="nvidia/cuda",
            command="python run.py",
        )
        expected_payload = dict(
            name="IoT 1",
            model_uuid="model-id-1",
            model_project_uuid="proj-id-1",
            image="nvidia/cuda",
            command="python run.py",
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "config_id": "device_config-id-1",
            "name": "IoT 1",
            "model_uuid": "model-id-1",
            "model_project_uuid": "proj-id-1",
            "image": "nvidia/cuda",
            "command": "python run.py",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await device_configs.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider/provider-id-1/region/region-id-1/device/config",
            "POST",
            None,
            expected_payload,
        )
        assert response.id == "device_config-id-1"


class device_configTests:
    def test_device_config_properties(self, device_config):
        assert isinstance(device_config.id, str)
        assert isinstance(device_config.provider_uuid, str)
        assert isinstance(device_config.region_uuid, str)
        assert isinstance(device_config.name, str)

    def test_device_config_str(self, device_config):
        string = str(device_config)
        regex = r"^{.*\"config_id\": \"" + device_config.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_device_config_repr(self, device_config):
        string = repr(device_config)
        regex = (
            r"^DeviceConfig\( trainml , \*\*{.*'config_id': '"
            + device_config.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_device_config_bool(self, device_config, mock_trainml):
        empty_device_config = specimen.DeviceConfig(mock_trainml)
        assert bool(device_config)
        assert not bool(empty_device_config)

    @mark.asyncio
    async def test_device_config_remove(self, device_config, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await device_config.remove()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/device/config/x", "DELETE"
        )

    @mark.asyncio
    async def test_device_config_refresh(self, device_config, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "config_id": "device_config-id-1",
            "name": "IoT 1",
            "model_uuid": "model-id-1",
            "model_project_uuid": "proj-id-1",
            "image": "nvidia/cuda",
            "command": "python run.py",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await device_config.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/provider/1/region/a/device/config/x", "GET"
        )
        assert device_config.id == "device_config-id-1"
        assert response.id == "device_config-id-1"
