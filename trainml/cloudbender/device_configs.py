import json
import logging


class DeviceConfigs(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, provider_uuid, region_uuid, id, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device/config/{id}",
            "GET",
            kwargs,
        )
        return DeviceConfig(self.trainml, **resp)

    async def list(self, provider_uuid, region_uuid, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device/config",
            "GET",
            kwargs,
        )
        device_configs = [
            DeviceConfig(self.trainml, **device_config)
            for device_config in resp
        ]
        return device_configs

    async def create(
        self,
        provider_uuid,
        region_uuid,
        name,
        model_uuid,
        image,
        command,
        **kwargs,
    ):
        logging.info(f"Creating Device Config {name}")
        data = dict(
            name=name,
            model_uuid=model_uuid,
            image=image,
            command=command,
            **kwargs,
        )
        if not data.get("model_project_uuid"):
            data["model_project_uuid"] = self.trainml.active_project

        payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device/config",
            "POST",
            None,
            payload,
        )
        device_config = DeviceConfig(self.trainml, **resp)
        logging.info(
            f"Created Device Config {name} with id {device_config.id}"
        )
        return device_config

    async def remove(self, provider_uuid, region_uuid, id, **kwargs):
        await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device/config/{id}",
            "DELETE",
            kwargs,
        )


class DeviceConfig:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._device_config = kwargs
        self._id = self._device_config.get("config_id")
        self._provider_uuid = self._device_config.get("provider_uuid")
        self._region_uuid = self._device_config.get("region_uuid")
        self._name = self._device_config.get("name")

    @property
    def id(self) -> str:
        return self._id

    @property
    def provider_uuid(self) -> str:
        return self._provider_uuid

    @property
    def region_uuid(self) -> str:
        return self._region_uuid

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return json.dumps({k: v for k, v in self._device_config.items()})

    def __repr__(self):
        return f"DeviceConfig( trainml , **{self._device_config.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/config/{self._id}",
            "DELETE",
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/config/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self
