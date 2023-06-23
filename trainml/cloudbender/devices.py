import json
import logging


class Devices(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, provider_uuid, region_uuid, id, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device/{id}",
            "GET",
            kwargs,
        )
        return Device(self.trainml, **resp)

    async def list(self, provider_uuid, region_uuid, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device",
            "GET",
            kwargs,
        )
        devices = [Device(self.trainml, **device) for device in resp]
        return devices

    async def create(
        self,
        provider_uuid,
        region_uuid,
        friendly_name,
        hostname,
        minion_id,
        **kwargs,
    ):
        logging.info(f"Creating Device {friendly_name}")
        data = dict(
            friendly_name=friendly_name,
            hostname=hostname,
            minion_id=minion_id,
            type="device",
            service="compute",
            **kwargs,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device",
            "POST",
            None,
            payload,
        )
        device = Device(self.trainml, **resp)
        logging.info(f"Created Device {friendly_name} with id {device.id}")
        return device

    async def remove(self, provider_uuid, region_uuid, id, **kwargs):
        await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/device/{id}",
            "DELETE",
            kwargs,
        )


class Device:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._device = kwargs
        self._id = self._device.get("device_id")
        self._provider_uuid = self._device.get("provider_uuid")
        self._region_uuid = self._device.get("region_uuid")
        self._name = self._device.get("friendly_name")
        self._hostname = self._device.get("hostname")
        self._status = self._device.get("status")
        self._online = self._device.get("online")
        self._maintenance_mode = self._device.get("maintenance_mode")
        self._device_config_id = self._device.get("device_config_id")
        self._job_status = self._device.get("job_status")
        self._job_last_deployed = self._device.get("job_last_deployed")
        self._job_config_id = self._device.get("job_config_id")
        self._job_config_revision = self._device.get("job_config_revision")

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

    @property
    def hostname(self) -> str:
        return self._hostname

    @property
    def status(self) -> str:
        return self._status

    @property
    def online(self) -> bool:
        return self._online

    @property
    def maintenance_mode(self) -> bool:
        return self._maintenance_mode

    @property
    def device_config_id(self) -> str:
        return self._device_config_id

    @property
    def job_status(self) -> str:
        return self._job_status

    @property
    def job_last_deployed(self) -> str:
        return self._job_last_deployed

    @property
    def job_config_id(self) -> str:
        return self._job_config_id

    @property
    def job_config_revision(self) -> str:
        return self._job_config_revision

    def __str__(self):
        return json.dumps({k: v for k, v in self._device.items()})

    def __repr__(self):
        return f"Device( trainml , **{self._device.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/{self._id}",
            "DELETE",
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self

    async def toggle_maintenance(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/{self._id}/maintenance",
            "PATCH",
        )

    async def run_action(self, command):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/{self._id}/action",
            "POST",
            None,
            dict(command=command),
        )

    async def set_config(self, device_config_id):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/{self._id}",
            "PATCH",
            None,
            dict(device_config_id=device_config_id),
        )
        self.__init__(self.trainml, **resp)
        return self

    async def deploy_endpoint(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/{self._id}/deploy",
            "PUT",
        )

    async def stop_endpoint(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/device/{self._id}/stop",
            "PUT",
        )
