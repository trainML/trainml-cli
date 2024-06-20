import json
import logging
import asyncio
import math

from trainml.exceptions import ApiError, SpecificationError, TrainMLException, NodeError


class Nodes(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, provider_uuid, region_uuid, id, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/node/{id}",
            "GET",
            kwargs,
        )
        return Node(self.trainml, **resp)

    async def list(self, provider_uuid, region_uuid, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/node",
            "GET",
            kwargs,
        )
        nodes = [Node(self.trainml, **node) for node in resp]
        return nodes

    async def create(
        self,
        provider_uuid,
        region_uuid,
        friendly_name,
        hostname,
        minion_id=None,
        type="permanent",
        service="compute",
        **kwargs,
    ):
        logging.info(f"Creating Node {friendly_name}")
        data = dict(
            friendly_name=friendly_name,
            hostname=hostname,
            minion_id=minion_id,
            type=type,
            service=service,
            **kwargs,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/node",
            "POST",
            None,
            payload,
        )
        node = Node(self.trainml, **resp)
        logging.info(f"Created Node {friendly_name} with id {node.id}")
        return node

    async def remove(self, provider_uuid, region_uuid, id, **kwargs):
        await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/node/{id}",
            "DELETE",
            kwargs,
        )


class Node:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._node = kwargs
        self._id = self._node.get("rig_uuid")
        self._provider_uuid = self._node.get("provider_uuid")
        self._region_uuid = self._node.get("region_uuid")
        self._type = self._node.get("type")
        self._service = self._node.get("service")
        self._name = self._node.get("friendly_name")
        self._hostname = self._node.get("hostname")
        self._status = self._node.get("status")
        self._online = self._node.get("online")
        self._maintenance_mode = self._node.get("maintenance_mode")

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
    def type(self) -> str:
        return self._type

    @property
    def service(self) -> str:
        return self._service

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

    def __str__(self):
        return json.dumps({k: v for k, v in self._node.items()})

    def __repr__(self):
        return f"Node( trainml , **{self._node.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/node/{self._id}",
            "DELETE",
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/node/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self

    async def toggle_maintenance(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/node/{self._id}/maintenance",
            "PATCH",
        )

    async def run_action(self, command):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/node/{self._id}/action",
            "POST",
            None,
            dict(command=command),
        )

    async def wait_for(self, status, timeout=300):
        if self.status == status:
            return
        valid_statuses = ["active", "maintenance", "offline", "stopped", "archived"]
        if not status in valid_statuses:
            raise SpecificationError(
                "status",
                f"Invalid wait_for status {status}.  Valid statuses are: {valid_statuses}",
            )
        MAX_TIMEOUT = 24 * 60 * 60
        if timeout > MAX_TIMEOUT:
            raise SpecificationError(
                "timeout",
                f"timeout must be less than {MAX_TIMEOUT} seconds.",
            )

        POLL_INTERVAL_MIN = 5
        POLL_INTERVAL_MAX = 60
        POLL_INTERVAL = max(min(timeout / 60, POLL_INTERVAL_MAX), POLL_INTERVAL_MIN)
        retry_count = math.ceil(timeout / POLL_INTERVAL)
        count = 0
        while count < retry_count:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                await self.refresh()
            except ApiError as e:
                if status == "archived" and e.status == 404:
                    return
                raise e
            if self.status in ["errored", "failed"]:
                raise NodeError(self.status, self)
            if self.status == status:
                return self
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
