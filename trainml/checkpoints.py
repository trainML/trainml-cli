import json
import logging
import math
import asyncio
from datetime import datetime

from .exceptions import (
    CheckpointError,
    ApiError,
    SpecificationError,
    TrainMLException,
)
from .connections import Connection


class Checkpoints(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id, **kwargs):
        resp = await self.trainml._query(f"/checkpoint/{id}", "GET", kwargs)
        return Checkpoint(self.trainml, **resp)

    async def list(self, **kwargs):
        resp = await self.trainml._query(f"/checkpoint", "GET", kwargs)
        checkpoints = [Checkpoint(self.trainml, **checkpoint) for checkpoint in resp]
        return checkpoints

    async def list_public(self, **kwargs):
        resp = await self.trainml._query(f"/checkpoint/public", "GET", kwargs)
        datasets = [Checkpoint(self.trainml, **dataset) for dataset in resp]
        return datasets

    async def create(self, name, source_type, source_uri, **kwargs):
        data = dict(
            name=name,
            source_type=source_type,
            source_uri=source_uri,
            source_options=kwargs.get("source_options"),
            project_uuid=kwargs.get("project_uuid") or self.trainml.active_project,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        logging.info(f"Creating Checkpoint {name}")
        resp = await self.trainml._query("/checkpoint", "POST", None, payload)
        checkpoint = Checkpoint(self.trainml, **resp)
        logging.info(f"Created Checkpoint {name} with id {checkpoint.id}")

        return checkpoint

    async def remove(self, id, **kwargs):
        await self.trainml._query(
            f"/checkpoint/{id}", "DELETE", dict(**kwargs, force=True)
        )


class Checkpoint:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._checkpoint = kwargs
        self._id = self._checkpoint.get("id", self._checkpoint.get("checkpoint_uuid"))
        self._status = self._checkpoint.get("status")
        self._name = self._checkpoint.get("name")
        self._size = self._checkpoint.get("size")
        self._project_uuid = self._checkpoint.get("project_uuid")

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> str:
        return self._status

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    def __str__(self):
        return json.dumps({k: v for k, v in self._checkpoint.items()})

    def __repr__(self):
        return f"Checkpoint( trainml , **{self._checkpoint.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def get_log_url(self):
        resp = await self.trainml._query(
            f"/checkpoint/{self._id}/logs",
            "GET",
            dict(project_uuid=self._project_uuid),
        )
        return resp

    async def get_details(self):
        resp = await self.trainml._query(
            f"/checkpoint/{self._id}/details",
            "GET",
            dict(project_uuid=self._project_uuid),
        )
        return resp

    async def get_connection_utility_url(self):
        resp = await self.trainml._query(
            f"/checkpoint/{self._id}/download",
            "GET",
            dict(project_uuid=self._project_uuid),
        )
        return resp

    def get_connection_details(self):
        if self._checkpoint.get("vpn"):
            details = dict(
                entity_type="checkpoint",
                project_uuid=self._checkpoint.get("project_uuid"),
                cidr=self._checkpoint.get("vpn").get("cidr"),
                ssh_port=self._checkpoint.get("vpn").get("client").get("ssh_port"),
                input_path=(
                    self._checkpoint.get("source_uri")
                    if self.status in ["new", "downloading"]
                    else None
                ),
                output_path=(
                    self._checkpoint.get("output_uri")
                    if self.status == "exporting"
                    else None
                ),
            )
        else:
            details = dict()
        return details

    async def connect(self):
        if self.status in ["ready", "failed"]:
            raise SpecificationError(
                "status",
                f"You can only connect to downloading or exporting checkpoints.",
            )
        if self.status == "new":
            await self.wait_for("downloading")
        connection = Connection(
            self.trainml, entity_type="checkpoint", id=self.id, entity=self
        )
        await connection.start()
        return connection.status

    async def disconnect(self):
        connection = Connection(
            self.trainml, entity_type="checkpoint", id=self.id, entity=self
        )
        await connection.stop()
        return connection.status

    async def remove(self, force=False):
        await self.trainml._query(
            f"/checkpoint/{self._id}",
            "DELETE",
            dict(project_uuid=self._project_uuid, force=force),
        )

    async def rename(self, name):
        resp = await self.trainml._query(
            f"/checkpoint/{self._id}",
            "PATCH",
            dict(project_uuid=self._project_uuid),
            dict(name=name),
        )
        self.__init__(self.trainml, **resp)
        return self

    async def export(self, output_type, output_uri, output_options=dict()):
        resp = await self.trainml._query(
            f"/checkpoint/{self._id}/export",
            "POST",
            dict(project_uuid=self._project_uuid),
            dict(
                output_type=output_type,
                output_uri=output_uri,
                output_options=output_options,
            ),
        )
        self.__init__(self.trainml, **resp)
        return self

    def _get_msg_handler(self, msg_handler):
        def handler(data):
            if data.get("type") == "subscription":
                if msg_handler:
                    msg_handler(data)
                else:
                    timestamp = datetime.fromtimestamp(int(data.get("time")) / 1000)
                    print(
                        f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: {data.get('msg').rstrip()}"
                    )

        return handler

    async def attach(self, msg_handler=None):
        await self.refresh()
        if self.status not in ["ready", "failed"]:
            await self.trainml._ws_subscribe(
                "checkpoint",
                self._project_uuid,
                self.id,
                self._get_msg_handler(msg_handler),
            )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/checkpoint/{self.id}",
            "GET",
            dict(project_uuid=self._project_uuid),
        )
        self.__init__(self.trainml, **resp)
        return self

    async def wait_for(self, status, timeout=300):
        if self.status == status:
            return
        valid_statuses = ["downloading", "ready", "archived"]
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
            if self.status == status:
                return self
            elif self.status == "failed":
                raise CheckpointError(self.status, self)
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
