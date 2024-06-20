import json
import logging
import asyncio
import math
from datetime import datetime

from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
    ProviderError,
)


class Providers(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id):
        resp = await self.trainml._query(f"/provider/{id}", "GET")
        return Provider(self.trainml, **resp)

    async def list(self):
        resp = await self.trainml._query(f"/provider", "GET")
        providers = [Provider(self.trainml, **provider) for provider in resp]
        return providers

    async def enable(self, type, **kwargs):
        data = dict(type=type, **kwargs)
        payload = {k: v for k, v in data.items() if v is not None}
        logging.info(f"Enabling Provider {type}")
        resp = await self.trainml._query("/provider", "POST", None, payload)
        provider = Provider(self.trainml, **resp)
        logging.info(f"Enabled Provider {type} with id {provider.id}")

        return provider

    async def remove(self, id):
        await self.trainml._query(f"/provider/{id}", "DELETE")


class Provider:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._provider = kwargs
        self._id = self._provider.get("provider_uuid")
        self._type = self._provider.get("type")
        self._status = self._provider.get("status")
        self._credits = self._provider.get("credits")

    @property
    def id(self) -> str:
        return self._id

    @property
    def type(self) -> str:
        return self._type

    @property
    def status(self) -> str:
        return self._status

    @property
    def credits(self) -> float:
        return self._credits

    def __str__(self):
        return json.dumps({k: v for k, v in self._provider.items()})

    def __repr__(self):
        return f"Provider( trainml , **{self._provider.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(f"/provider/{self._id}", "DELETE")

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self

    async def wait_for(self, status, timeout=300):
        if self.status == status:
            return
        valid_statuses = ["ready", "archived"]
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
                raise ProviderError(self.status, self)
            if self.status == status:
                return self
            else:
                count += 1
                logging.debug(f"self: {self}, retry count {count}")

        raise TrainMLException(f"Timeout waiting for {status}")
