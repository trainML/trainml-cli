import json
from datetime import datetime


class GpuTypes(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self):
        resp = await self.trainml._query(f"/gpu/pub/types", "GET")
        gpu_types = [GpuType(self.trainml, **gpu_type) for gpu_type in resp]
        return gpu_types


class GpuType:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._gpu_type = kwargs
        self._id = self._gpu_type.get("id", self._gpu_type.get("gpu_type_id"))
        self._name = self._gpu_type.get("name")
        self._available = self._gpu_type.get("available")
        self._credits_per_hour = self._gpu_type.get("credits_per_hour")

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> int:
        return self._available

    @property
    def credits_per_hour(self) -> float:
        return self._credits_per_hour

    def __repr__(self):
        return json.dumps(
            {
                k: v
                for k, v in self._gpu_type.items()
                if k not in ["createdAt", "updatedAt"]
            }
        )

    async def refresh(self):
        resp = await self.trainml._query(f"/gpu/pub/types/{self.id}", "GET")
        self.__init__(self.trainml, **resp)
        return self