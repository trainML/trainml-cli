import json
import logging


class Services(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, provider_uuid, region_uuid, id, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/service/{id}",
            "GET",
            kwargs,
        )
        return Service(self.trainml, **resp)

    async def list(self, provider_uuid, region_uuid, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/service",
            "GET",
            kwargs,
        )
        services = [Service(self.trainml, **service) for service in resp]
        return services

    async def create(
        self,
        provider_uuid,
        region_uuid,
        name,
        public,
        **kwargs,
    ):
        logging.info(f"Creating Service {name}")
        data = dict(
            name=name,
            public=public,
            **kwargs,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/service",
            "POST",
            None,
            payload,
        )
        service = Service(self.trainml, **resp)
        logging.info(f"Created Service {name} with id {service.id}")
        return service

    async def remove(self, provider_uuid, region_uuid, id, **kwargs):
        await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/service/{id}",
            "DELETE",
            kwargs,
        )


class Service:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._service = kwargs
        self._id = self._service.get("service_id")
        self._provider_uuid = self._service.get("provider_uuid")
        self._region_uuid = self._service.get("region_uuid")
        self._public = self._service.get("public")
        self._name = self._service.get("name")
        self._hostname = self._service.get("hostname")

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
    def public(self) -> bool:
        return self._public

    @property
    def name(self) -> str:
        return self._name

    @property
    def hostname(self) -> str:
        return self._hostname

    def __str__(self):
        return json.dumps({k: v for k, v in self._service.items()})

    def __repr__(self):
        return f"Service( trainml , **{self._service.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/service/{self._id}",
            "DELETE",
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/service/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self
