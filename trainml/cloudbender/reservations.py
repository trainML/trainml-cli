import json
import logging


class Reservations(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, provider_uuid, region_uuid, id, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/reservation/{id}",
            "GET",
            kwargs,
        )
        return Reservation(self.trainml, **resp)

    async def list(self, provider_uuid, region_uuid, **kwargs):
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/reservation",
            "GET",
            kwargs,
        )
        reservations = [
            Reservation(self.trainml, **reservation) for reservation in resp
        ]
        return reservations

    async def create(
        self,
        provider_uuid,
        region_uuid,
        name,
        type,
        resource,
        hostname,
        **kwargs,
    ):
        logging.info(f"Creating Reservation {name}")
        data = dict(
            name=name,
            type=type,
            resource=resource,
            hostname=hostname,
            **kwargs,
        )
        payload = {k: v for k, v in data.items() if v is not None}
        resp = await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/reservation",
            "POST",
            None,
            payload,
        )
        reservation = Reservation(self.trainml, **resp)
        logging.info(f"Created Reservation {name} with id {reservation.id}")
        return reservation

    async def remove(self, provider_uuid, region_uuid, id, **kwargs):
        await self.trainml._query(
            f"/provider/{provider_uuid}/region/{region_uuid}/reservation/{id}",
            "DELETE",
            kwargs,
        )


class Reservation:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._reservation = kwargs
        self._id = self._reservation.get("reservation_id")
        self._provider_uuid = self._reservation.get("provider_uuid")
        self._region_uuid = self._reservation.get("region_uuid")
        self._type = self._reservation.get("type")
        self._name = self._reservation.get("name")
        self._resource = self._reservation.get("resource")
        self._hostname = self._reservation.get("hostname")

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
    def name(self) -> str:
        return self._name

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def hostname(self) -> str:
        return self._hostname

    def __str__(self):
        return json.dumps({k: v for k, v in self._reservation.items()})

    def __repr__(self):
        return f"Reservation( trainml , **{self._reservation.__repr__()})"

    def __bool__(self):
        return bool(self._id)

    async def remove(self):
        await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/reservation/{self._id}",
            "DELETE",
        )

    async def refresh(self):
        resp = await self.trainml._query(
            f"/provider/{self._provider_uuid}/region/{self._region_uuid}/reservation/{self._id}",
            "GET",
        )
        self.__init__(self.trainml, **resp)
        return self
