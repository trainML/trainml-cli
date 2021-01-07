import json
from datetime import datetime


class Datasets(object):
    def __init__(self, trainml):
        self.trainml = trainml

    async def get(self, id):
        resp = await self.trainml._query(f"/dataset/pub/{id}", "GET")
        return Dataset(self.trainml, **resp)

    async def create(self, name, source_type, source_uri, **kwargs):
        data = dict(
            name=name,
            source_type=source_type,
            source_uri=source_uri,
            source_options=kwargs.get("source_options"),
        )
        print(f"Creating Dataset {name}")
        resp = await self.trainml._query("/dataset/pub", "POST", None, data)
        dataset = Dataset(self.trainml, **resp)
        print(f"Created Dataset {name} with id {dataset.id}")

        if kwargs.get("wait"):
            await dataset.attach()
            dataset = await self.get(dataset.id)
        return dataset

    async def remove(self, id):
        await self.trainml._query(f"/dataset/pub/{id}", "DELETE")


class Dataset:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._dataset = kwargs
        self._id = self._dataset.get("id", self._dataset.get("dataset_uuid"))
        self._status = self._dataset.get("status")

    @property
    def id(self) -> str:
        return self._id

    @property
    def status(self) -> str:
        return self._status

    def __repr__(self):
        return json.dumps({k: v for k, v in self._dataset.items()})

    async def get_connection_utility_url(self):
        resp = await self.trainml._query(f"/dataset/pub/{self._id}/download", "GET")
        return resp

    async def destroy(self):
        await self.trainml._query(f"/dataset/pub/{self._id}", "DELETE")

    async def attach(self):
        def msg_handler(msg):
            data = json.loads(msg.data)
            if data.get("type") == "subscription":
                timestamp = datetime.fromtimestamp(int(data.get("time")) / 1000)
                print(
                    f"{timestamp.strftime('%m/%d/%Y, %H:%M:%S')}: {data.get('msg').rstrip()}"
                )

        await self.trainml._ws_subscribe("dataset", self.id, msg_handler)