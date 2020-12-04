import time
import sys


class Datasets(object):
    def __init__(self, trainml):
        self.trainml = trainml

    def get(self, id):
        resp = self.trainml._query(f"/dataset/pub/{id}", "GET")
        return Dataset(self.trainml, **resp)

    def create(self, name, source_type, source_uri, **kwargs):
        data = dict(
            name=name,
            source_type=source_type,
            source_uri=source_uri,
            source_options=kwargs.get("source_options"),
        )
        resp = self.trainml._query("/dataset/pub", "POST", None, data)
        dataset = Dataset(self.trainml, **resp)
        print(f"Created Dataset {dataset.id}, status {dataset.status}")
        if kwargs.get("wait"):
            sys.stdout.write("Downloading: ")
            sys.stdout.flush()
            while dataset.status != "ready":
                time.sleep(5)
                dataset = self.get(dataset.id)
                sys.stdout.write(".")
                sys.stdout.flush()
            print("\nDownload Complete")
        return dataset

    def remove(self, id):
        self.trainml._query(f"/dataset/pub/{id}", "DELETE")


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

    def destroy(self):
        self.trainml._query(f"/dataset/pub/{self._id}", "DELETE")