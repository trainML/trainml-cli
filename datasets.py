class Datasets(object):
    def __init__(self, trainml):
        self.trainml = trainml


class Dataset:
    def __init__(self, trainml, **kwargs):
        self.trainml = trainml
        self._dataset = kwargs
        self._id = self._dataset.get("id", self._dataset.get("dataset_uuid"))

    @property
    def id(self) -> str:
        return self._id