from pytest import mark, fixture
import re

pytestmark = mark.datasets


@fixture(scope="module")
async def dataset(trainml):
    dataset = await trainml.datasets.create(
        name="CLI Automated AWS",
        source_type="aws",
        source_uri="s3://trainml-examples/data/cifar10",
        wait=True,
    )
    yield dataset
    await dataset.remove()


@fixture(scope="class")
async def kaggle_dataset(trainml):
    dataset = await trainml.datasets.create(
        name="CLI Automated Kaggle",
        source_type="kaggle",
        source_uri="lish-moa",
        source_options=dict(type="competition"),
    )
    yield dataset


@mark.asyncio
async def test_get_public_datasets(trainml):
    datasets = await trainml.datasets.list_public()
    assert len(datasets) > 0


@mark.create
@mark.asyncio
class GetDatasetTests:
    async def test_get_my_datasets(self, trainml, dataset):
        datasets = await trainml.datasets.list()
        assert len(datasets) > 0

    async def test_get_dataset(self, trainml, dataset):
        response = await trainml.datasets.get(dataset.id)
        assert response.id == dataset.id

    async def test_dataset_properties(self, dataset):
        assert isinstance(dataset.id, str)
        assert isinstance(dataset.status, str)
        assert isinstance(dataset.name, str)

    def test_dataset_str(self, dataset):
        string = str(dataset)
        regex = r"^{.*\"dataset_uuid\": \"" + dataset.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_dataset_repr(self, dataset):
        string = repr(dataset)
        regex = (
            r"^Dataset\( trainml , {.*'dataset_uuid': '"
            + dataset.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)


@mark.create
@mark.asyncio
class DatasetLifeCycleTests:
    async def test_wait_for_ready(self, kaggle_dataset):
        assert kaggle_dataset.status != "ready"
        dataset = await kaggle_dataset.waitFor("ready", 60)
        assert dataset.status == "ready"

    async def test_remove_dataset(self, kaggle_dataset):
        assert kaggle_dataset.status == "ready"
        await kaggle_dataset.remove()
        dataset = await kaggle_dataset.waitFor("archived", 60)
        assert dataset is None
