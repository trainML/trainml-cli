import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.models]


@mark.create
@mark.asyncio
class GetModelTests:
    @fixture(scope="class")
    async def model(self, trainml):
        model = await trainml.models.create(
            name="CLI Automated",
            source_type="git",
            source_uri="git@github.com:trainML/test-private.git",
        )
        model = await model.wait_for("ready", 60)
        yield model
        await model.remove()
        model = await model.wait_for("archived", 60)

    async def test_get_models(self, trainml, model):
        models = await trainml.models.list()
        assert len(models) > 0

    async def test_get_model(self, trainml, model):
        response = await trainml.models.get(model.id)
        assert response.id == model.id

    async def test_model_properties(self, model):
        assert isinstance(model.id, str)
        assert isinstance(model.status, str)
        assert isinstance(model.name, str)
        assert isinstance(model.provider, str)
        assert isinstance(model.size, int)

    def test_model_str(self, model):
        string = str(model)
        regex = r"^{.*\"model_uuid\": \"" + model.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_model_repr(self, model):
        string = repr(model)
        regex = (
            r"^Model\( trainml , \*\*{.*'model_uuid': '"
            + model.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)


@mark.create
@mark.asyncio
async def test_model_aws(trainml, capsys):
    model = await trainml.models.create(
        name="CLI Automated AWS",
        source_type="aws",
        source_uri="s3://trainml-examples/models/mxnet-model.zip",
    )
    model = await model.wait_for("ready", 60)
    status = model.status
    size = model.size
    await model.remove()
    assert status == "ready"
    assert size >= 1000000


@mark.create
@mark.asyncio
async def test_model_local(trainml, capsys):
    model = await trainml.models.create(
        name="CLI Automated Local",
        source_type="local",
        source_uri="~/tensorflow-example",
    )
    attach_task = asyncio.create_task(model.attach())
    connect_task = asyncio.create_task(model.connect())
    await asyncio.gather(attach_task, connect_task)
    await model.disconnect()
    await model.refresh()
    status = model.status
    size = model.size
    await model.remove()
    assert status == "ready"
    assert size >= 10000000
    captured = capsys.readouterr()
    sys.stdout.write(captured.out)
    sys.stderr.write(captured.err)
    assert "Starting data upload from local" in captured.out
    assert "data_batch_1.bin  30733788 bytes" in captured.out
    assert "Upload complete" in captured.out