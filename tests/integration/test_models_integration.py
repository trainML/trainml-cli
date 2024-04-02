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
            source_uri="git@github.com:trainML/environment-tests.git",
        )
        model = await model.wait_for("ready", 120)
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
        assert isinstance(model.size, int)

    async def test_model_str(self, model):
        string = str(model)
        regex = r"^{.*\"model_uuid\": \"" + model.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_model_repr(self, model):
        string = repr(model)
        regex = r"^Model\( trainml , \*\*{.*'model_uuid': '" + model.id + r"'.*}\)$"
        assert isinstance(string, str)
        assert re.match(regex, string)


@mark.create
@mark.asyncio
async def test_model_wasabi(trainml, capsys):
    model = await trainml.models.create(
        name="CLI Automated Wasabi",
        source_type="wasabi",
        source_uri="s3://trainml-example/models/trainml-examples",
        capacity="10G",
        source_options=dict(endpoint_url="https://s3.wasabisys.com"),
    )
    model = await model.wait_for("ready", 300)
    status = model.status
    size = model.size
    await model.remove()
    assert status == "ready"
    assert size >= 500000


@mark.create
@mark.asyncio
async def test_model_local(trainml, capsys):
    model = await trainml.models.create(
        name="CLI Automated Local",
        source_type="local",
        source_uri="~/tensorflow-model",
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
    assert size >= 1000000
    captured = capsys.readouterr()
    sys.stdout.write(captured.out)
    sys.stderr.write(captured.err)
    assert "Starting data upload from local" in captured.out
    assert "official/LICENSE  11456 bytes" in captured.out
    assert "Upload complete" in captured.out
