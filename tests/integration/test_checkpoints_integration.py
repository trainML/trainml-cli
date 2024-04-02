import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.checkpoints]


@mark.create
@mark.asyncio
class GetCheckpointTests:
    @fixture(scope="class")
    async def checkpoint(self, trainml):
        checkpoint = await trainml.checkpoints.create(
            name="CLI Automated",
            source_type="git",
            source_uri="git@github.com:trainML/environment-tests.git",
        )
        checkpoint = await checkpoint.wait_for("ready", 120)
        yield checkpoint
        await checkpoint.remove()
        checkpoint = await checkpoint.wait_for("archived", 60)

    async def test_get_checkpoints(self, trainml, checkpoint):
        checkpoints = await trainml.checkpoints.list()
        assert len(checkpoints) > 0

    async def test_get_checkpoint(self, trainml, checkpoint):
        response = await trainml.checkpoints.get(checkpoint.id)
        assert response.id == checkpoint.id

    async def test_checkpoint_properties(self, checkpoint):
        assert isinstance(checkpoint.id, str)
        assert isinstance(checkpoint.status, str)
        assert isinstance(checkpoint.name, str)
        assert isinstance(checkpoint.size, int)

    async def test_checkpoint_str(self, checkpoint):
        string = str(checkpoint)
        regex = r"^{.*\"checkpoint_uuid\": \"" + checkpoint.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_checkpoint_repr(self, checkpoint):
        string = repr(checkpoint)
        regex = (
            r"^Checkpoint\( trainml , \*\*{.*'checkpoint_uuid': '"
            + checkpoint.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)


@mark.create
@mark.asyncio
async def test_checkpoint_wasabi(trainml, capsys):
    checkpoint = await trainml.checkpoints.create(
        name="CLI Automated Wasabi",
        source_type="wasabi",
        source_uri="s3://trainml-example/models/trainml-examples",
        capacity="10G",
        source_options=dict(endpoint_url="https://s3.wasabisys.com"),
    )
    checkpoint = await checkpoint.wait_for("ready", 300)
    status = checkpoint.status
    size = checkpoint.size
    await checkpoint.remove()
    assert status == "ready"
    assert size >= 500000


@mark.create
@mark.asyncio
async def test_checkpoint_local(trainml, capsys):
    checkpoint = await trainml.checkpoints.create(
        name="CLI Automated Local",
        source_type="local",
        source_uri="~/tensorflow-model",
    )
    attach_task = asyncio.create_task(checkpoint.attach())
    connect_task = asyncio.create_task(checkpoint.connect())
    await asyncio.gather(attach_task, connect_task)
    await checkpoint.disconnect()
    await checkpoint.refresh()
    status = checkpoint.status
    size = checkpoint.size
    await checkpoint.remove()
    assert status == "ready"
    assert size >= 1000000
    captured = capsys.readouterr()
    sys.stdout.write(captured.out)
    sys.stderr.write(captured.err)
    assert "Starting data upload from local" in captured.out
    assert "official/LICENSE  11456 bytes" in captured.out
    assert "Upload complete" in captured.out
