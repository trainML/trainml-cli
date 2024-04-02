import re
import sys
import asyncio
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.volumes]


@mark.create
@mark.asyncio
class GetVolumeTests:
    @fixture(scope="class")
    async def volume(self, trainml):
        volume = await trainml.volumes.create(
            name="CLI Automated",
            source_type="git",
            source_uri="git@github.com:trainML/environment-tests.git",
            capacity="10G",
        )
        volume = await volume.wait_for("ready", 120)
        yield volume
        await volume.remove()
        volume = await volume.wait_for("archived", 60)

    async def test_get_volumes(self, trainml, volume):
        volumes = await trainml.volumes.list()
        assert len(volumes) > 0

    async def test_get_volume(self, trainml, volume):
        response = await trainml.volumes.get(volume.id)
        assert response.id == volume.id

    async def test_volume_properties(self, volume):
        assert isinstance(volume.id, str)
        assert isinstance(volume.status, str)
        assert isinstance(volume.name, str)
        assert isinstance(volume.capacity, str)
        assert isinstance(volume.used_size, int)
        assert isinstance(volume.billed_size, int)

    async def test_volume_str(self, volume):
        string = str(volume)
        regex = r"^{.*\"id\": \"" + volume.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_volume_repr(self, volume):
        string = repr(volume)
        regex = r"^Volume\( trainml , \*\*{.*'id': '" + volume.id + r"'.*}\)$"
        assert isinstance(string, str)
        assert re.match(regex, string)


@mark.create
@mark.asyncio
async def test_volume_wasabi(trainml, capsys):
    volume = await trainml.volumes.create(
        name="CLI Automated Wasabi",
        source_type="wasabi",
        source_uri="s3://trainml-example/models/trainml-examples",
        capacity="10G",
        source_options=dict(endpoint_url="https://s3.wasabisys.com"),
    )
    volume = await volume.wait_for("ready", 300)
    status = volume.status
    billed_size = volume.billed_size
    used_size = volume.used_size
    await volume.remove()
    assert status == "ready"
    assert billed_size >= 10000000
    assert used_size >= 500000


@mark.create
@mark.asyncio
async def test_volume_local(trainml, capsys):
    volume = await trainml.volumes.create(
        name="CLI Automated Local",
        source_type="local",
        source_uri="~/tensorflow-model",
        capacity="10G",
    )
    attach_task = asyncio.create_task(volume.attach())
    connect_task = asyncio.create_task(volume.connect())
    await asyncio.gather(attach_task, connect_task)
    await volume.disconnect()
    await volume.refresh()
    status = volume.status
    billed_size = volume.billed_size
    used_size = volume.used_size
    await volume.remove()
    assert status == "ready"
    assert billed_size >= 10000000
    assert used_size >= 1000000
    captured = capsys.readouterr()
    sys.stdout.write(captured.out)
    sys.stderr.write(captured.err)
    assert "Starting data upload from local" in captured.out
    assert "official/LICENSE  11456 bytes" in captured.out
    assert "Upload complete" in captured.out
