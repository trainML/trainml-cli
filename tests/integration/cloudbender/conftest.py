"""Shared CloudBender integration fixtures.

Pytest 9+ ignores marks placed on *fixture* functions (see PytestRemovedIn9Warning).
``xdist_group`` / ``create`` must therefore live on tests or be applied via
``pytest_collection_modifyitems`` below so ``--dist=loadgroup`` still serializes
tests that share session-scoped real resources (provider / region).
"""

from pytest import fixture, mark


pytestmark = [mark.integration, mark.cloudbender]


def pytest_collection_modifyitems(config, items):
    """Bind loadgroup + create markers to tests that use session CloudBender fixtures."""
    for item in items:
        names = getattr(item, "fixturenames", ())
        if "provider" not in names and "region" not in names:
            continue
        if item.get_closest_marker("xdist_group") is None:
            item.add_marker(mark.xdist_group("cloudbender_resources"))
        if item.get_closest_marker("create") is None:
            item.add_marker(mark.create)


@fixture(scope="session")
async def provider(trainml):
    provider = await trainml.cloudbender.providers.enable(type="test")
    await provider.wait_for("ready")
    yield provider
    await provider.remove()

@fixture(scope="session")
async def region(trainml, provider):
    region = await trainml.cloudbender.regions.create(provider_uuid=provider.id,name="test-region",
        public=False,
        storage=dict(mode="local"),)
    await region.wait_for("healthy")
    yield region
    await region.remove()
    await region.wait_for("archived")