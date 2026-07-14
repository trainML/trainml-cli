"""Shared project integration fixtures.

Pytest 9+ ignores marks on fixture callables; see cloudbender ``conftest`` docstring.
"""

from pytest import fixture, mark


def pytest_collection_modifyitems(config, items):
    """Bind loadgroup + create markers to tests that use the shared ``project`` fixture."""
    for item in items:
        if "project" not in getattr(item, "fixturenames", ()):
            continue
        if item.get_closest_marker("xdist_group") is None:
            item.add_marker(mark.xdist_group("project_resources"))
        if item.get_closest_marker("create") is None:
            item.add_marker(mark.create)


@fixture(scope="module")
async def project(trainml):
    project = await trainml.projects.create(
        name="New Project", copy_credentials=False, copy_secrets=False
    )
    yield project
    await project.remove()
