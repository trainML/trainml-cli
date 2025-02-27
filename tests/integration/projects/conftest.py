from pytest import fixture, mark


@fixture(scope="module")
@mark.xdist_group("project_resources")
async def project(trainml):
    project = await trainml.projects.create(
        name="New Project", copy_credentials=False, copy_secrets=False
    )
    yield project
    await project.remove()
