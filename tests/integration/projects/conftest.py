from pytest import fixture


@fixture(scope="module")
async def project(trainml):
    project = await trainml.projects.create(name="New Project", copy_keys=False)
    yield project
    await project.remove()
