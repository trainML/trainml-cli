import asyncio
from pytest import fixture

from trainml.trainml import TrainML


ENVS = {
    "dev": dict(
        region="us-east-2",
        client_id="6hiktq1842ko01jmtbafd0ki87",
        pool_id="us-east-2_OhcBqdjVS",
        api_url="api.trainml.dev",
        ws_url="api-ws.trainml.dev",
    ),
    "staging": dict(
        region="us-east-2",
        client_id="6hiktq1842ko01jmtbafd0ki87",
        pool_id="us-east-2_OhcBqdjVS",
        api_url="api.trainml.page",
        ws_url="api-ws.trainml.page",
    ),
    "prod": dict(
        region="us-east-2",
        client_id="32mc1obk9nq97iv015fnmc5eq5",
        pool_id="us-east-2_68kbvTL5p",
        api_url="api.trainml.ai",
        ws_url="api-ws.trainml.ai",
    ),
}


def pytest_addoption(parser):
    parser.addoption(
        "--env", action="store", default="dev", help="Environment to run tests against"
    )


@fixture(scope="session")
def env(request):
    env = request.config.getoption("--env")
    return ENVS[env]


@fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@fixture(scope="module")
def trainml(env):
    trainml = TrainML(**env)
    return trainml