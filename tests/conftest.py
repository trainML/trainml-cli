import asyncio
from pytest import fixture
from unittest.mock import Mock

from trainml.trainml import TrainML
from trainml.datasets import Dataset


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
        "--env",
        action="store",
        default="dev",
        help="Environment to run tests against",
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


@fixture(scope="session")
def my_datasets():
    trainml = Mock()
    return [
        Dataset(
            trainml,
            dataset_uuid="1",
            name="first one",
            status="ready",
            provider="trainml",
        ),
        Dataset(
            trainml,
            dataset_uuid="2",
            name="second one",
            status="ready",
            provider="trainml",
        ),
        Dataset(
            trainml,
            dataset_uuid="3",
            name="first one",
            status="ready",
            provider="gcp",
        ),
        Dataset(
            trainml,
            dataset_uuid="4",
            name="other one",
            status="ready",
            provider="gcp",
        ),
        Dataset(
            trainml,
            dataset_uuid="5",
            name="not ready",
            status="new",
            provider="trainml",
        ),
        Dataset(
            trainml,
            dataset_uuid="6",
            name="failed",
            status="failed",
            provider="trainml",
        ),
    ]


@fixture(scope="session")
def public_datasets():
    trainml = Mock()
    return [
        Dataset(
            trainml,
            dataset_uuid="11",
            name="first one",
            status="ready",
            provider="trainml",
        ),
        Dataset(
            trainml,
            dataset_uuid="12",
            name="second one",
            status="ready",
            provider="trainml",
        ),
        Dataset(
            trainml,
            dataset_uuid="13",
            name="first one",
            status="ready",
            provider="gcp",
        ),
        Dataset(
            trainml,
            dataset_uuid="14",
            name="other one",
            status="ready",
            provider="gcp",
        ),
        Dataset(
            trainml,
            dataset_uuid="15",
            name="not ready",
            status="new",
            provider="trainml",
        ),
        Dataset(
            trainml,
            dataset_uuid="16",
            name="failed",
            status="failed",
            provider="trainml",
        ),
    ]