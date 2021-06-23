import asyncio
from pytest import fixture, mark
from unittest.mock import Mock, patch, create_autospec

from trainml.trainml import TrainML

pytestmark = mark.integration

ENVS = {
    "dev": dict(
        region="us-east-2",
        client_id="6hiktq1842ko01jmtbafd0ki87",
        pool_id="us-east-2_OhcBqdjVS",
        domain_suffix="trainml.dev",
        api_url="api.trainml.dev",
        ws_url="api-ws.trainml.dev",
    ),
    "staging": dict(
        region="us-east-2",
        client_id="6hiktq1842ko01jmtbafd0ki87",
        pool_id="us-east-2_OhcBqdjVS",
        domain_suffix="trainml.page",
        api_url="api.trainml.page",
        ws_url="api-ws.trainml.page",
    ),
    "prod": dict(
        region="us-east-2",
        client_id="32mc1obk9nq97iv015fnmc5eq5",
        pool_id="us-east-2_68kbvTL5p",
        domain_suffix="trainml.ai",
        api_url="api.trainml.ai",
        ws_url="api-ws.trainml.ai",
    ),
}


@fixture(scope="session")
def env(request):
    env = request.config.getoption("--env")
    yield ENVS[env]


@fixture(scope="module")
def trainml(env):
    trainml = TrainML(**env)
    yield trainml
