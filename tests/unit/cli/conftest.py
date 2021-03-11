from click.testing import CliRunner
from pytest import fixture, mark
from unittest.mock import Mock, AsyncMock, patch, create_autospec


pytestmark = [mark.cli,mark.unit]

@fixture
def runner():
    runner = CliRunner()
    yield runner