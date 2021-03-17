import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.environments]

from trainml.cli import environment as specimen
from trainml.environments import Environment


def test_list(runner, mock_environments):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.environments = AsyncMock()
        mock_trainml.environments.list = AsyncMock(
            return_value=mock_environments
        )
        result = runner.invoke(specimen, ["list"])
        assert result.exit_code == 0
        mock_trainml.environments.list.assert_called_once()
