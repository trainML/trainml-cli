import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.cloudbender, mark.providers]

from trainml.cli.cloudbender import provider as specimen
from trainml.cloudbender.providers import Provider


def test_list(runner, mock_providers):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.cloudbender = AsyncMock()
        mock_trainml.cloudbender.providers = AsyncMock()
        mock_trainml.cloudbender.providers.list = AsyncMock(
            return_value=mock_providers
        )
        result = runner.invoke(specimen, ["list"])
        assert result.exit_code == 0
        mock_trainml.cloudbender.providers.list.assert_called_once()
