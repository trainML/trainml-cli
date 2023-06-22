import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.models]

from trainml.cli import model as specimen
from trainml.models import Model


def test_list(runner, mock_models):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.models = AsyncMock()
        mock_trainml.models.list = AsyncMock(return_value=mock_models)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.models.list.assert_called_once()
