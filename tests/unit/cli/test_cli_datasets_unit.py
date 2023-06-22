import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.datasets]

from trainml.cli import dataset as specimen
from trainml.datasets import Dataset


def test_list(runner, mock_my_datasets):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.datasets = AsyncMock()
        mock_trainml.datasets.list = AsyncMock(return_value=mock_my_datasets)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.datasets.list.assert_called_once()
