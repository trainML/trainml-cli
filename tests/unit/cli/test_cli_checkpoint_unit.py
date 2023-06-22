import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.checkpoints]

from trainml.cli import checkpoint as specimen
from trainml.checkpoints import Checkpoint


def test_list(runner, mock_my_checkpoints):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = AsyncMock(
            return_value=mock_my_checkpoints
        )
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.checkpoints.list.assert_called_once()
