import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.volumes]

from trainml.cli import volume as specimen
from trainml.volumes import Volume


def test_list(runner, mock_my_volumes):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.volumes = AsyncMock()
        mock_trainml.volumes.list = AsyncMock(return_value=mock_my_volumes)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.volumes.list.assert_called_once()
