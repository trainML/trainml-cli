import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.gpu_types]

from trainml.cli import gpu as specimen
from trainml.gpu_types import GpuType


def test_list(runner, mock_gpu_types):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.gpu_types = AsyncMock()
        mock_trainml.gpu_types.list = AsyncMock(return_value=mock_gpu_types)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.gpu_types.list.assert_called_once()
