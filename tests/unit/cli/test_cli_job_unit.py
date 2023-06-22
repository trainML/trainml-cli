import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.jobs]

from trainml.cli import job as specimen
from trainml.jobs import Job


def test_list(runner, mock_jobs):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.jobs = AsyncMock()
        mock_trainml.jobs.list = AsyncMock(return_value=mock_jobs)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.jobs.list.assert_called_once()
