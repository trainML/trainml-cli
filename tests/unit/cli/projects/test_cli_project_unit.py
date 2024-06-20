import re
import json
import click
from unittest.mock import AsyncMock, patch, create_autospec
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.projects]

from trainml.cli import project as specimen


def test_list(runner, mock_projects):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.projects = AsyncMock()
        mock_trainml.projects.list = AsyncMock(return_value=mock_projects)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.projects.list.assert_called_once()
