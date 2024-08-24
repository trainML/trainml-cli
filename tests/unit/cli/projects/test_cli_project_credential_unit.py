import re
import json
import click
from unittest.mock import AsyncMock, patch, create_autospec
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.projects]

from trainml.cli.project import credential as specimen
from trainml.projects import (
    Project,
)


def test_list_credentials(runner, mock_project_credentials):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        project = create_autospec(Project)
        mock_trainml.projects = AsyncMock()
        mock_trainml.projects.get = AsyncMock(return_value=project)
        mock_trainml.projects.get_current = AsyncMock(return_value=project)
        project.credentials = AsyncMock()
        project.credentials.list = AsyncMock(return_value=mock_project_credentials)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        project.credentials.list.assert_called_once()
