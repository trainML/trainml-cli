import re
import json
import click
from unittest.mock import AsyncMock, patch, create_autospec
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.projects]

from trainml.cli.project import service as specimen
from trainml.projects import (
    Project,
)


def test_list_services(runner, mock_project_services):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        project = create_autospec(Project)
        mock_trainml.projects = AsyncMock()
        mock_trainml.projects.get = AsyncMock(return_value=project)
        mock_trainml.projects.get_current = AsyncMock(return_value=project)
        project.services = AsyncMock()
        project.services.list = AsyncMock(return_value=mock_project_services)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        project.services.list.assert_called_once()
