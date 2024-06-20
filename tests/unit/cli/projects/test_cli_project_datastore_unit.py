import re
import json
import click
from unittest.mock import AsyncMock, patch, create_autospec
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.projects]

from trainml.cli.project import datastore as specimen
from trainml.projects import (
    Project,
)


def test_list_datastores(runner, mock_project_datastores):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        project = create_autospec(Project)
        mock_trainml.projects = AsyncMock()
        mock_trainml.projects.get = AsyncMock(return_value=project)
        mock_trainml.projects.get_current = AsyncMock(return_value=project)
        project.datastores = AsyncMock()
        project.datastores.list = AsyncMock(return_value=mock_project_datastores)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        project.datastores.list.assert_called_once()
