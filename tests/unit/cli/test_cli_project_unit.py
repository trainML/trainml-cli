import re
import json
import click
from unittest.mock import AsyncMock, patch, create_autospec
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.projects]

from trainml.cli import project as specimen
from trainml.projects import Project


def test_list(runner, mock_projects):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.projects = AsyncMock()
        mock_trainml.projects.list = AsyncMock(return_value=mock_projects)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.projects.list.assert_called_once()


def test_list_datastores(runner, mock_project_datastores):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_project = create_autospec(Project)
        mock_project.list_datastores = AsyncMock(
            return_value=mock_project_datastores
        )
        mock_trainml.projects.get = AsyncMock(return_value=mock_project)
        result = runner.invoke(specimen, ["list-datastores"])
        print(result)
        assert result.exit_code == 0
        mock_project.list_datastores.assert_called_once()


def test_list_reservations(runner, mock_project_reservations):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_project = create_autospec(Project)
        mock_project.list_reservations = AsyncMock(
            return_value=mock_project_reservations
        )
        mock_trainml.projects.get = AsyncMock(return_value=mock_project)
        result = runner.invoke(specimen, ["list-reservations"])
        print(result)
        assert result.exit_code == 0
        mock_project.list_reservations.assert_called_once()
