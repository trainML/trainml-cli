import re
import json
import click
from unittest.mock import AsyncMock, patch, create_autospec
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.projects]

from trainml.cli.project import authorizer as specimen
from trainml.projects import (
    Project,
)


def test_list_authorizers(runner, mock_project_authorizers):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        project = create_autospec(Project)
        mock_trainml.projects = AsyncMock()
        mock_trainml.projects.get = AsyncMock(return_value=project)
        mock_trainml.projects.get_current = AsyncMock(return_value=project)
        project.authorizers = AsyncMock()
        project.authorizers.list = AsyncMock(return_value=mock_project_authorizers)
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        project.authorizers.list.assert_called_once()


def test_add_authorizer_api_key(runner):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        project = create_autospec(Project)
        mock_trainml.projects = AsyncMock()
        mock_trainml.projects.get = AsyncMock(return_value=project)
        mock_trainml.projects.get_current = AsyncMock(return_value=project)
        project.authorizers = AsyncMock()
        project.authorizers.create = AsyncMock()
        result = runner.invoke(
            specimen,
            [
                "add",
                "api_key_authorizer",
                "--type",
                "api_key",
                "--client-id",
                "key-1",
            ],
            input="secret-key\n",
        )
        print(result)
        assert result.exit_code == 0
        project.authorizers.create.assert_called_once()
