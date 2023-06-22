import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.cloudbender, mark.nodes]

from trainml.cli.cloudbender import node as specimen
from trainml.cloudbender.nodes import Node


def test_list(runner, mock_nodes):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.cloudbender = AsyncMock()
        mock_trainml.cloudbender.nodes = AsyncMock()
        mock_trainml.cloudbender.nodes.list = AsyncMock(
            return_value=mock_nodes
        )
        result = runner.invoke(
            specimen,
            args=["list", "--provider=prov-id-1", "--region=reg-id-1"],
        )
        assert result.exit_code == 0
        mock_trainml.cloudbender.nodes.list.assert_called_once_with(
            provider_uuid="prov-id-1", region_uuid="reg-id-1"
        )


def test_list_no_provider(runner, mock_nodes):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.cloudbender = AsyncMock()
        mock_trainml.cloudbender.nodes = AsyncMock()
        mock_trainml.cloudbender.nodes.list = AsyncMock(
            return_value=mock_nodes
        )
        result = runner.invoke(specimen, ["list"])
        assert result.exit_code != 0
