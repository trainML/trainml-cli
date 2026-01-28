import re
import json
import click
from unittest.mock import AsyncMock, patch, Mock
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.checkpoints]

from trainml.cli import checkpoint as specimen
from trainml.cli.checkpoint import pretty_size
from trainml.checkpoints import Checkpoint


def test_pretty_size_zero():
    """Test pretty_size with zero/None (line 7)."""
    result = pretty_size(None)
    assert result == "0.00   B"
    result = pretty_size(0)
    assert result == "0.00   B"


def test_list(runner, mock_my_checkpoints):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = AsyncMock(
            return_value=mock_my_checkpoints
        )
        result = runner.invoke(specimen, ["list"])
        print(result)
        assert result.exit_code == 0
        mock_trainml.checkpoints.list.assert_called_once()


def test_attach_success(runner, mock_my_checkpoints):
    """Test attach command success (lines 32-38)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def list_async():
            return mock_my_checkpoints

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = Mock(side_effect=lambda: list_async())

        # Use the first checkpoint from the list
        checkpoint = mock_my_checkpoints[0]

        async def attach_async():
            return None

        checkpoint.attach = Mock(return_value=attach_async())

        with patch("trainml.cli.search_by_id_name", return_value=checkpoint):
            result = runner.invoke(specimen, ["attach", "1"])
            assert result.exit_code == 0
            checkpoint.attach.assert_called_once()


def test_attach_not_found(runner, mock_my_checkpoints):
    """Test attach command when checkpoint not found (line 36)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def list_async():
            return mock_my_checkpoints

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = Mock(side_effect=lambda: list_async())

        with patch("trainml.cli.search_by_id_name", return_value=None):
            result = runner.invoke(specimen, ["attach", "nonexistent"])
            assert result.exit_code != 0
            assert "Cannot find specified checkpoint" in result.output


def test_connect_with_attach(runner, mock_my_checkpoints):
    """Test connect command with attach (lines 56-65, attach=True)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def list_async():
            return mock_my_checkpoints

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = Mock(side_effect=lambda: list_async())

        checkpoint = mock_my_checkpoints[0]

        async def connect_async():
            return None

        async def attach_async():
            return None

        checkpoint.connect = Mock(return_value=connect_async())
        checkpoint.attach = Mock(return_value=attach_async())

        with patch("trainml.cli.search_by_id_name", return_value=checkpoint):
            result = runner.invoke(specimen, ["connect", "1"])
            assert result.exit_code == 0
            checkpoint.connect.assert_called_once()
            checkpoint.attach.assert_called_once()


def test_connect_no_attach(runner, mock_my_checkpoints):
    """Test connect command without attach (lines 56-65, attach=False)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def list_async():
            return mock_my_checkpoints

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = Mock(side_effect=lambda: list_async())

        checkpoint = mock_my_checkpoints[0]

        async def connect_async():
            return None

        checkpoint.connect = Mock(return_value=connect_async())

        with patch("trainml.cli.search_by_id_name", return_value=checkpoint):
            result = runner.invoke(specimen, ["connect", "--no-attach", "1"])
            assert result.exit_code == 0
            checkpoint.connect.assert_called_once()


def test_connect_not_found(runner, mock_my_checkpoints):
    """Test connect command when checkpoint not found (line 60)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def list_async():
            return mock_my_checkpoints

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = Mock(side_effect=lambda: list_async())

        with patch("trainml.cli.search_by_id_name", return_value=None):
            result = runner.invoke(specimen, ["connect", "nonexistent"])
            assert result.exit_code != 0
            assert "Cannot find specified checkpoint" in result.output


def test_create_with_connect_and_attach(runner, tmp_path, mock_my_checkpoints):
    """Test create command with connect and attach (lines 103-115)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        checkpoint = mock_my_checkpoints[0]

        async def connect_async():
            return None

        async def attach_async():
            return None

        checkpoint.connect = Mock(return_value=connect_async())
        checkpoint.attach = Mock(return_value=attach_async())

        async def create_async(**kwargs):
            return checkpoint

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.create = Mock(
            side_effect=lambda **kwargs: create_async(**kwargs)
        )

        test_dir = tmp_path / "test_checkpoint"
        test_dir.mkdir()
        result = runner.invoke(
            specimen, ["create", "test-checkpoint", str(test_dir)]
        )
        assert result.exit_code == 0
        mock_trainml.checkpoints.create.assert_called_once()
        checkpoint.connect.assert_called_once()
        checkpoint.attach.assert_called_once()


def test_create_with_connect_no_attach(runner, tmp_path, mock_my_checkpoints):
    """Test create command with connect but no attach (lines 103-115)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        checkpoint = mock_my_checkpoints[0]

        async def connect_async():
            return None

        checkpoint.connect = Mock(return_value=connect_async())

        async def create_async(**kwargs):
            return checkpoint

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.create = Mock(
            side_effect=lambda **kwargs: create_async(**kwargs)
        )

        test_dir = tmp_path / "test_checkpoint"
        test_dir.mkdir()
        result = runner.invoke(
            specimen,
            ["create", "--no-attach", "test-checkpoint", str(test_dir)],
        )
        assert result.exit_code == 0
        checkpoint.connect.assert_called_once()


def test_create_no_connect(runner, tmp_path):
    """Test create command without connect (lines 103-115, line 115)."""
    mock_checkpoint = Mock(spec=Checkpoint)

    mock_trainml_runner = Mock()
    mock_trainml_runner.client = Mock()
    mock_trainml_runner.client.checkpoints = Mock()
    mock_trainml_runner.client.checkpoints.create = AsyncMock(
        return_value=mock_checkpoint
    )
    mock_trainml_runner.run = Mock(
        side_effect=lambda x: x if not hasattr(x, "__call__") else x()
    )

    with patch("trainml.cli.TrainMLRunner", return_value=mock_trainml_runner):
        test_dir = tmp_path / "test_checkpoint"
        test_dir.mkdir()
        result = runner.invoke(
            specimen,
            ["create", "--no-connect", "test-checkpoint", str(test_dir)],
        )
        assert result.exit_code != 0
        assert "No logs to show" in result.output


def test_list_public(runner, mock_my_checkpoints):
    """Test list_public command (lines 152-171)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list_public = AsyncMock(
            return_value=mock_my_checkpoints
        )

        result = runner.invoke(specimen, ["list-public"])
        assert result.exit_code == 0
        mock_trainml.checkpoints.list_public.assert_called_once()


def test_remove_success(runner, mock_my_checkpoints):
    """Test remove command success (lines 192-201)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def list_async():
            return mock_my_checkpoints

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = Mock(side_effect=lambda: list_async())

        checkpoint = mock_my_checkpoints[0]

        async def remove_async():
            return None

        checkpoint.remove = Mock(return_value=remove_async())

        with patch("trainml.cli.search_by_id_name", return_value=checkpoint):
            result = runner.invoke(specimen, ["remove", "1"])
            assert result.exit_code == 0
            checkpoint.remove.assert_called_once_with(force=False)


def test_remove_not_found(runner, mock_my_checkpoints):
    """Test remove command when checkpoint not found (lines 192-201)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def list_async():
            return mock_my_checkpoints

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.list = Mock(side_effect=lambda: list_async())

        with patch("trainml.cli.search_by_id_name", return_value=None):
            result = runner.invoke(specimen, ["remove", "nonexistent"])
            assert result.exit_code != 0
            assert "Cannot find specified checkpoint" in result.output


def test_rename_success(runner, mock_my_checkpoints):
    """Test rename command success (lines 214-223)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        checkpoint = mock_my_checkpoints[0]

        async def rename_async():
            return None

        checkpoint.rename = Mock(return_value=rename_async())

        async def get_async(checkpoint_id):
            return checkpoint

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.get = Mock(
            side_effect=lambda checkpoint_id: get_async(checkpoint_id)
        )

        result = runner.invoke(specimen, ["rename", "1", "new-name"])
        assert result.exit_code == 0
        checkpoint.rename.assert_called_once_with(name="new-name")


def test_rename_not_found_none(runner):
    """Test rename command when checkpoint is None (lines 214-223, line 219)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def get_async(checkpoint_id):
            return None

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.get = Mock(
            side_effect=lambda checkpoint_id: get_async(checkpoint_id)
        )

        result = runner.invoke(specimen, ["rename", "nonexistent", "new-name"])
        assert result.exit_code != 0
        assert "Cannot find specified checkpoint" in result.output


def test_rename_not_found_exception(runner):
    """Test rename command when exception occurs (lines 214-223, line 221)."""
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:

        async def get_async(checkpoint_id):
            raise Exception("Not found")

        mock_trainml.checkpoints = AsyncMock()
        mock_trainml.checkpoints.get = Mock(
            side_effect=lambda checkpoint_id: get_async(checkpoint_id)
        )

        result = runner.invoke(specimen, ["rename", "nonexistent", "new-name"])
        assert result.exit_code != 0
        assert "Cannot find specified checkpoint" in result.output
