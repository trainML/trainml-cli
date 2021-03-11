import re
import json
import click
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises

pytestmark = [mark.cli, mark.unit, mark.environments]

from trainml.cli import environment as specimen
from trainml.environments import Environment


def test_list(runner):
    with patch("trainml.cli.TrainML", new=AsyncMock) as mock_trainml:
        mock_trainml.environments = AsyncMock()
        mock_trainml.environments.list = AsyncMock(
            return_value=[
                Environment(
                    mock_trainml,
                    **{
                        "id": "DEEPLEARNING_PY37",
                        "framework": "Deep Learning",
                        "py_version": "3.7",
                        "cuda_version": "10.2",
                        "name": "Deep Learning - Python 3.7",
                    },
                ),
                Environment(
                    mock_trainml,
                    **{
                        "id": "DEEPLEARNING_PY38",
                        "framework": "Deep Learning",
                        "py_version": "3.8",
                        "cuda_version": "11.1",
                        "name": "Deep Learning - Python 3.8",
                    },
                ),
                Environment(
                    mock_trainml,
                    **{
                        "id": "PYTORCH_PY38_17",
                        "framework": "PyTorch",
                        "py_version": "3.8",
                        "version": "1.7",
                        "cuda_version": "11.1",
                        "name": "PyTorch 1.7 - Python 3.8",
                    },
                ),
                Environment(
                    mock_trainml,
                    **{
                        "id": "PYTORCH_PY37_17",
                        "framework": "PyTorch",
                        "py_version": "3.7",
                        "version": "1.7",
                        "cuda_version": "10.2",
                        "name": "PyTorch 1.7 - Python 3.7",
                    },
                ),
                Environment(
                    mock_trainml,
                    **{
                        "id": "PYTORCH_PY37_16",
                        "framework": "PyTorch",
                        "py_version": "3.7",
                        "version": "1.6",
                        "cuda_version": "10.2",
                        "name": "PyTorch 1.6 - Python 3.7",
                    },
                ),
                Environment(
                    mock_trainml,
                    **{
                        "id": "PYTORCH_PY37_15",
                        "framework": "PyTorch",
                        "py_version": "3.7",
                        "version": "1.5",
                        "cuda_version": "10.1",
                        "name": "PyTorch 1.5 - Python 3.7",
                    },
                ),
                Environment(
                    mock_trainml,
                    **{
                        "id": "TENSORFLOW_PY38_24",
                        "framework": "Tensorflow",
                        "py_version": "3.8",
                        "version": "2.4",
                        "cuda_version": "11.1",
                        "name": "Tensorflow 2.4 - Python 3.8",
                    },
                ),
                Environment(
                    mock_trainml,
                    **{
                        "id": "TENSORFLOW_PY37_114",
                        "framework": "Tensorflow",
                        "py_version": "3.7",
                        "version": "1.14",
                        "cuda_version": "10.1",
                        "name": "Tensorflow 1.14 - Python 3.7",
                    },
                ),
            ]
        )
        result = runner.invoke(specimen, ["list"])
        assert result.exit_code == 0
        assert "DEEPLEARNING_PY38" in result.output
