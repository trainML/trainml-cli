import re
import json
import os
from unittest.mock import (
    Mock,
    AsyncMock,
    patch,
    create_autospec,
    call,
    mock_open,
)
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType
from aiodocker.containers import DockerContainer

import trainml.connections as specimen
from trainml.jobs import Job
from trainml.datasets import Dataset
from trainml.exceptions import ApiError, ConnectionError

pytestmark = [mark.sdk, mark.unit, mark.connections]


@fixture
def connections(mock_trainml):
    yield specimen.Connections(mock_trainml)


@fixture
def dataset_con(mock_trainml):
    DatasetMock = create_autospec(Dataset)
    yield specimen.Connection(
        mock_trainml,
        entity_type="dataset",
        id="data-id-1",
        entity=DatasetMock(
            mock_trainml,
            dataset_uuid="data-id-1",
            name="first one",
            status="new",
            provider="trainml",
        ),
    )


@fixture
def job_con(mock_trainml):
    JobMock = create_autospec(Job)
    yield specimen.Connection(
        mock_trainml,
        entity_type="job",
        id="job-id-1",
        entity=JobMock(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "job_uuid": "job-id-1",
                "name": "test notebook",
                "type": "interactive",
                "status": "new",
                "provider": "trainml",
            },
        ),
    )


class ConnectionsTests:
    @mark.asyncio
    async def test_connections_list(
        self,
        connections,
        mock_trainml,
    ):
        with patch("trainml.connections.os") as mock_os:
            mock_os.listdir = Mock(
                return_value=["job_job-id-1", "dataset_data-id-1", "baddir"]
            )
            with patch(
                "trainml.connections.Connection", autospec=True
            ) as mock_connection:
                connection = mock_connection.return_value
                connections = await connections.list()
                connection.check.assert_called
                assert len(connections) == 2

    @mark.asyncio
    async def test_connections_cleanup(
        self,
        connections,
        mock_trainml,
    ):
        with patch("trainml.connections.os") as mock_os:
            dir_list = ["job_job-id-1", "dataset_data-id-1"]
            mock_os.listdir = Mock(return_value=dir_list)
            with patch(
                "trainml.connections._cleanup_containers"
            ) as mock_cleanup:
                await connections.cleanup()
                assert mock_cleanup.mock_calls == [
                    call(
                        os.path.expanduser("~/.trainml/connections"),
                        dir_list,
                        "vpn",
                    ),
                    call(
                        os.path.expanduser("~/.trainml/connections"),
                        dir_list,
                        "storage",
                    ),
                ]

    @mark.asyncio
    async def test_connections_cleanup_containers(
        self,
        mock_trainml,
    ):
        with patch("trainml.connections.open", mock_open(read_data="keep-me")):

            with patch(
                "trainml.connections.aiodocker.Docker", autospec=True
            ) as mock_docker:
                containers = [
                    create_autospec(DockerContainer),
                    create_autospec(DockerContainer),
                ]
                containers[0].id = "keep-me"
                containers[1].id = "delete-me"
                docker = mock_docker.return_value
                docker.containers = AsyncMock()
                docker.containers.list = AsyncMock(return_value=containers)
                await specimen._cleanup_containers(
                    os.path.expanduser("~/.trainml/connections"),
                    ["dir_1"],
                    "job",
                )
                docker.containers.list.assert_called_once_with(
                    all=True,
                    filters=json.dumps(
                        dict(label=["service=trainml", "type=job"])
                    ),
                )
                containers[0].delete.assert_not_called
                containers[1].delete.assert_called_once_with(force=True)


class ConnectionTests:
    def test_connection_properties(self, dataset_con):
        assert isinstance(dataset_con.id, str)
        assert isinstance(dataset_con.status, str)
        assert isinstance(dataset_con.type, str)
        assert dataset_con.type == "dataset"

    def test_connection_str(self, dataset_con):
        string = str(dataset_con)
        regex = (
            r"^Connection for "
            + dataset_con.type
            + " - "
            + dataset_con.id
            + ": "
            + dataset_con.status
            + r"$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_connection_repr(self, dataset_con):
        string = repr(dataset_con)
        regex = (
            r"^Connection\( trainml , "
            + dataset_con.id
            + ", "
            + dataset_con.type
            + r"\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
