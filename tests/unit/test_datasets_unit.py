import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.datasets as specimen
from trainml.exceptions import (
    ApiError,
    DatasetError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.datasets]


@fixture
def datasets(mock_trainml):
    yield specimen.Datasets(mock_trainml)


@fixture
def dataset(mock_trainml):
    yield specimen.Dataset(
        mock_trainml,
        dataset_uuid="1",
        project_uuid="proj-id-a",
        name="first one",
        status="new",
        size=100000,
        createdAt="2020-12-31T23:59:59.000Z",
    )


class DatasetsTests:
    @mark.asyncio
    async def test_get_dataset(
        self,
        datasets,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datasets.get("1234")
        mock_trainml._query.assert_called_once_with("/dataset/pub/1234", "GET")

    @mark.asyncio
    async def test_list_datasets(
        self,
        datasets,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datasets.list()
        mock_trainml._query.assert_called_once_with("/dataset/pub", "GET")

    @mark.asyncio
    async def test_list_public_datasets(self, datasets, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datasets.list_public()
        mock_trainml._query.assert_called_once_with(
            "/dataset/pub/public", "GET"
        )

    @mark.asyncio
    async def test_remove_dataset(
        self,
        datasets,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await datasets.remove("4567")
        mock_trainml._query.assert_called_once_with(
            "/dataset/pub/4567", "DELETE", dict(force=True)
        )

    @mark.asyncio
    async def test_create_dataset_simple(self, datasets, mock_trainml):
        requested_config = dict(
            name="new dataset",
            source_type="aws",
            source_uri="s3://trainml-examples/data/cifar10",
        )
        expected_payload = dict(
            project_uuid="proj-id-a",
            name="new dataset",
            source_type="aws",
            source_uri="s3://trainml-examples/data/cifar10",
        )
        api_response = {
            "customer_uuid": "cus-id-1",
            "project_uuid": "proj-id-a",
            "dataset_uuid": "data-id-1",
            "name": "new dataset",
            "status": "new",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await datasets.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/dataset/pub", "POST", None, expected_payload
        )
        assert response.id == "data-id-1"


class DatasetTests:
    def test_dataset_properties(self, dataset):
        assert isinstance(dataset.id, str)
        assert isinstance(dataset.status, str)
        assert isinstance(dataset.name, str)
        assert isinstance(dataset.size, int)

    def test_dataset_str(self, dataset):
        string = str(dataset)
        regex = r"^{.*\"dataset_uuid\": \"" + dataset.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_dataset_repr(self, dataset):
        string = repr(dataset)
        regex = (
            r"^Dataset\( trainml , \*\*{.*'dataset_uuid': '"
            + dataset.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_dataset_bool(self, dataset, mock_trainml):
        empty_dataset = specimen.Dataset(mock_trainml)
        assert bool(dataset)
        assert not bool(empty_dataset)

    @mark.asyncio
    async def test_dataset_get_log_url(self, dataset, mock_trainml):

        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/logs/first_one.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await dataset.get_log_url()
        mock_trainml._query.assert_called_once_with(
            "/dataset/pub/1/logs", "GET"
        )
        assert response == api_response

    @mark.asyncio
    async def test_dataset_get_details(self, dataset, mock_trainml):

        api_response = {
            "type": "directory",
            "name": "/",
            "count": "8",
            "size": "177M",
            "contents": [],
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await dataset.get_details()
        mock_trainml._query.assert_called_once_with(
            "/dataset/pub/1/details", "GET"
        )
        assert response == api_response

    @mark.asyncio
    async def test_dataset_get_connection_utility_url(
        self, dataset, mock_trainml
    ):

        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/vpn/first_one.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await dataset.get_connection_utility_url()
        mock_trainml._query.assert_called_once_with(
            "/dataset/pub/1/download", "GET"
        )
        assert response == api_response

    def test_dataset_get_connection_details_no_vpn(self, dataset):
        details = dataset.get_connection_details()
        expected_details = dict()
        assert details == expected_details

    def test_dataset_get_connection_details_local_data(self, mock_trainml):
        dataset = specimen.Dataset(
            mock_trainml,
            dataset_uuid="1",
            project_uuid="proj-id-a",
            name="first one",
            status="new",
            size=100000,
            createdAt="2020-12-31T23:59:59.000Z",
            source_type="local",
            source_uri="~/tensorflow-example/data",
            vpn={
                "status": "new",
                "cidr": "10.106.171.0/24",
                "client": {
                    "port": "36017",
                    "id": "cus-id-1",
                    "address": "10.106.171.253",
                    "ssh_port": 46600,
                },
            },
        )
        details = dataset.get_connection_details()
        expected_details = dict(
            project_uuid="proj-id-a",
            entity_type="dataset",
            cidr="10.106.171.0/24",
            ssh_port=46600,
            input_path="~/tensorflow-example/data",
            output_path=None,
        )
        assert details == expected_details

    @mark.asyncio
    async def test_dataset_connect(self, dataset, mock_trainml):
        with patch(
            "trainml.datasets.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "connected"
            resp = await dataset.connect()
            connection.start.assert_called_once()
            assert resp == "connected"

    @mark.asyncio
    async def test_dataset_disconnect(self, dataset, mock_trainml):
        with patch(
            "trainml.datasets.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "removed"
            resp = await dataset.disconnect()
            connection.stop.assert_called_once()
            assert resp == "removed"

    @mark.asyncio
    async def test_dataset_remove(self, dataset, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await dataset.remove()
        mock_trainml._query.assert_called_once_with(
            "/dataset/pub/1", "DELETE", dict(force=False)
        )

    def test_dataset_default_ws_msg_handler(self, dataset, capsys):

        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = dataset._get_msg_handler(None)
        handler(data)
        captured = capsys.readouterr()
        assert (
            captured.out
            == "02/11/2021, 15:35:45: download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n"
        )

    def test_dataset_custom_ws_msg_handler(self, dataset, capsys):
        def custom_handler(msg):
            print(msg.get("stream"))

        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = dataset._get_msg_handler(custom_handler)
        handler(data)
        captured = capsys.readouterr()
        assert captured.out == "worker-id-1\n"

    @mark.asyncio
    async def test_dataset_attach(self, dataset, mock_trainml):
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "dataset_uuid": "data-id-1",
            "name": "new dataset",
            "status": "downloading",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        dataset.refresh = AsyncMock(return_value=refresh_response)
        await dataset.attach()
        mock_trainml._ws_subscribe.assert_called_once()

    @mark.asyncio
    async def test_dataset_attach_immediate_return(self, mock_trainml):
        dataset = specimen.Dataset(
            mock_trainml,
            dataset_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "dataset_uuid": "1",
            "name": "new dataset",
            "status": "ready",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        dataset.refresh = AsyncMock(return_value=refresh_response)
        await dataset.attach()
        mock_trainml._ws_subscribe.assert_not_called()

    @mark.asyncio
    async def test_dataset_refresh(self, dataset, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "dataset_uuid": "data-id-1",
            "name": "new dataset",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await dataset.refresh()
        mock_trainml._query.assert_called_once_with(f"/dataset/pub/1", "GET")
        assert dataset.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_dataset_wait_for_successful(self, dataset, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "dataset_uuid": "data-id-1",
            "name": "new dataset",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await dataset.wait_for("ready")
        mock_trainml._query.assert_called_once_with(f"/dataset/pub/1", "GET")
        assert dataset.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_dataset_wait_for_current_status(self, mock_trainml):
        dataset = specimen.Dataset(
            mock_trainml,
            dataset_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await dataset.wait_for("ready")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_dataset_wait_for_incorrect_status(
        self, dataset, mock_trainml
    ):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(SpecificationError):
            await dataset.wait_for("stopped")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_dataset_wait_for_with_delay(self, dataset, mock_trainml):
        api_response_initial = dict(
            dataset_uuid="1",
            name="first one",
            status="new",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response_final = dict(
            dataset_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock()
        mock_trainml._query.side_effect = [
            api_response_initial,
            api_response_initial,
            api_response_final,
        ]
        response = await dataset.wait_for("ready")
        assert dataset.status == "ready"
        assert response.status == "ready"

    @mark.asyncio
    async def test_dataset_wait_for_timeout(self, dataset, mock_trainml):
        api_response = dict(
            dataset_uuid="1",
            name="first one",
            status="new",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(TrainMLException):
            await dataset.wait_for("ready", 10)
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_dataset_wait_for_dataset_failed(
        self, dataset, mock_trainml
    ):
        api_response = dict(
            dataset_uuid="1",
            name="first one",
            status="failed",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(DatasetError):
            await dataset.wait_for("ready")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_dataset_wait_for_archived_succeeded(
        self, dataset, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Dataset Not Found"))
        )
        await dataset.wait_for("archived")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_dataset_wait_for_unexpected_api_error(
        self, dataset, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Dataset Not Found"))
        )
        with raises(ApiError):
            await dataset.wait_for("ready")
        mock_trainml._query.assert_called()