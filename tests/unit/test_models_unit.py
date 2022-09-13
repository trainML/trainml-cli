import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.models as specimen
from trainml.exceptions import (
    ApiError,
    ModelError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.models]


@fixture
def models(mock_trainml):
    yield specimen.Models(mock_trainml)


@fixture
def model(mock_trainml):
    yield specimen.Model(
        mock_trainml,
        model_uuid="1",
        project_uuid="a",
        name="first one",
        status="new",
        size=100000,
        createdAt="2020-12-31T23:59:59.000Z",
    )


class ModelsTests:
    @mark.asyncio
    async def test_get_model(
        self,
        models,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await models.get("1234")
        mock_trainml._query.assert_called_once_with("/model/pub/1234", "GET")

    @mark.asyncio
    async def test_list_models(
        self,
        models,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await models.list()
        mock_trainml._query.assert_called_once_with("/model/pub", "GET")

    @mark.asyncio
    async def test_remove_model(
        self,
        models,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await models.remove("4567")
        mock_trainml._query.assert_called_once_with(
            "/model/pub/4567", "DELETE", dict(force=True)
        )

    @mark.asyncio
    async def test_create_model_simple(self, models, mock_trainml):
        requested_config = dict(
            name="new model",
            source_type="aws",
            source_uri="s3://trainml-examples/models/resnet50",
        )
        expected_payload = dict(
            project_uuid="proj-id-a",
            name="new model",
            source_type="aws",
            source_uri="s3://trainml-examples/models/resnet50",
        )
        api_response = {
            "customer_uuid": "cus-id-1",
            "model_uuid": "model-id-1",
            "name": "new model",
            "status": "new",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/models/resnet50",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await models.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/model/pub", "POST", None, expected_payload
        )
        assert response.id == "model-id-1"


class ModelTests:
    def test_model_properties(self, model):
        assert isinstance(model.id, str)
        assert isinstance(model.status, str)
        assert isinstance(model.name, str)
        assert isinstance(model.size, int)

    def test_model_str(self, model):
        string = str(model)
        regex = r"^{.*\"model_uuid\": \"" + model.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_model_repr(self, model):
        string = repr(model)
        regex = (
            r"^Model\( trainml , \*\*{.*'model_uuid': '"
            + model.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_model_bool(self, model, mock_trainml):
        empty_model = specimen.Model(mock_trainml)
        assert bool(model)
        assert not bool(empty_model)

    @mark.asyncio
    async def test_model_get_log_url(self, model, mock_trainml):

        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/logs/first_one.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await model.get_log_url()
        mock_trainml._query.assert_called_once_with("/model/pub/1/logs", "GET")
        assert response == api_response

    @mark.asyncio
    async def test_model_get_details(self, model, mock_trainml):

        api_response = {
            "type": "directory",
            "name": "/",
            "count": "8",
            "size": "177M",
            "contents": [],
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await model.get_details()
        mock_trainml._query.assert_called_once_with(
            "/model/pub/1/details", "GET"
        )
        assert response == api_response

    @mark.asyncio
    async def test_model_get_connection_utility_url(self, model, mock_trainml):

        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/vpn/first_one.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await model.get_connection_utility_url()
        mock_trainml._query.assert_called_once_with(
            "/model/pub/1/download", "GET"
        )
        assert response == api_response

    def test_model_get_connection_details_no_vpn(self, model):
        details = model.get_connection_details()
        expected_details = dict()
        assert details == expected_details

    def test_model_get_connection_details_local_data(self, mock_trainml):
        model = specimen.Model(
            mock_trainml,
            model_uuid="1",
            project_uuid="a",
            name="first one",
            status="new",
            size=100000,
            createdAt="2020-12-31T23:59:59.000Z",
            source_type="local",
            source_uri="~/tensorflow-example",
            vpn={
                "status": "new",
                "cidr": "10.106.171.0/24",
                "client": {
                    "port": "36017",
                    "id": "cus-id-1",
                    "address": "10.106.171.253",
                    "ssh_port": 46600,
                },
                "net_prefix_type_id": 1,
            },
        )
        details = model.get_connection_details()
        expected_details = dict(
            project_uuid="a",
            entity_type="model",
            cidr="10.106.171.0/24",
            ssh_port=46600,
            input_path="~/tensorflow-example",
            output_path=None,
        )
        assert details == expected_details

    @mark.asyncio
    async def test_model_connect(self, model, mock_trainml):
        with patch(
            "trainml.models.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "connected"
            resp = await model.connect()
            connection.start.assert_called_once()
            assert resp == "connected"

    @mark.asyncio
    async def test_model_disconnect(self, model, mock_trainml):
        with patch(
            "trainml.models.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "removed"
            resp = await model.disconnect()
            connection.stop.assert_called_once()
            assert resp == "removed"

    @mark.asyncio
    async def test_model_remove(self, model, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await model.remove()
        mock_trainml._query.assert_called_once_with(
            "/model/pub/1", "DELETE", dict(force=False)
        )

    def test_model_default_ws_msg_handler(self, model, capsys):
        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = model._get_msg_handler(None)
        handler(data)
        captured = capsys.readouterr()
        assert (
            captured.out
            == "02/11/2021, 15:35:45: download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n"
        )

    def test_model_custom_ws_msg_handler(self, model, capsys):
        def custom_handler(msg):
            print(msg.get("stream"))

        data = {
            "msg": "download: s3://trainml-examples/data/cifar10/data_batch_2.bin to ./data_batch_2.bin\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = model._get_msg_handler(custom_handler)
        handler(data)
        captured = capsys.readouterr()
        assert captured.out == "worker-id-1\n"

    @mark.asyncio
    async def test_model_attach(self, model, mock_trainml):
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "model_uuid": "data-id-1",
            "name": "new model",
            "status": "downloading",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        model.refresh = AsyncMock(return_value=refresh_response)
        await model.attach()
        mock_trainml._ws_subscribe.assert_called_once()

    @mark.asyncio
    async def test_model_attach_immediate_return(self, mock_trainml):
        model = specimen.Model(
            mock_trainml,
            model_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        refresh_response = {
            "customer_uuid": "cus-id-1",
            "model_uuid": "1",
            "name": "new model",
            "status": "ready",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        model.refresh = AsyncMock(return_value=refresh_response)
        await model.attach()
        mock_trainml._ws_subscribe.assert_not_called()

    @mark.asyncio
    async def test_model_refresh(self, model, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "model_uuid": "data-id-1",
            "name": "new model",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await model.refresh()
        mock_trainml._query.assert_called_once_with(f"/model/pub/1", "GET")
        assert model.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_model_wait_for_successful(self, model, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "model_uuid": "data-id-1",
            "name": "new model",
            "status": "ready",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/data/cifar10",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await model.wait_for("ready")
        mock_trainml._query.assert_called_once_with(f"/model/pub/1", "GET")
        assert model.id == "data-id-1"
        assert response.id == "data-id-1"

    @mark.asyncio
    async def test_model_wait_for_current_status(self, mock_trainml):
        model = specimen.Model(
            mock_trainml,
            model_uuid="1",
            name="first one",
            status="ready",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await model.wait_for("ready")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_model_wait_for_incorrect_status(self, model, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(SpecificationError):
            await model.wait_for("stopped")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_model_wait_for_with_delay(self, model, mock_trainml):
        api_response_initial = dict(
            model_uuid="1",
            name="first one",
            status="new",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        api_response_final = dict(
            model_uuid="1",
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
        response = await model.wait_for("ready")
        assert model.status == "ready"
        assert response.status == "ready"

    @mark.asyncio
    async def test_model_wait_for_timeout(self, model, mock_trainml):
        api_response = dict(
            model_uuid="1",
            name="first one",
            status="new",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(TrainMLException):
            await model.wait_for("ready", 10)
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_model_wait_for_model_failed(self, model, mock_trainml):
        api_response = dict(
            model_uuid="1",
            name="first one",
            status="failed",
            createdAt="2020-12-31T23:59:59.000Z",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(ModelError):
            await model.wait_for("ready")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_model_wait_for_archived_succeeded(
        self, model, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Model Not Found"))
        )
        await model.wait_for("archived")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_model_wait_for_unexpected_api_error(
        self, model, mock_trainml
    ):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Model Not Found"))
        )
        with raises(ApiError):
            await model.wait_for("ready")
        mock_trainml._query.assert_called()