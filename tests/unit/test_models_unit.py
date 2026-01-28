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
        project_uuid="proj-id-1",
        name="first one",
        type="evefs",
        status="downloading",
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
        mock_trainml._query.assert_called_once_with("/model/1234", "GET", dict())

    @mark.asyncio
    async def test_list_models(
        self,
        models,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await models.list()
        mock_trainml._query.assert_called_once_with("/model", "GET", dict())

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
            "/model/4567", "DELETE", dict(force=True)
        )

    @mark.asyncio
    async def test_create_model_simple(self, models, mock_trainml):
        requested_config = dict(
            name="new model",
            source_type="aws",
            source_uri="s3://trainml-examples/models/resnet50",
        )
        expected_payload = dict(
            project_uuid="proj-id-1",
            name="new model",
            source_type="aws",
            source_uri="s3://trainml-examples/models/resnet50",
            type="evefs",
        )
        api_response = {
            "customer_uuid": "cus-id-1",
            "model_uuid": "model-id-1",
            "name": "new model",
            "status": "new",
            "type": "evefs",
            "source_type": "aws",
            "source_uri": "s3://trainml-examples/models/resnet50",
            "createdAt": "2020-12-20T16:46:23.909Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await models.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/model", "POST", None, expected_payload
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
        regex = r"^Model\( trainml , \*\*{.*'model_uuid': '" + model.id + r"'.*}\)$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_model_bool(self, model, mock_trainml):
        empty_model = specimen.Model(mock_trainml)
        assert bool(model)
        assert not bool(empty_model)

    @mark.asyncio
    async def test_model_get_log_url(self, model, mock_trainml):
        api_response = (
            "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/1/logs/first_one.zip"
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await model.get_log_url()
        mock_trainml._query.assert_called_once_with(
            "/model/1/logs", "GET", dict(project_uuid="proj-id-1")
        )
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
            "/model/1/details", "GET", dict(project_uuid="proj-id-1")
        )
        assert response == api_response

    @mark.asyncio
    async def test_model_connect_downloading_status(self, mock_trainml):
        model = specimen.Model(
            mock_trainml,
            model_uuid="1",
            project_uuid="proj-id-1",
            name="test model",
            status="downloading",
            auth_token="test-token",
            hostname="example.com",
            source_uri="/path/to/source",
        )
        
        with patch("trainml.models.Model.refresh", new_callable=AsyncMock) as mock_refresh:
            with patch("trainml.models.upload", new_callable=AsyncMock) as mock_upload:
                await model.connect()
                mock_refresh.assert_called_once()
                mock_upload.assert_called_once_with("example.com", "test-token", "/path/to/source")

    @mark.asyncio
    async def test_model_connect_exporting_status(self, mock_trainml, tmp_path):
        output_dir = str(tmp_path / "output")
        model = specimen.Model(
            mock_trainml,
            model_uuid="1",
            project_uuid="proj-id-1",
            name="test model",
            status="exporting",
            auth_token="test-token",
            hostname="example.com",
            output_uri=output_dir,
        )
        
        with patch("trainml.models.Model.refresh", new_callable=AsyncMock) as mock_refresh:
            with patch("trainml.models.download", new_callable=AsyncMock) as mock_download:
                await model.connect()
                mock_refresh.assert_called_once()
                mock_download.assert_called_once_with("example.com", "test-token", output_dir)

    @mark.asyncio
    async def test_model_connect_invalid_status(self, mock_trainml):
        model = specimen.Model(
            mock_trainml,
            model_uuid="1",
            project_uuid="proj-id-1",
            name="test model",
            status="ready",
        )
        
        with raises(SpecificationError, match="You can only connect to downloading or exporting models"):
            await model.connect()

    @mark.asyncio
    async def test_model_remove(self, model, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await model.remove()
        mock_trainml._query.assert_called_once_with(
            "/model/1",
            "DELETE",
            dict(project_uuid="proj-id-1", force=False),
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
        mock_trainml._query.assert_called_once_with(
            f"/model/1", "GET", dict(project_uuid="proj-id-1")
        )
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
        mock_trainml._query.assert_called_once_with(
            f"/model/1", "GET", dict(project_uuid="proj-id-1")
        )
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
    async def test_model_wait_for_archived_succeeded(self, model, mock_trainml):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Model Not Found"))
        )
        await model.wait_for("archived")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_model_wait_for_unexpected_api_error(self, model, mock_trainml):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Model Not Found"))
        )
        with raises(ApiError):
            await model.wait_for("ready")
        mock_trainml._query.assert_called()
