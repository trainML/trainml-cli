import re
import logging
import json
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.jobs as specimen
from trainml.exceptions import (
    ApiError,
    JobError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.jobs]


@fixture
def jobs(mock_trainml):
    yield specimen.Jobs(mock_trainml)


@fixture
def job(mock_trainml):
    yield specimen.Job(
        mock_trainml,
        **{
            "customer_uuid": "cus-id-1",
            "project_uuid": "proj-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "start": "2021-02-11T15:46:22.455Z",
            "type": "notebook",
            "status": "new",
            "credits_per_hour": 0.1,
            "credits": 0.1007,
            "workers": [
                {
                    "rig_uuid": "rig-id-1",
                    "job_worker_uuid": "worker-id-1",
                    "command": "jupyter lab",
                    "status": "new",
                }
            ],
            "worker_status": "new",
            "resources": {
                "gpu_count": 1,
                "gpu_type_id": "1060-id",
                "disk_size": 10,
                "max_price": 10,
            },
            "model": {
                "size": 7176192,
                "git_uri": "git@github.com:trainML/test-private.git",
                "status": "new",
            },
            "data": {
                "datasets": [
                    {
                        "dataset_uuid": "data-id-1",
                        "name": "first one",
                        "type": "public",
                        "size": 184549376,
                    },
                    {
                        "dataset_uuid": "data-id-2",
                        "name": "second one",
                        "type": "public",
                        "size": 5068061409,
                    },
                ],
                "status": "ready",
            },
            "environment": {
                "type": "DEEPLEARNING_PY38",
                "image_size": 44966989795,
                "env": [
                    {"value": "env1val", "key": "env1"},
                    {"value": "env2val", "key": "env2"},
                ],
                "worker_key_types": ["aws", "gcp"],
                "status": "new",
            },
            "vpn": {
                "status": "new",
                "cidr": "10.106.171.0/24",
                "client": {
                    "port": "36017",
                    "id": "cus-id-1",
                    "address": "10.106.171.253",
                },
                "net_prefix_type_id": 1,
            },
            "nb_token": "token",
        },
    )


@fixture
def training_job(mock_trainml):
    yield specimen.Job(
        mock_trainml,
        **{
            "customer_uuid": "cus-id-1",
            "project_uuid": "proj-id-1",
            "job_uuid": "job-id-1",
            "name": "test training",
            "start": "2021-02-11T15:46:22.455Z",
            "type": "training",
            "status": "running",
            "credits_per_hour": 0.1,
            "credits": 0.1007,
            "workers": [
                {
                    "rig_uuid": "rig-id-1",
                    "job_worker_uuid": "worker-id-1",
                    "command": "python train.py",
                    "status": "running",
                }
            ],
            "worker_status": "running",
            "resources": {
                "gpu_count": 1,
                "gpu_type_id": "1060-id",
                "disk_size": 10,
                "max_price": 10,
            },
            "model": {
                "size": 7176192,
                "git_uri": "git@github.com:trainML/test-private.git",
                "status": "new",
            },
            "data": {
                "datasets": [
                    {
                        "dataset_uuid": "data-id-1",
                        "name": "first one",
                        "type": "public",
                        "size": 184549376,
                    },
                    {
                        "dataset_uuid": "data-id-2",
                        "name": "second one",
                        "type": "public",
                        "size": 5068061409,
                    },
                ],
                "status": "ready",
            },
            "environment": {
                "type": "DEEPLEARNING_PY38",
                "image_size": 44966989795,
                "env": [
                    {"value": "env1val", "key": "env1"},
                    {"value": "env2val", "key": "env2"},
                ],
                "worker_key_types": ["aws", "gcp"],
                "status": "ready",
            },
            "vpn": {
                "status": "running",
                "cidr": "10.106.171.0/24",
                "client": {
                    "port": "36017",
                    "id": "cus-id-1",
                    "address": "10.106.171.253",
                },
            },
        },
    )


class CleanDatasetSelectionTests:
    @mark.parametrize(
        "datasets,expected",
        [
            (
                [dict(id="1", type="existing")],
                [dict(id="1", type="existing")],
            ),
            (
                [dict(name="first one", type="existing")],
                [dict(id="first one", type="existing")],
            ),
            (
                [dict(dataset_uuid="3", type="existing")],
                [dict(dataset_uuid="3", type="existing")],
            ),
            (
                [dict(name="first one", type="public")],
                [dict(id="first one", type="public")],
            ),
            (
                [
                    dict(id="1", type="existing"),
                    dict(name="second one", type="existing"),
                    dict(id="11", type="public"),
                    dict(name="second one", type="public"),
                ],
                [
                    dict(id="1", type="existing"),
                    dict(id="second one", type="existing"),
                    dict(id="11", type="public"),
                    dict(id="second one", type="public"),
                ],
            ),
        ],
    )
    def test_clean_datasets_selection_successful(
        self,
        datasets,
        expected,
    ):
        result = specimen._clean_datasets_selection(datasets)
        assert result == expected

    def test_invalid_dataset_type_specified(self):
        with raises(SpecificationError):
            specimen._clean_datasets_selection(
                [dict(name="Not Found", type="invalid")],
            )

    def test_invalid_dataset_identifier_specified(self):
        with raises(SpecificationError):
            specimen._clean_datasets_selection(
                [dict(dataset_id="Not Found", type="existing")],
            )


@mark.asyncio
class JobsTests:
    async def test_jobs_get(
        self,
        jobs,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await jobs.get("1234")
        mock_trainml._query.assert_called_once_with("/job/1234", "GET")

    async def test_jobs_list(
        self,
        jobs,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await jobs.list()
        mock_trainml._query.assert_called_once_with("/job", "GET")

    async def test_jobs_remove(
        self,
        jobs,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await jobs.remove("1234")
        mock_trainml._query.assert_called_once_with(
            "/job/1234", "DELETE", dict(force=True)
        )

    async def test_job_create_minimal(
        self,
        jobs,
        mock_trainml,
    ):
        requested_config = dict(
            name="job_name",
            type="notebook",
            gpu_type="GTX 1060",
            gpu_count=1,
            disk_size=10,
        )
        expected_payload = dict(
            project_uuid="proj-id-a",
            name="job_name",
            type="notebook",
            resources=dict(
                gpu_type_id="GTX 1060", gpu_count=1, disk_size=10, max_price=10
            ),
            environment=dict(type="DEEPLEARNING_PY39"),
            model=dict(),
            worker_commands=[],
        )

        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await jobs.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/job", "POST", None, expected_payload
        )

    async def test_job_create_from_empty_copy(
        self,
        jobs,
        mock_trainml,
    ):
        requested_config = dict(
            name="job_name",
            type="notebook",
            gpu_type="1060-id",
            gpu_count=1,
            disk_size=10,
            max_price=2,
            worker_commands=None,
            environment=None,
            data=None,
            source_job_uuid="job-id-1",
        )
        expected_payload = dict(
            name="job_name",
            project_uuid="proj-id-a",
            type="notebook",
            resources=dict(
                gpu_type_id="1060-id", gpu_count=1, disk_size=10, max_price=2
            ),
            source_job_uuid="job-id-1",
        )

        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await jobs.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/job", "POST", None, expected_payload
        )


class JobTests:
    def test_job_properties(self, job):
        assert isinstance(job.id, str)
        assert isinstance(job.name, str)
        assert isinstance(job.status, str)
        assert isinstance(job.type, str)

    def test_job_str(self, job):
        string = str(job)
        regex = r"^{.*\"job_uuid\": \"" + job.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_job_repr(self, job):
        string = repr(job)
        regex = r"^Job\( trainml , {.*'job_uuid': '" + job.id + r"'.*}\)$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_job_bool(self, job, mock_trainml):
        empty_job = specimen.Job(mock_trainml)
        assert bool(job)
        assert not bool(empty_job)

    @mark.asyncio
    async def test_job_start(self, job, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await job.start()
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1", "PATCH", None, dict(command="start")
        )

    @mark.asyncio
    async def test_job_stop(self, job, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await job.stop()
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1", "PATCH", None, dict(command="stop")
        )

    @mark.asyncio
    async def test_job_get_worker_log_url(self, job, mock_trainml):

        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/job-id-1/logs/worker-id-1/test_notebook.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await job.get_worker_log_url("worker-id-1")
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1/worker/worker-id-1/logs", "GET"
        )
        assert response == api_response

    @mark.asyncio
    async def test_job_get_connection_utility_url(self, job, mock_trainml):

        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/job-id-1/vpn/trainml-test_notebook.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await job.get_connection_utility_url()
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1/download", "GET"
        )
        assert response == api_response

    def test_job_get_connection_details_no_data(self, job):
        details = job.get_connection_details()
        expected_details = dict(
            project_uuid="proj-id-1",
            entity_type="job",
            cidr="10.106.171.0/24",
            ssh_port=None,
            model_path=None,
            input_path=None,
            output_path=None,
        )
        assert details == expected_details

    def test_job_get_connection_details_local_output_data(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test notebook",
                "type": "notebook",
                "status": "new",
                "model": {},
                "data": {
                    "datasets": [],
                    "output_type": "local",
                    "output_uri": "~/tensorflow-example/output",
                    "status": "ready",
                },
                "vpn": {
                    "status": "new",
                    "cidr": "10.106.171.0/24",
                    "client": {
                        "port": "36017",
                        "id": "cus-id-1",
                        "address": "10.106.171.253",
                        "ssh_port": 46600,
                    },
                },
            },
        )
        details = job.get_connection_details()
        expected_details = dict(
            project_uuid="proj-id-1",
            entity_type="job",
            cidr="10.106.171.0/24",
            ssh_port=46600,
            model_path=None,
            input_path=None,
            output_path="~/tensorflow-example/output",
        )
        assert details == expected_details

    def test_job_get_connection_details_local_model_and_input_data(
        self, mock_trainml
    ):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test notebook",
                "type": "notebook",
                "status": "new",
                "model": {"source_type": "local", "source_uri": "~/model_dir"},
                "data": {
                    "datasets": [],
                    "input_type": "local",
                    "input_uri": "~/data_dir",
                    "status": "ready",
                },
                "vpn": {
                    "status": "new",
                    "cidr": "10.106.171.0/24",
                    "client": {
                        "port": "36017",
                        "id": "cus-id-1",
                        "address": "10.106.171.253",
                        "ssh_port": 46600,
                    },
                },
            },
        )
        details = job.get_connection_details()
        expected_details = dict(
            project_uuid="proj-id-1",
            entity_type="job",
            cidr="10.106.171.0/24",
            ssh_port=46600,
            model_path="~/model_dir",
            input_path="~/data_dir",
            output_path=None,
        )
        assert details == expected_details

    @mark.asyncio
    async def test_job_connect(self, training_job, mock_trainml):
        with patch(
            "trainml.jobs.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "connected"
            resp = await training_job.connect()
            connection.start.assert_called_once()
            assert resp == "connected"

    @mark.asyncio
    async def test_job_disconnect(self, training_job, mock_trainml):
        with patch(
            "trainml.jobs.Connection",
            autospec=True,
        ) as mock_connection:
            connection = mock_connection.return_value
            connection.status = "removed"
            resp = await training_job.disconnect()
            connection.stop.assert_called_once()
            assert resp == "removed"

    @mark.asyncio
    async def test_job_remove(self, job, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await job.remove()
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1", "DELETE", dict(force=False)
        )

    @mark.asyncio
    async def test_job_refresh(self, job, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "running",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await job.refresh()
        mock_trainml._query.assert_called_once_with("/job/job-id-1", "GET")
        assert job.status == "running"
        assert response.status == "running"

    def test_job_default_ws_msg_handler(self, job, capsys):

        data = {
            "msg": "Epoch (1/1000)\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = job._get_msg_handler(None)
        handler(data)
        captured = capsys.readouterr()
        assert captured.out == "02/11/2021, 15:35:45: Epoch (1/1000)\n"

    def test_job_default_ws_msg_handler_multiple_workers(
        self, mock_trainml, capsys
    ):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "job_uuid": "job-id-1",
                "name": "test notebook",
                "type": "notebook",
                "status": "running",
                "workers": [
                    {
                        "rig_uuid": "rig-id-1",
                        "job_worker_uuid": "worker-id-1",
                        "command": "jupyter lab",
                        "status": "running",
                    },
                    {
                        "rig_uuid": "rig-id-1",
                        "job_worker_uuid": "worker-id-2",
                        "command": "jupyter lab",
                        "status": "running",
                    },
                ],
            },
        )

        data = {
            "msg": "Epoch (1/1000)\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = job._get_msg_handler(None)
        handler(data)
        captured = capsys.readouterr()
        assert (
            captured.out == "02/11/2021, 15:35:45: Worker 1 - Epoch (1/1000)\n"
        )

    def test_job_custom_ws_msg_handler(self, job, capsys):
        def custom_handler(msg):
            print(msg.get("stream"))

        data = {
            "msg": "Epoch (1/1000)\n",
            "time": 1613079345318,
            "type": "subscription",
            "stream": "worker-id-1",
            "job_worker_uuid": "worker-id-1",
        }

        handler = job._get_msg_handler(custom_handler)
        handler(data)
        captured = capsys.readouterr()
        assert captured.out == "worker-id-1\n"

    @mark.asyncio
    async def test_notebook_attach_error(self, job, mock_trainml):
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        with raises(SpecificationError):
            await job.attach()
        mock_trainml._ws_subscribe.create.assert_not_called()

    @mark.asyncio
    async def test_job_attach(self, mock_trainml):
        job_spec = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test training",
            "type": "training",
            "status": "running",
            "workers": [
                {
                    "rig_uuid": "rig-id-1",
                    "job_worker_uuid": "worker-id-1",
                    "command": "jupyter lab",
                    "status": "running",
                },
                {
                    "rig_uuid": "rig-id-1",
                    "job_worker_uuid": "worker-id-2",
                    "command": "jupyter lab",
                    "status": "running",
                },
            ],
        }
        job = specimen.Job(
            mock_trainml,
            **job_spec,
        )
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        job.refresh = AsyncMock(return_value=job_spec)
        await job.attach()
        mock_trainml._ws_subscribe.assert_called_once()

    @mark.asyncio
    async def test_job_attach_immediate_return(self, mock_trainml):
        job_spec = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test training",
            "type": "training",
            "status": "finished",
            "workers": [],
        }
        job = specimen.Job(
            mock_trainml,
            **job_spec,
        )
        api_response = None
        mock_trainml._ws_subscribe = AsyncMock(return_value=api_response)
        job.refresh = AsyncMock(return_value=job_spec)
        await job.attach()
        mock_trainml._ws_subscribe.assert_not_called()

    @mark.asyncio
    async def test_job_copy_minimal(
        self,
        job,
        mock_trainml,
    ):
        requested_config = dict(
            name="job_copy",
        )
        expected_kwargs = {
            "type": "notebook",
            "gpu_type": "1060-id",
            "gpu_count": 1,
            "disk_size": 10,
            "max_price": 10,
            "worker_commands": None,
            "workers": None,
            "environment": None,
            "data": None,
            "source_job_uuid": "job-id-1",
        }

        api_response = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "job_uuid": "job-id-2",
                "name": "job_copy",
                "type": "notebook",
                "status": "new",
            },
        )
        mock_trainml.jobs.create = AsyncMock(return_value=api_response)
        new_job = await job.copy(**requested_config)
        mock_trainml.jobs.create.assert_called_once_with(
            "job_copy", **expected_kwargs
        )
        assert new_job.id == "job-id-2"

    @mark.asyncio
    async def test_job_copy_training_job_failure(
        self,
        mock_trainml,
    ):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "job_uuid": "job-id-1",
                "name": "test training",
                "type": "training",
                "status": "running",
            },
        )
        mock_trainml.jobs.create = AsyncMock()
        with raises(SpecificationError):
            await job.copy("new job")
        mock_trainml.jobs.create.assert_not_called()

    @mark.asyncio
    async def test_job_wait_for_successful(self, job, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "running",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await job.wait_for("running")
        mock_trainml._query.assert_called_once_with("/job/job-id-1", "GET")
        assert job.status == "running"
        assert response.status == "running"

    @mark.asyncio
    async def test_job_wait_for_current_status(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "job_uuid": "job-id-1",
                "name": "test notebook",
                "type": "notebook",
                "status": "running",
            },
        )
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        await job.wait_for("running")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_job_wait_for_incorrect_status(self, job, mock_trainml):
        api_response = None
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(SpecificationError):
            await job.wait_for("ready")
        mock_trainml._query.assert_not_called()

    @mark.asyncio
    async def test_job_wait_for_with_delay(self, job, mock_trainml):
        api_response_initial = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "new",
        }
        api_response_final = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "running",
        }
        mock_trainml._query = AsyncMock()
        mock_trainml._query.side_effect = [
            api_response_initial,
            api_response_initial,
            api_response_final,
        ]
        response = await job.wait_for("running")
        assert job.status == "running"
        assert response.status == "running"

    @mark.asyncio
    async def test_job_wait_for_timeout(self, job, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "new",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(TrainMLException):
            await job.wait_for("running", 10)
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_job_wait_for_job_failed(self, job, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "failed",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        with raises(JobError):
            await job.wait_for("running")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_job_wait_for_archived_succeeded(self, job, mock_trainml):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Job Not Found"))
        )
        await job.wait_for("archived")
        mock_trainml._query.assert_called()

    @mark.asyncio
    async def test_job_wait_for_unexpected_api_error(self, job, mock_trainml):
        mock_trainml._query = AsyncMock(
            side_effect=ApiError(404, dict(errorMessage="Job Not Found"))
        )
        with raises(ApiError):
            await job.wait_for("running")
        mock_trainml._query.assert_called()