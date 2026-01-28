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
                "gpu_types": ["1060-id"],
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
                "type": "DEEPLEARNING_PY310",
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
                "type": "DEEPLEARNING_PY310",
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
        mock_trainml._query.assert_called_once_with("/job/1234", "GET", dict())

    async def test_jobs_list(
        self,
        jobs,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await jobs.list()
        mock_trainml._query.assert_called_once_with("/job", "GET", dict())

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
            project_uuid="proj-id-1",
            name="job_name",
            type="notebook",
            resources=dict(
                gpu_type_id="GTX 1060", gpu_count=1, disk_size=10, max_price=10
            ),
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
            project_uuid="proj-id-1",
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

    async def test_job_missing_gpu_type(
        self,
        jobs,
    ):
        requested_config = dict(
            name="job_name",
            type="notebook",
            disk_size=10,
        )

        with raises(SpecificationError) as error:
            await jobs.create(**requested_config)
        assert (
            "Invalid resource specification, either 'gpu_type' or 'gpu_types' must be provided"
            in error.value.message
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
        api_response = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "starting",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        await job.start()
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1",
            "PATCH",
            dict(project_uuid="proj-id-1"),
            dict(command="start"),
        )

    @mark.asyncio
    async def test_job_stop(self, job, mock_trainml):
        api_response = {
            "customer_uuid": "cus-id-1",
            "job_uuid": "job-id-1",
            "name": "test notebook",
            "type": "notebook",
            "status": "stopping",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        await job.stop()
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1",
            "PATCH",
            dict(project_uuid="proj-id-1"),
            dict(command="stop"),
        )

    @mark.asyncio
    async def test_job_get_worker_log_url(self, job, mock_trainml):
        api_response = "https://trainml-jobs-dev.s3.us-east-2.amazonaws.com/job-id-1/logs/worker-id-1/test_notebook.zip"
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await job.get_worker_log_url("worker-id-1")
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1/worker/worker-id-1/logs",
            "GET",
            dict(project_uuid="proj-id-1"),
        )
        assert response == api_response

    @mark.asyncio
    async def test_job_connect_waiting_for_data_model_download_local_model_only(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "waiting for data/model download",
                "model": {
                    "source_type": "local",
                    "auth_token": "model-token",
                    "hostname": "model-host.com",
                    "source_uri": "/path/to/model",
                },
                "data": {
                    "input_type": "trainml",
                },
            },
        )
        
        with patch("trainml.jobs.Job.refresh", new_callable=AsyncMock) as mock_refresh:
            with patch("trainml.jobs.upload", new_callable=AsyncMock) as mock_upload:
                await job.connect()
                mock_refresh.assert_called_once()
                mock_upload.assert_called_once_with("model-host.com", "model-token", "/path/to/model")

    @mark.asyncio
    async def test_job_connect_waiting_for_data_model_download_local_data_only(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "waiting for data/model download",
                "model": {
                    "source_type": "trainml",
                },
                "data": {
                    "input_type": "local",
                    "input_auth_token": "data-token",
                    "input_hostname": "data-host.com",
                    "input_uri": "/path/to/data",
                },
            },
        )
        
        with patch("trainml.jobs.Job.refresh", new_callable=AsyncMock) as mock_refresh:
            with patch("trainml.jobs.upload", new_callable=AsyncMock) as mock_upload:
                await job.connect()
                mock_refresh.assert_called_once()
                mock_upload.assert_called_once_with("data-host.com", "data-token", "/path/to/data")

    @mark.asyncio
    async def test_job_connect_waiting_for_data_model_download_both_local_parallel(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "waiting for data/model download",
                "model": {
                    "source_type": "local",
                    "auth_token": "model-token",
                    "hostname": "model-host.com",
                    "source_uri": "/path/to/model",
                },
                "data": {
                    "input_type": "local",
                    "input_auth_token": "data-token",
                    "input_hostname": "data-host.com",
                    "input_uri": "/path/to/data",
                },
            },
        )
        
        with patch("trainml.jobs.Job.refresh", new_callable=AsyncMock) as mock_refresh:
            with patch("trainml.jobs.upload", new_callable=AsyncMock) as mock_upload:
                await job.connect()
                mock_refresh.assert_called_once()
                assert mock_upload.call_count == 2
                # Verify both were called with correct parameters
                calls = mock_upload.call_args_list
                assert any(call[0] == ("model-host.com", "model-token", "/path/to/model") for call in calls)
                assert any(call[0] == ("data-host.com", "data-token", "/path/to/data") for call in calls)

    @mark.asyncio
    async def test_job_connect_waiting_for_data_model_download_neither_local_error(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "waiting for data/model download",
                "model": {
                    "source_type": "trainml",
                },
                "data": {
                    "input_type": "trainml",
                },
            },
        )
        
        with patch("trainml.jobs.Job.refresh", new_callable=AsyncMock):
            with raises(SpecificationError, match="Job has no local model or data to upload"):
                await job.connect()

    @mark.asyncio
    async def test_job_connect_uploading_status_single_worker(self, mock_trainml, tmp_path):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "uploading",
                "data": {
                    "output_type": "local",
                    "output_uri": str(tmp_path / "output"),
                },
                "workers": [
                    {
                        "job_worker_uuid": "worker-1",
                        "status": "uploading",
                        "output_auth_token": "worker-token",
                        "output_hostname": "worker-host.com",
                    }
                ],
            },
        )
        
        # Mock refresh to preserve job state and control loop behavior
        refresh_call_count = [0]
        async def mock_refresh():
            refresh_call_count[0] += 1
            # First refresh (before loop, line 346) - ensure state is correct
            if refresh_call_count[0] == 1:
                job._status = "uploading"
                # Ensure workers list exists and has the uploading worker
                if not job._job.get("workers") or len(job._job["workers"]) == 0:
                    job._job["workers"] = [{
                        "job_worker_uuid": "worker-1",
                        "status": "uploading",
                        "output_auth_token": "worker-token",
                        "output_hostname": "worker-host.com",
                    }]
                else:
                    job._job["workers"][0]["status"] = "uploading"
                    job._job["workers"][0]["output_auth_token"] = "worker-token"
                    job._job["workers"][0]["output_hostname"] = "worker-host.com"
                # Also update _workers property
                job._workers = job._job.get("workers")
            # Second refresh (in loop, line 418, first iteration) - keep uploading so download task is created
            elif refresh_call_count[0] == 2:
                job._status = "uploading"
                job._job["workers"][0]["status"] = "uploading"
                job._job["workers"][0]["output_auth_token"] = "worker-token"
                job._job["workers"][0]["output_hostname"] = "worker-host.com"
                job._workers = job._job.get("workers")
            # Third refresh (in loop, second iteration) - mark as finished to exit
            elif refresh_call_count[0] == 3:
                job._status = "finished"
                job._job["workers"][0]["status"] = "finished"
                job._workers = job._job.get("workers")
            return job
        
        with patch.object(job, 'refresh', side_effect=mock_refresh):
            with patch("trainml.jobs.download", new_callable=AsyncMock) as mock_download:
                # Mock sleep - allow loop to continue
                async def sleep_side_effect(delay):
                    # After sleep, next refresh will mark as finished
                    pass
                
                with patch("asyncio.sleep", side_effect=sleep_side_effect):
                    await job.connect()
                    # Download should be called once for the uploading worker
                    # The download task is created in the first loop iteration, then we wait for it
                    assert mock_download.call_count == 1
                    mock_download.assert_called_with("worker-host.com", "worker-token", str(tmp_path / "output"))

    @mark.asyncio
    async def test_job_connect_running_status_multi_worker_polling(self, mock_trainml, tmp_path):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "running",
                "data": {
                    "output_type": "local",
                    "output_uri": str(tmp_path / "output"),
                },
                "workers": [
                    {
                        "job_worker_uuid": "worker-1",
                        "status": "running",
                    },
                    {
                        "job_worker_uuid": "worker-2",
                        "status": "running",
                    },
                ],
            },
        )
        
        refresh_count = [0]
        with patch("trainml.jobs.Job.refresh", new_callable=AsyncMock) as mock_refresh:
            def refresh_side_effect():
                refresh_count[0] += 1
                if refresh_count[0] == 1:
                    # First refresh: worker-1 becomes uploading
                    job._job["workers"][0]["status"] = "uploading"
                    job._job["workers"][0]["output_auth_token"] = "token-1"
                    job._job["workers"][0]["output_hostname"] = "host-1.com"
                elif refresh_count[0] == 2:
                    # Second refresh: worker-2 becomes uploading
                    job._job["workers"][1]["status"] = "uploading"
                    job._job["workers"][1]["output_auth_token"] = "token-2"
                    job._job["workers"][1]["output_hostname"] = "host-2.com"
                else:
                    # Third refresh: both finished
                    job._status = "finished"
                    job._job["workers"][0]["status"] = "finished"
                    job._job["workers"][1]["status"] = "finished"
            mock_refresh.side_effect = refresh_side_effect
            
            with patch("trainml.jobs.download", new_callable=AsyncMock) as mock_download:
                sleep_mock = AsyncMock()
                with patch("asyncio.sleep", sleep_mock):
                    await job.connect()
                    # Should have called download twice (once per worker)
                    assert mock_download.call_count == 2
                    # Should have slept between polls (at least once before both workers finish)
                    assert sleep_mock.call_count >= 1
                    # Verify both downloads were called with correct parameters
                    calls = mock_download.call_args_list
                    assert any(call[0] == ("host-1.com", "token-1", str(tmp_path / "output")) for call in calls)
                    assert any(call[0] == ("host-2.com", "token-2", str(tmp_path / "output")) for call in calls)

    @mark.asyncio
    async def test_job_connect_invalid_status(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "finished",
            },
        )
        
        with raises(SpecificationError, match="You can only connect to active jobs"):
            await job.connect()

    @mark.asyncio
    async def test_job_connect_uploading_no_local_output_error(self, mock_trainml):
        job = specimen.Job(
            mock_trainml,
            **{
                "customer_uuid": "cus-id-1",
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test job",
                "type": "training",
                "status": "uploading",
                "data": {
                    "output_type": "s3",
                },
            },
        )
        
        with patch("trainml.jobs.Job.refresh", new_callable=AsyncMock):
            with raises(SpecificationError, match="Job output_type is not 'local'"):
                await job.connect()

    @mark.asyncio
    async def test_job_remove(self, job, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await job.remove()
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1",
            "DELETE",
            dict(project_uuid="proj-id-1", force=False),
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
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1", "GET", dict(project_uuid="proj-id-1")
        )
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
            "gpu_type": None,
            "gpu_types": ["1060-id"],
            "gpu_count": 1,
            "cpu_count": None,
            "disk_size": 10,
            "max_price": 10,
            "worker_commands": None,
            "workers": None,
            "environment": None,
            "data": None,
            "model": None,
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
        mock_trainml._query.assert_called_once_with(
            "/job/job-id-1", "GET", dict(project_uuid="proj-id-1")
        )
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
