import re
import sys
import tempfile
import os
import asyncio
import aiohttp
from pytest import mark, fixture, raises
from trainml.exceptions import ApiError
from urllib.parse import urlparse

pytestmark = [mark.sdk, mark.integration, mark.jobs]


def extract_domain_suffix(hostname):
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    else:
        return None


@fixture(scope="class")
async def job(trainml):
    job = await trainml.jobs.create(
        name="CLI Automated Tests - Job Lifecycle",
        type="notebook",
        gpu_types=["gtx1060"],
        gpu_count=1,
        disk_size=11,
        data=dict(datasets=[dict(id="CIFAR-10", public=True)]),
        model=dict(
            source_type="git",
            source_uri="git@github.com:trainML/environment-tests.git",
        ),
    )
    yield job


@mark.create
@mark.asyncio
class JobLifeCycleTests:
    async def test_wait_for_running(self, job):
        assert job.status != "running"
        job = await job.wait_for("running")
        assert job.status == "running"
        assert job.url
        assert extract_domain_suffix(urlparse(job.url).hostname) == "proximl.cloud"

    async def test_stop_job(self, job):
        assert job.status == "running"
        await job.stop()
        job = await job.wait_for("stopped", 120)
        assert job.status == "stopped"

    async def test_get_jobs(self, trainml, job):
        jobs = await trainml.jobs.list()
        assert len(jobs) > 0

    async def test_get_job(self, trainml, job):
        response = await trainml.jobs.get(job.id)
        assert response.id == job.id

    async def test_job_properties(self, job):
        assert isinstance(job.id, str)
        assert isinstance(job.name, str)
        assert isinstance(job.status, str)
        assert isinstance(job.type, str)

    async def test_job_str(self, job):
        string = str(job)
        regex = r"^{.*\"job_uuid\": \"" + job.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_job_repr(self, job):
        string = repr(job)
        regex = r"^Job\( trainml , {.*'job_uuid': '" + job.id + r"'.*}\)$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_start_job(self, job):
        assert job.status == "stopped"
        await job.start()
        job = await job.wait_for("running", 120)
        assert job.status == "running"

    async def test_copy_job_not_enough_disk(self, job):
        with raises(ApiError) as error:
            await job.copy(
                "CLI Automated Tests - Job Copy Not Enough Disk", disk_size=10
            )

        assert (
            "Invalid Request - Copied job disk_size must be greater than or equal to source job disk_size"
            in error.value.message
        )

    async def test_copy_job(self, job):
        await job.wait_for("running", 300)
        job_copy = await job.copy("CLI Automated Tests - Job Copy")
        assert job_copy.id != job.id
        await job_copy.wait_for("running", 300)
        assert job_copy.status == "running"
        await job_copy.stop()
        await job_copy.wait_for("stopped", 60)
        assert job_copy.status == "stopped"
        await job_copy.remove()

    async def test_convert_job(self, job):
        await job.wait_for("running", 300)
        training_job = await job.copy(
            name="CLI Automated Tests - Job Convert",
            type="training",
            workers=["python $ML_MODEL_PATH/tensorflow/main.py"],
            data=dict(
                datasets=[
                    dict(
                        id="CIFAR-10",
                        public=True,
                    )
                ],
                output_uri="s3://trainml-examples/output/resnet_cifar10",
                output_type="aws",
            ),
        )
        assert training_job.id
        training_job = await training_job.wait_for("finished", 300)
        assert training_job.status == "finished"
        await asyncio.sleep(10)  ## give billing a chance to update
        await training_job.refresh()
        assert training_job.credits > 0
        assert training_job.credits < 0.1
        await training_job.remove()

    async def test_remove_job(self, job):
        assert job.status == "running"
        await job.stop()
        await job.wait_for("stopped", 90)
        await job.remove()
        job = await job.wait_for("archived", 60)
        assert job is None


@mark.asyncio
class JobAPIResourceValidationTests:
    async def test_invalid_gpu_type(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid GPU Type",
                type="training",
                gpu_types=["k80"],
                disk_size=10,
            )
        assert "Invalid Request - GPU Type k80 Invalid" in error.value.message

    async def test_invalid_disk_size(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid Disk Size",
                type="training",
                gpu_types=["rtx3090"],
                disk_size=1,
            )
        assert (
            "Invalid Request - Disk Size must be between 10 and 2000"
            in error.value.message
        )

    async def test_combine_cpu_and_gpu_types(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Combine CPU and GPU Types",
                type="training",
                gpu_types=["cpu", "rtx3090"],
                disk_size=10,
            )
        assert (
            "Invalid Request - None (CPU Only) may be not be combined with other GPU Types"
            in error.value.message
        )

    async def test_missing_cpu_count(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Missing CPU Count",
                type="training",
                gpu_types=["cpu"],
                disk_size=10,
            )
        assert (
            "Invalid Request - cpu_count required for CPU only jobs"
            in error.value.message
        )

    async def test_invalid_cpu_count(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid CPU Count",
                type="training",
                gpu_types=["cpu"],
                cpu_count=1,
                disk_size=10,
            )
        assert (
            "Invalid Request - CPU Count must be a multiple of 4" in error.value.message
        )

    async def test_invalid_gpu_count_for_cpu(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid GPU Count For CPU",
                type="training",
                gpu_types=["cpu"],
                gpu_count=1,
                cpu_count=4,
                disk_size=10,
            )
        assert (
            "Invalid Request - gpu_count not valid for CPU only jobs"
            in error.value.message
        )

    async def test_invalid_cpu_count_for_gpu(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid CPU Count for GPU",
                type="notebook",
                gpu_types=["rtx3090"],
                cpu_count=4,
                disk_size=10,
            )
        assert (
            "Invalid Request - CPU Count must be at least 8 for gpu types rtx3090"
            in error.value.message
        )

    async def test_invalid_cpu_count_for_gpu_2(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid CPU Count for GPU 2",
                type="notebook",
                gpu_types=["rtx2080ti"],
                gpu_count=2,
                cpu_count=4,
                disk_size=10,
            )
        assert (
            "Invalid Request - CPU Count must be at least 8 for gpu types rtx2080ti and gpu_count 2"
            in error.value.message
        )

    async def test_invalid_cpu_count_for_gpu_3(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid CPU Count for GPU 3",
                type="training",
                gpu_types=["rtx2080ti", "rtx3090"],
                gpu_count=2,
                cpu_count=8,
                disk_size=10,
            )
        assert (
            "Invalid Request - CPU Count must be at least 16 for gpu types rtx2080ti,rtx3090 and gpu_count 2"
            in error.value.message
        )


@mark.asyncio
class JobAPIDataValidationTests:
    async def test_invalid_output_type_for_notebook(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid Output Type for Notebook",
                type="notebook",
                gpu_types=["rtx3090"],
                disk_size=10,
                data=dict(
                    output_uri="s3://trainml-examples/output/resnet_cifar10",
                    output_type="aws",
                ),
            )
        assert (
            "Invalid Request - output_type invalid for Notebook and Endpoint jobs"
            in error.value.message
        )

    async def test_invalid_output_type_for_endpoint(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid Output Type for Endpoint",
                type="endpoint",
                gpu_types=["rtx3090"],
                disk_size=10,
                endpoint=dict(
                    routes=[
                        dict(
                            path="/predict",
                            verb="POST",
                            function="predict_image",
                            file="predict",
                            positional=True,
                            body=[dict(name="filename", type="str")],
                        )
                    ]
                ),
                data=dict(
                    output_uri="s3://trainml-examples/output/resnet_cifar10",
                    output_type="aws",
                ),
            )
        assert (
            "Invalid Request - output_type invalid for Notebook and Endpoint jobs"
            in error.value.message
        )

    async def test_invalid_volumes_for_training(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid Volumes for Training",
                type="training",
                gpu_types=["rtx3090"],
                disk_size=10,
                data=dict(
                    output_uri="s3://trainml-examples/output/resnet_cifar10",
                    output_type="aws",
                    volumes=["volume-id"],
                ),
                workers=["python train.py"],
            )
        assert (
            "Invalid Request - Only Notebook and Endpoint job types can use writable volumes"
            in error.value.message
        )

    async def test_invalid_volumes_for_inference(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid Volumes for Inference",
                type="inference",
                gpu_types=["rtx3090"],
                disk_size=10,
                data=dict(
                    output_uri="s3://trainml-examples/output/resnet_cifar10",
                    output_type="aws",
                    volumes=["volume-id"],
                ),
                workers=["python predict.py"],
            )
        assert (
            "Invalid Request - Only Notebook and Endpoint job types can use writable volumes"
            in error.value.message
        )

    async def test_invalid_datasets_for_inference(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Datasets for Inference Job",
                type="inference",
                gpu_types=["rtx3090"],
                disk_size=10,
                data=dict(datasets=[dict(id="CIFAR-10", public=True)]),
                workers=["python predict.py"],
            )
        assert (
            "Invalid Request - Inference jobs cannot use datasets"
            in error.value.message
        )


@mark.asyncio
class JobAPIWorkerValidationTests:
    async def test_missing_workers_for_training(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Missing Workers for Training Job",
                type="training",
                gpu_types=["rtx3090"],
                disk_size=10,
            )
        assert (
            "Invalid Request - Training jobs must have at least one worker"
            in error.value.message
        )

    async def test_too_many_workers_for_inference(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Too Many Workers for Inference Job",
                type="inference",
                gpu_types=["rtx3090"],
                disk_size=10,
                workers=["python predict.py", "python predict.py"],
            )
        assert (
            "Invalid Request - Inference jobs must have exactly one worker"
            in error.value.message
        )

    async def test_invalid_worker_spec_for_endpoint(self, trainml):
        with raises(ApiError) as error:
            await trainml.jobs.create(
                name="Invalid Worker Spec for Endpoint",
                type="endpoint",
                gpu_types=["rtx3090"],
                disk_size=10,
                workers=["python predict.py"],
            )
        assert (
            "Invalid Request - Endpoints do not use worker commands"
            in error.value.message
        )


@mark.create
@mark.asyncio
class JobIOTests:
    async def test_job_local_output(self, trainml, capsys):
        temp_dir = tempfile.TemporaryDirectory()
        job = await trainml.jobs.create(
            name="CLI Automated Tests - Local Output",
            type="training",
            gpu_types=["gtx1060"],
            disk_size=10,
            workers=["python $ML_MODEL_PATH/tensorflow/main.py"],
            environment=dict(
                type="DEEPLEARNING_PY310",
                env=[
                    dict(
                        key="CHECKPOINT_FILE",
                        value="model.ckpt-0050",
                    )
                ],
            ),
            data=dict(
                datasets=[
                    dict(
                        id="CIFAR-10",
                        public=True,
                    )
                ],
                output_uri=temp_dir.name,
                output_type="local",
            ),
            model=dict(
                source_type="git",
                source_uri="git@github.com:trainML/environment-tests.git",
                checkpoints=[
                    "tensorflow-checkpoint",
                ],
            ),
        )
        await job.wait_for("waiting for data/model download")
        attach_task = asyncio.create_task(job.attach())
        connect_task = asyncio.create_task(job.connect())
        await asyncio.gather(attach_task, connect_task)
        await job.refresh()
        assert job.status == "finished"
        await job.disconnect()
        await job.remove()
        upload_contents = os.listdir(temp_dir.name)
        temp_dir.cleanup()
        assert any(
            "CLI_Automated_Tests_-_Local_Output" in content
            for content in upload_contents
        )

        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Epoch 1/2" in captured.out
        assert "Epoch 2/2" in captured.out
        assert "adding: model.ckpt-0001.data-00000-of-00001" in captured.out
        assert "Send complete" in captured.out

    async def test_job_model_input_and_output(self, trainml, capsys):
        model = await trainml.models.create(
            name="CLI Automated Tests - Job Git Model",
            source_type="git",
            source_uri="git@github.com:trainML/environment-tests.git",
        )
        await model.wait_for("ready", 300)
        assert model.size >= 50000

        job = await trainml.jobs.create(
            "CLI Automated Tests - Training With trainML Model Output",
            type="training",
            gpu_types=["gtx1060"],
            gpu_count=1,
            cpu_count=8,
            disk_size=10,
            worker_commands=["python $ML_MODEL_PATH/tensorflow/main.py"],
            data=dict(
                datasets=[
                    dict(
                        id="CIFAR-10",
                        public=True,
                    )
                ],
                output_type="trainml",
                output_uri="model",
            ),
            model=dict(source_type="trainml", source_uri=model.id),
        )
        await job.attach()
        await job.refresh()
        assert job.status == "finished"
        workers = job.workers
        await job.remove()
        await model.remove()
        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Epoch 1/2" in captured.out
        assert "Epoch 2/2" in captured.out

        new_model = await trainml.models.get(workers[0].get("output_uuid"))
        assert new_model.id
        await new_model.wait_for("ready")
        await new_model.refresh()
        assert new_model.size > model.size + 1000000
        assert (
            new_model.name
            == "Job - CLI Automated Tests - Training With trainML Model Output"
        )
        await new_model.remove()


@mark.create
@mark.asyncio
class JobTypeTests:
    async def test_endpoint(self, trainml):

        job = await trainml.jobs.create(
            "CLI Automated Tests - Endpoint",
            type="endpoint",
            gpu_type="GTX 1060",
            gpu_count=1,
            disk_size=10,
            model=dict(
                source_type="git",
                source_uri="https://github.com/trainML/simple-tensorflow-classifier.git",
            ),
            endpoint=dict(
                routes=[
                    dict(
                        path="/predict",
                        verb="POST",
                        function="predict_image",
                        file="predict",
                        positional=True,
                        body=[dict(name="filename", type="str")],
                    )
                ]
            ),
        )
        await job.wait_for("running")
        await job.refresh()
        assert job.url
        assert extract_domain_suffix(urlparse(job.url).hostname) == "proximl.cloud"
        tries = 0
        await asyncio.sleep(30)
        async with aiohttp.ClientSession() as session:
            retry = True
            while retry:
                async with session.request(
                    "GET",
                    f"{job.url}/ping",
                ) as resp:
                    if resp.status in [404, 502, 503]:
                        tries += 1
                        print(resp)
                        resp.close()
                        if tries == 5:
                            raise Exception("Too many errors")
                        else:
                            await asyncio.sleep(5)
                    else:
                        retry = False
                        results = await resp.json()
                        assert results["message"] == "pong"

        await job.stop()
        await job.wait_for("stopped")
        await job.refresh()
        assert not job.url
        await job.remove()

    async def test_job_custom_container(self, trainml, capsys):
        job = await trainml.jobs.create(
            name="CLI Automated Tests - Custom Container",
            type="training",
            gpu_types=["gtx1060", "rtx3090", "rtx2080ti"],
            gpu_count=1,
            disk_size=10,
            model=dict(
                source_type="git",
                source_uri="git@github.com:trainML/environment-tests.git",
            ),
            environment=dict(
                type="CUSTOM",
                custom_image="tensorflow/tensorflow:2.13.0-gpu",
                packages=dict(
                    pip=[
                        "matplotlib",
                        "scipy",
                        "tensorflow_hub",
                        "keras_applications",
                        "keras_preprocessing",
                        "protobuf",
                        "typing-extensions",
                    ]
                ),
            ),
            worker_commands=[
                "python $ML_MODEL_PATH/tensorflow/main.py",
            ],
            data=dict(
                datasets=[
                    dict(
                        id="CIFAR-10",
                        public=True,
                    )
                ],
                output_type="wasabi",
                output_uri="s3://trainml-example/output/resnet_cifar10",
                output_options=dict(
                    endpoint_url="https://s3.wasabisys.com", archive=False
                ),
            ),
        )
        assert job.id
        await job.attach()
        await job.refresh()
        assert job.status == "finished"
        await job.remove()
        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Epoch 1/2" in captured.out
        assert "Epoch 2/2" in captured.out
        assert "Uploading s3://trainml-example/output/resnet_cifar10" in captured.out
        assert (
            "upload: ./model.ckpt-0002.data-00000-of-00001 to s3://trainml-example/output/resnet_cifar10/model.ckpt-0002.data-00000-of-00001"
            in captured.out
        )
        assert "Upload complete" in captured.out


@mark.create
@mark.asyncio
class JobFeatureTests:
    async def test_cpu_instance(self, trainml, capsys):
        job = await trainml.jobs.create(
            name="CLI Automated Tests - CPU Instance",
            type="training",
            gpu_types=["cpu"],
            cpu_count=4,
            disk_size=10,
            model=dict(
                source_type="git",
                source_uri="git@github.com:trainML/environment-tests.git",
            ),
            worker_commands=[
                "python $ML_MODEL_PATH/pytorch/main.py",
            ],
            data=dict(
                datasets=[dict(id="MNIST", public=True)],
            ),
        )
        assert job.id
        await job.attach()
        await job.refresh()
        assert job.status == "finished"
        await job.remove()
        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Train Epoch: 1 [0/60000 (0%)]" in captured.out
        assert "Train Epoch: 1 [59520/60000 (99%)]" in captured.out

    async def test_inference_job(self, trainml, capsys):
        temp_dir = tempfile.TemporaryDirectory()
        job = await trainml.jobs.create(
            name="CLI Automated Tests - Inference Job",
            type="inference",
            gpu_type="GTX 1060",
            gpu_count=1,
            disk_size=10,
            workers=[
                "python $ML_MODEL_PATH/tensorflow/main.py",
            ],
            data=dict(
                input_type="wasabi",
                input_uri="s3://trainml-example/input/cifar-10",
                input_options=dict(endpoint_url="https://s3.wasabisys.com"),
                output_type="local",
                output_uri=temp_dir.name,
                output_options=dict(archive=False),
            ),
            model=dict(git_uri="git@github.com:trainML/environment-tests.git"),
        )
        assert job.id
        await job.wait_for("running")
        await job.connect()
        await job.attach()
        await job.refresh()
        assert job.status == "finished"
        await job.disconnect()
        await job.remove()
        await job.wait_for("archived")
        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        upload_contents = os.listdir(temp_dir.name)
        temp_dir.cleanup()
        assert len(upload_contents) > 4
        assert any(
            "model.ckpt-0002.data-00000-of-00001" in content
            for content in upload_contents
        )

        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Epoch 1/2" in captured.out
        assert "Epoch 2/2" in captured.out
        assert "Number of regular files transferred: 7" in captured.out
        assert "Send complete" in captured.out
