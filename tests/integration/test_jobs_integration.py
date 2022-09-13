import re
import sys
import tempfile
import os
import asyncio
import aiohttp
from pytest import mark, fixture, raises

pytestmark = [mark.sdk, mark.integration, mark.jobs]


@fixture(scope="class")
async def job(trainml):
    job = await trainml.jobs.create(
        name="CLI Automated Job Lifecycle",
        type="notebook",
        gpu_types=["gtx1060"],
        gpu_count=1,
        disk_size=10,
        data=dict(datasets=[dict(name="CIFAR-10", type="public")]),
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
        job = await job.wait_for("running", 180)
        assert job.status == "running"

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

    async def test_start_job(self, job):
        assert job.status == "stopped"
        await job.start()
        job = await job.wait_for("running", 120)
        assert job.status == "running"

    async def test_copy_job(self, job):
        job_copy = await job.copy("CLI Automated Job Copy")
        assert job_copy.id != job.id
        await job_copy.wait_for("running", 300)
        assert job_copy.status == "running"
        await job_copy.stop()
        await job_copy.wait_for("stopped", 60)
        assert job_copy.status == "stopped"
        await job_copy.remove()

    async def test_convert_job(self, job):
        training_job = await job.copy(
            name="CLI Automated Job Convert",
            type="training",
            workers=["python $TRAINML_MODEL_PATH/tensorflow/main.py"],
            data=dict(
                datasets=[
                    dict(
                        name="CIFAR-10",
                        type="public",
                    )
                ],
                output_uri="s3://trainml-examples/output/resnet_cifar10",
                output_type="aws",
            ),
        )
        assert training_job.id
        training_job = await training_job.wait_for("finished", 300)
        assert training_job.status == "finished"
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


@mark.create
@mark.asyncio
class JobIOTests:
    async def test_job_local_output(self, trainml, capsys):
        temp_dir = tempfile.TemporaryDirectory()
        job = await trainml.jobs.create(
            name="CLI Automated Local Output Test",
            type="training",
            gpu_types=["gtx1060"],
            gpu_count=1,
            disk_size=10,
            workers=["python $TRAINML_MODEL_PATH/tensorflow/main.py"],
            environment=dict(type="TENSORFLOW_PY39_29"),
            data=dict(
                datasets=[
                    dict(
                        name="CIFAR-10",
                        type="public",
                    )
                ],
                output_uri=temp_dir.name,
                output_type="local",
            ),
            model=dict(
                source_type="git",
                source_uri="git@github.com:trainML/environment-tests.git",
            ),
        )
        await job.wait_for("running")
        attach_task = asyncio.create_task(job.attach())
        connect_task = asyncio.create_task(job.connect())
        await asyncio.gather(attach_task, connect_task)
        await job.refresh()
        assert job.status == "finished"
        await job.disconnect()
        await job.remove()
        upload_contents = os.listdir(temp_dir.name)
        result = any(
            "CLI_Automated_Tensorflow_Test_1" in content
            for content in upload_contents
        )
        assert result is not None
        temp_dir.cleanup()

        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Epoch 1/2" in captured.out
        assert "Epoch 2/2" in captured.out
        assert "adding: model.ckpt-0001.data-00000-of-00001" in captured.out
        assert "Send complete" in captured.out

    async def test_job_model_input_and_output(self, trainml, capsys):

        model = await trainml.models.create(
            name="CLI Automated Jobs -  Git Model",
            source_type="git",
            source_uri="git@github.com:trainML/environment-tests.git",
        )
        await model.wait_for("ready", 300)
        assert model.size >= 500000

        job = await trainml.jobs.create(
            "CLI Automated Training With trainML Model Output",
            type="training",
            gpu_types=["gtx1060"],
            gpu_count=1,
            disk_size=10,
            worker_commands=["python $TRAINML_MODEL_PATH/tensorflow/main.py"],
            environment=dict(type="TENSORFLOW_PY39_29"),
            data=dict(
                datasets=[
                    dict(
                        name="CIFAR-10",
                        type="public",
                    )
                ],
                output_type="trainml",
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

        new_model = await trainml.models.get(
            workers[0].get("output_model_uuid")
        )
        assert new_model.id
        await new_model.wait_for("ready")
        await new_model.refresh()
        assert new_model.size > model.size + 1000000
        assert (
            new_model.name
            == "Job - CLI Automated Training With trainML Model Output"
        )
        await new_model.remove()


@mark.create
@mark.asyncio
class JobTypeTests:
    async def test_endpoint(self, trainml):
        job = await trainml.jobs.create(
            "CLI Automated Endpoint",
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
            name="Test Custom Container",
            type="training",
            gpu_types=["gtx1060", "rtx3090", "a100", "v100", "rtx2080ti"],
            gpu_count=1,
            disk_size=10,
            model=dict(
                source_type="git",
                source_uri="git@github.com:trainML/environment-tests.git",
            ),
            environment=dict(
                type="CUSTOM",
                custom_image="tensorflow/tensorflow:2.5.1-gpu",
                packages=dict(
                    pip=[
                        "tensorflow_addons",
                        "matplotlib",
                        "scipy",
                        "tensorflow_hub",
                        "keras_applications",
                        "keras_preprocessing",
                    ]
                ),
            ),
            worker_commands=[
                "python $TRAINML_MODEL_PATH/tensorflow/main.py",
            ],
            data=dict(
                datasets=[
                    dict(
                        name="CIFAR-10",
                        type="public",
                    )
                ],
                output_uri="s3://trainml-examples/output/resnet_cifar10",
                output_type="aws",
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
        assert "adding: model.ckpt-0001.data-00000-of-00001" in captured.out
        assert "s3://trainml-examples/output/resnet_cifar10" in captured.out
        assert "Upload complete" in captured.out