import re
import logging
from pytest import mark, fixture, raises

import trainml.jobs as specimen

pytestmark = [mark.integration, mark.jobs]


@fixture(scope="module")
async def notebook_job(trainml):
    job = await trainml.jobs.create(
        name="CLI Automated Test Notebook For Coverting",
        type="interactive",
        gpu_type="GTX 1060",
        gpu_count=1,
        disk_size=1,
        data=dict(datasets=[dict(name="CIFAR-10", type="public")]),
        model=dict(git_uri="git@github.com:trainML/test-private.git"),
    )
    job = await job.wait_for("running")
    logging.info(job)
    yield job
    await job.stop()
    await job.wait_for("stopped")
    await job.remove()


@fixture(scope="class")
async def job(trainml):
    job = await trainml.jobs.create(
        name="CLI Automated Test Empty Notebook",
        type="interactive",
        gpu_type="GTX 1060",
        gpu_count=1,
        disk_size=1,
    )
    logging.info(job)
    yield job


@mark.create
@mark.asyncio
class GetJobTests:
    async def test_get_jobs(self, trainml, notebook_job):
        jobs = await trainml.jobs.list()
        assert len(jobs) > 0

    async def test_get_job(self, trainml, notebook_job):
        response = await trainml.jobs.get(notebook_job.id)
        assert response.id == notebook_job.id

    async def test_job_properties(self, notebook_job):
        assert isinstance(notebook_job.id, str)
        assert isinstance(notebook_job.name, str)
        assert isinstance(notebook_job.status, str)
        assert isinstance(notebook_job.provider, str)
        assert isinstance(notebook_job.type, str)

    def test_job_str(self, notebook_job):
        string = str(notebook_job)
        regex = r"^{.*\"job_uuid\": \"" + notebook_job.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_job_repr(self, notebook_job):
        string = repr(notebook_job)
        regex = (
            r"^Job\( trainml , {.*'job_uuid': '" + notebook_job.id + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)


@mark.create
@mark.asyncio
class JobConvertTests:
    async def test_convert_job(self, notebook_job):
        training_job = await notebook_job.copy(
            "CLI Automated Job Convert",
            type="headless",
            worker_count=1,
            worker_commands=[
                "PYTHONPATH=$PYTHONPATH:$TRAINML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$TRAINML_DATA_PATH --model_dir=$TRAINML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=1 --batch_size=1024"
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
        assert training_job.id
        training_job = await training_job.wait_for("stopped", 180)
        assert training_job.status == "stopped"
        await training_job.remove()


@mark.create
@mark.asyncio
class JobLifeCycleTests:
    async def test_wait_for_running(self, job):
        assert job.status != "running"
        job = await job.wait_for("running", 120)
        assert job.status == "running"

    async def test_stop_job(self, job):
        assert job.status == "running"
        await job.stop()
        job = await job.wait_for("stopped", 60)
        assert job.status == "stopped"

    async def test_start_job(self, job):
        assert job.status == "stopped"
        await job.start()
        job = await job.wait_for("running", 60)
        assert job.status == "running"

    async def test_remove_job(self, job):
        assert job.status == "running"
        await job.stop()
        await job.wait_for("stopped", 60)
        await job.remove()
        job = await job.wait_for("archived", 60)
        assert job is None
