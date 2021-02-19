import re
import sys
import logging
import tempfile
import os
from pytest import mark, fixture, raises

pytestmark = [mark.integration, mark.jobs]


@fixture(scope="class")
async def job(trainml):
    job = await trainml.jobs.create(
        name="CLI Automated Job Lifecycle",
        type="interactive",
        gpu_type="GTX 1060",
        gpu_count=1,
        disk_size=1,
        data=dict(datasets=[dict(name="CIFAR-10", type="public")]),
        model=dict(git_uri="git@github.com:trainML/test-private.git"),
    )
    yield job


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
        assert isinstance(job.provider, str)
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
        job = await job.wait_for("running", 60)
        assert job.status == "running"

    async def test_convert_job(self, job):
        training_job = await job.copy(
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

    async def test_remove_job(self, job):
        assert job.status == "running"
        await job.stop()
        await job.wait_for("stopped", 60)
        await job.remove()
        job = await job.wait_for("archived", 60)
        assert job is None


@mark.create
@mark.asyncio
class JobEnvironmentTests:
    async def test_job_tensorflow(self, trainml, capsys):
        job = await trainml.jobs.create(
            "CLI Automated Tensorflow Test",
            type="headless",
            gpu_type="GTX 1060",
            gpu_count=1,
            disk_size=1,
            worker_count=1,
            worker_commands=[
                "PYTHONPATH=$PYTHONPATH:$TRAINML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$TRAINML_DATA_PATH --model_dir=$TRAINML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=2 --batch_size=1024"
            ],
            environment=dict(type="TENSORFLOW_PY38_24"),
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
            model=dict(git_uri="git@github.com:trainML/test-private.git"),
        )
        await job.attach()
        await job.refresh()
        assert job.status == "stopped"
        await job.remove()
        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Epoch 1/2" in captured.out
        assert "Epoch 2/2" in captured.out
        assert "adding: model.ckpt-0001.data-00000-of-00001" in captured.out
        assert "s3://trainml-examples/output/resnet_cifar10" in captured.out
        assert "Upload complete" in captured.out

    async def test_job_pytorch(self, trainml, capsys):
        job = await trainml.jobs.create(
            "CLI Automated PyTorch Test",
            type="headless",
            gpu_type="GTX 1060",
            gpu_count=1,
            disk_size=1,
            worker_count=1,
            worker_commands=[
                "cd $TRAINML_OUTPUT_PATH && python $TRAINML_MODEL_PATH/mnist/main.py  --epochs 1  2>&1 | tee $TRAINML_OUTPUT_PATH/train.log"
            ],
            environment=dict(type="PYTORCH_PY38_17"),
            data=dict(
                datasets=[],
                output_uri="gs://trainml-example/output/mnist",
                output_type="gcp",
            ),
            model=dict(git_uri="https://github.com/pytorch/examples"),
        )
        await job.attach()
        await job.refresh()
        assert job.status == "stopped"
        await job.remove()
        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "Train Epoch: 1 [0/60000 (0%)]" in captured.out
        assert "Train Epoch: 1 [59520/60000 (99%)]" in captured.out
        assert "adding: train.log" in captured.out
        assert "gs://trainml-example/output/mnist" in captured.out
        assert "Operation completed over 1 objects" in captured.out

    async def test_job_mxnet(self, trainml, capsys):
        temp_dir = tempfile.TemporaryDirectory()
        job = await trainml.jobs.create(
            "CLI Automated MXNet Test",
            type="headless",
            gpu_type="GTX 1060",
            gpu_count=1,
            disk_size=1,
            worker_count=1,
            worker_commands=[
                "python scripts/classification/cifar/train_cifar10.py --model cifar_resnet56_v2 --num-gpus 1 --save-dir $TRAINML_OUTPUT_PATH --num-epochs 2 --batch-size 1024"
            ],
            environment=dict(
                type="MXNET_PY38_17",
                env=[dict(key="MXNET_CUDNN_AUTOTUNE_DEFAULT", value="0")],
            ),
            data=dict(
                datasets=[],
                output_uri=temp_dir.name,
                output_type="local",
            ),
            model=dict(git_uri="https://github.com/dmlc/gluon-cv.git"),
        )
        await job.wait_for("running")
        await job.connect()
        await job.attach()
        await job.refresh()
        assert job.status == "stopped"
        await job.disconnect()
        await job.remove()
        upload_contents = os.listdir(temp_dir.name)
        result = any(
            "CLI_Automated_MXNet_Test_1" in content
            for content in upload_contents
        )
        assert result is not None
        temp_dir.cleanup()
        captured = capsys.readouterr()
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        assert "INFO:root:[Epoch 0]" in captured.out
        assert "INFO:root:[Epoch 1]" in captured.out
        assert "adding: cifar10-cifar_resnet56_v2" in captured.out
        assert "Starting send CLI_Automated_MXNet_Test_1" in captured.out
        assert "Send complete" in captured.out
