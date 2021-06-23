from trainml.trainml import TrainML
import asyncio


trainml_client = TrainML()


async def create_dataset():
    # Create the dataset
    dataset = await trainml_client.datasets.create(
        name="Local Dataset",
        source_type="local",
        source_uri="~/tensorflow-example/data",
    )

    print(dataset)

    # Connect to the dataset and watch the logs
    attach_task = asyncio.create_task(dataset.attach())
    connect_task = asyncio.create_task(dataset.connect())
    await asyncio.gather(attach_task, connect_task)
    await dataset.disconnect()
    return dataset


dataset = asyncio.run(create_dataset())


async def run_job(dataset):
    # Create the job

    job = await trainml_client.jobs.create(
        name="Training Job with Local Output",
        type="training",
        gpu_type="GTX 1060",
        gpu_count=1,
        disk_size=10,
        workers=[
            "PYTHONPATH=$PYTHONPATH:$TRAINML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$TRAINML_DATA_PATH --model_dir=$TRAINML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=10 --batch_size=1024",
        ],
        data=dict(
            datasets=[dict(id=dataset.id, type="existing")],
            output_uri="~/tensorflow-example/output",
            output_type="local",
        ),
        model=dict(source_type="local", source_uri="~/tensorflow-model"),
    )

    print(job)

    # Jobs using Local Model will wait for you to connect in the "waiting for data/model download" state
    await job.wait_for("waiting for data/model download")
    attach_task = asyncio.create_task(job.attach())
    connect_task = asyncio.create_task(job.connect())
    await asyncio.gather(attach_task, connect_task)

    # Cleanup job
    await job.disconnect()
    await job.remove()


asyncio.run(run_job(dataset))

# Cleanup Dataset
asyncio.run(dataset.remove())