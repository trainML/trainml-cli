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
        gpu_types=["rtx2080ti", "rtx3090"],
        gpu_count=1,
        disk_size=10,
        workers=[
            "python training/image-classification/resnet_cifar.py --epochs 10 --optimizer adam --batch-size 128",
        ],
        data=dict(
            datasets=[dict(id=dataset.id, type="existing")],
            output_uri="~/tensorflow-example/output",
            output_type="local",
        ),
        model=dict(source_type="local", source_uri="~/trainml-examples"),
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