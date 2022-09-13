from trainml.trainml import TrainML
import asyncio


trainml_client = TrainML()

# Create the dataset
dataset = asyncio.run(
    trainml_client.datasets.create(
        name="Example Dataset",
        source_type="aws",
        source_uri="s3://trainml-examples/data/cifar10",
    )
)

print(dataset)

# Watch the log output, attach will return when data transfer is complete
asyncio.run(dataset.attach())

# Create the job
job = asyncio.run(
    trainml_client.jobs.create(
        name="Example Training Job",
        type="training",
        gpu_types=["rtx2080ti", "rtx3090"],
        gpu_count=1,
        disk_size=10,
        workers=[
            "python training/image-classification/resnet_cifar.py --epochs 10 --optimizer adam --batch-size 128",
        ],
        data=dict(
            datasets=[dict(id=dataset.id, type="existing")],
            output_uri="s3://trainml-examples/output/resnet_cifar10",
            output_type="aws",
        ),
        model=dict(
            source_type="git",
            source_uri="https://github.com/trainML/examples.git",
        ),
    )
)
print(job)

# Watch the log output, attach will return when the training job stops
asyncio.run(job.attach())
