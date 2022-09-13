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
training_job = asyncio.run(
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
            output_type="trainml",
        ),
        model=dict(
            source_type="git",
            source_uri="https://github.com/trainML/examples.git",
        ),
    )
)
print(training_job)

# Watch the log output, attach will return when the training job stops
asyncio.run(training_job.attach())

# Get the trained model id from the workers
training_job = asyncio.run(training_job.refresh())

model = asyncio.run(
    trainml_client.models.get(training_job.workers[0].get("output_model_uuid"))
)

# Ensure the model is ready to use
asyncio.run(model.wait_for("ready"))

# Use the model in an inference job on new data
inference_job = asyncio.run(
    trainml_client.jobs.create(
        name="Example Inference Job",
        type="inference",
        gpu_types=["rtx2080ti", "rtx3090"],
        gpu_count=1,
        disk_size=10,
        workers=[
            "python training/image-classification/resnet_cifar.py",
        ],
        data=dict(
            input_type="aws",
            input_uri="s3://trainml-examples/data/new_data",
            output_type="aws",
            output_uri="s3://trainml-examples/output/model_predictions",
        ),
        model=dict(source_type="trainml", source_uri=model.id),
    )
)
print(inference_job)

# Watch the log output, attach will return when the training job stops
asyncio.run(inference_job.attach())

# (Optional) Cleanup
asyncio.gather(
    training_job.remove(),
    inference_job.remove(),
    model.remove(),
    dataset.remove(),
)
