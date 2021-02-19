from trainml.trainml import TrainML
import asyncio


trainml_client = TrainML()

# Create the dataset
dataset = asyncio.run(
    trainml_client.datasets.create(
        name="Local Dataset",
        source_type="local",
        source_uri="~/tensorflow-example/data",
    )
)

print(dataset)

# Connect to the dataset and watch the logs
asyncio.run(dataset.connect())
asyncio.run(dataset.attach())
asyncio.run(dataset.disconnect())

# # Create the job
job = asyncio.run(
    trainml_client.jobs.create(
        name="Training Job with Local Output",
        type="headless",
        gpu_type="GTX 1060",
        gpu_count=1,
        disk_size=10,
        worker_count=1,
        worker_commands=[
            "PYTHONPATH=$PYTHONPATH:$TRAINML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$TRAINML_DATA_PATH --model_dir=$TRAINML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=10 --batch_size=1024",
        ],
        data=dict(
            datasets=[dict(id=dataset.id, type="existing")],
            output_uri="~/tensorflow-example/output",
            output_type="local",
        ),
        model=dict(git_uri="git@github.com:trainML/test-private.git"),
    )
)
print(job)

# # Connect to the job once it's running and attach to watch the logs
asyncio.run(job.wait_for("running"))
asyncio.run(job.connect())
asyncio.run(job.attach())


# ## Cleanup resources
asyncio.run(job.disconnect())
asyncio.run(job.remove())
asyncio.run(dataset.remove())