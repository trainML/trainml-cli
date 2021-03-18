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
        type="headless",
        gpu_type="GTX 1060",
        gpu_count=1,
        disk_size=10,
        workers=[
            "PYTHONPATH=$PYTHONPATH:$TRAINML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$TRAINML_DATA_PATH --model_dir=$TRAINML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=10 --batch_size=1024",
        ],
        data=dict(
            datasets=[dict(id=dataset.id, type="existing")],
            output_uri="s3://trainml-examples/output/resnet_cifar10",
            output_type="aws",
        ),
        model=dict(git_uri="git@github.com:trainML/test-private.git"),
    )
)
print(job)

# Watch the log output, attach will return when the training job stops
asyncio.run(job.attach())
