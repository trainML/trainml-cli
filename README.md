<div align="center">
  <a href="https://www.trainml.ai/"><img src="https://www.trainml.ai/static/img/trainML-logo-purple.png"></a><br>
</div>

# trainML Python SDK and Command Line Tools

Provides programmatic access to [trainML platform](https://app.trainml.ai).

## Installation

Python 3.8 or above is required.

```
pip install trainml
```

## Authentication

### Prerequisites

You must have a valid [trainML account](https://app.trainml.ai). On the [account settings page](https://app.trainml.ai/account/settings) click the `Create` button in the `API Keys` section. This will automatically download a `credentials.json` file. This file can only be generated once per API key. Treat this file as a password, as anyone with access to your API key will have the ability to create and control resources in your trainML account. You can deactivate any API key by clicking the `Remove` button.

> Creating resources on the trainML platform requires a non-zero credit balance. To purchase credits or sign-up for automatic credit top-ups, visit the [billing page](https://app.trainml.ai/account/billing).

### Methods

#### Credentials File

The easiest way to authenticate is to place the credentials file downloaded into the `.trainml` folder of your home directory and ensure only you have access to it. From the directory that the `credentials.json` file was downloaded, run the following command:

```
mkdir -p ~/.trainml
mv credentials.json ~/.trainml/credentials.json
chmod 600 ~/.trainml/credentials.json
```

#### Environment Variables

You can also use environment variables `TRAINML_USER` and `TRAINML_KEY` and set them to their respective values from the `credentials.json` file.

```
export TRAINML_USER=<'user' field from credentials.json>
export TRAINML_KEY=<'key' field from credentials.json>
python create_job.py
```

Environment variables will override any credentials stored in `~/.trainml/credentials.json`

#### Runtime Variables

API credentials can also be passed directly to the TrainML object constructor at runtime.

```
import trainml
trainml = trainml.TrainML(user="user field from credentials.json",key="key field from credentials.json>")
await trainml.jobs.create(...)
```

Passing credentials to the TrainML constructor will override all other methods for setting credentials.

## Usage

### Python SDK

The trainML SDK utilizes the [asyncio library](https://docs.python.org/3/library/asyncio.html) to ease the concurrent execution of long running tasks. An example of how to create a dataset from an S3 bucket and immediately run a training job on that dataset is the following:

```
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

# Cleanup job and dataset
asyncio.run(job.remove())
asyncio.run(dataset.remove())
```

See more examples in the [examples folder](examples)

### Command Line Interface

The command line interface is rooted in the `trainml` command. To see the available options, run:

```
trainml --help
```

To list all jobs:

```
trainml job list
```

To list all datasets:

```
trainml dataset list
```

To connect to a job that requires the [connection capability](https://docs.trainml.ai/reference/connection-capability):

```
trainml job connect <job ID or name>
```

To watch the realtime job logs:

```
trainml job attach <job ID or name>
```

To create and open a notebook job:

```
trainml job create notebook "My Notebook Job"
```

To create a multi-GPU notebook job on a specific GPU type with larger scratch directory space:

```
trainml job create notebook --gpu-type "RTX 3090" --gpu-count 4 --disk-size 50 "My Notebook Job"
```

To run the model training code in the `train.py` file in your local `~/model-code` directory on the training data in your local `~/data` directory:

```
trainml job create training --model-dir ~/model-code --data-dir ~/data "My Training Job" "python train.py"
```

Stop a job by job ID:

```
trainml job stop fe52527c-1f4b-468f-b57d-86db864cc089
```

Stop a job by name:

```
trainml job stop "My Notebook Job"
```

Restart a notebook job:

```
trainml job start "My Notebook Job"
```

Remove a job by job ID:

```
trainml job remove fe52527c-1f4b-468f-b57d-86db864cc089
```
