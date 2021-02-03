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

> Creating resources on the trainML platform requires a non-zero credit balance. To purchase credits or sign-up for automatic credit top-ups, visit the [billing page](https://app.trainml.ai/account/payments).

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
