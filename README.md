<div align="center">
  <a href="https://www.trainml.ai/"><img src="https://www.trainml.ai/static/img/trainML-logo-purple.png"></a><br>
</div>

trainML CLI
=========================
Provides programmatic access to trainML platform.

## Prerequisites
You must first generate an API key for your trainML account from [account settings page](https://app.trainml.ai/account/settings).  Once you have downloaded the credentials file, move it to the `~/.trainml` directory and ensure only you have access to it:

```
mkdir -p ~/.trainml
mv credentials.json ~/.trainml/credentials.json
chmod 600 ~/.trainml/credentials.json
```

Install the project requirements in the python environment you intend to use:

```
pip install -r requirements.txt
```
