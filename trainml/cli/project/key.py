import click
import os
import json
import base64
from pathlib import Path
from trainml.cli import pass_config
from trainml.cli.project import project


@project.group()
@pass_config
def key(config):
    """trainML project key commands."""
    pass


@key.command()
@pass_config
def list(config):
    """List keys."""
    data = [
        ["TYPE", "KEY ID", "UPDATED AT"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]
    project = config.trainml.run(config.trainml.client.projects.get_current())
    keys = config.trainml.run(project.keys.list())

    for key in keys:
        data.append(
            [
                key.type,
                key.key_id,
                key.updated_at.isoformat(timespec="seconds"),
            ]
        )

    for row in data:
        click.echo(
            "{: >13.11} {: >37.35} {: >28.26}" "".format(*row),
            file=config.stdout,
        )


@key.command()
@click.argument(
    "type",
    type=click.Choice(
        [
            "aws",
            "azure",
            "docker",
            "gcp",
            "huggingface",
            "kaggle",
            "ngc",
            "wasabi",
        ],
        case_sensitive=False,
    ),
)
@pass_config
def put(config, type):
    """
    Set a key.

    A key is uploaded.
    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    tenant = None

    if type in ["aws", "wasabi"]:
        key_id = click.prompt("Enter the key ID", type=str, hide_input=False)
        secret = click.prompt("Enter the secret key", type=str, hide_input=True)
    elif type == "azure":
        key_id = click.prompt(
            "Enter the Application (client) ID", type=str, hide_input=False
        )
        tenant = click.prompt(
            "Enter the Directory (tenant) ley", type=str, hide_input=False
        )
        secret = click.prompt("Enter the client secret", type=str, hide_input=True)
    elif type in ["docker", "huggingface"]:
        key_id = click.prompt("Enter the username", type=str, hide_input=False)
        secret = click.prompt("Enter the access token", type=str, hide_input=True)
    elif type in ["gcp", "kaggle"]:
        file_name = click.prompt(
            "Enter the path of the credentials file",
            type=click.Path(
                exists=True, file_okay=True, dir_okay=False, resolve_path=True
            ),
            hide_input=False,
        )
        key_id = os.path.basename(file_name)
        with open(file_name) as f:
            secret = json.load(f)
        secret = json.dumps(secret)
    elif type == "ngc":
        key_id = "$oauthtoken"
        secret = click.prompt("Enter the access token", type=str, hide_input=True)
    else:
        raise click.UsageError("Unsupported key type")

    return config.trainml.run(
        project.keys.put(type=type, key_id=key_id, secret=secret, tenant=tenant)
    )


@key.command()
@click.argument("name", type=click.STRING)
@pass_config
def remove(config, name):
    """
    Remove a key.


    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    return config.trainml.run(project.key.remove(name))
