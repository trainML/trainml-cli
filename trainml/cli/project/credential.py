import click
import os
import json
import base64
from pathlib import Path
from trainml.cli import pass_config
from trainml.cli.project import project


@project.group()
@pass_config
def credential(config):
    """trainML project credential commands."""
    pass


@credential.command()
@pass_config
def list(config):
    """List credentials."""
    data = [
        ["TYPE", "KEY ID", "UPDATED AT"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]
    project = config.trainml.run(config.trainml.client.projects.get_current())
    credentials = config.trainml.run(project.credentials.list())

    for credential in credentials:
        data.append(
            [
                credential.type,
                credential.key_id,
                credential.updated_at.isoformat(timespec="seconds"),
            ]
        )

    for row in data:
        click.echo(
            "{: >13.11} {: >37.35} {: >28.26}" "".format(*row),
            file=config.stdout,
        )


@credential.command()
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
    Set a credential.

    A credential is uploaded.
    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    tenant = None

    if type in ["aws", "wasabi"]:
        credential_id = click.prompt(
            "Enter the credential ID", type=str, hide_input=False
        )
        secret = click.prompt("Enter the secret credential", type=str, hide_input=True)
    elif type == "azure":
        credential_id = click.prompt(
            "Enter the Application (client) ID", type=str, hide_input=False
        )
        tenant = click.prompt(
            "Enter the Directory (tenant) ley", type=str, hide_input=False
        )
        secret = click.prompt("Enter the client secret", type=str, hide_input=True)
    elif type in ["docker", "huggingface"]:
        credential_id = click.prompt("Enter the username", type=str, hide_input=False)
        secret = click.prompt("Enter the access token", type=str, hide_input=True)
    elif type in ["gcp", "kaggle"]:
        file_name = click.prompt(
            "Enter the path of the credentials file",
            type=click.Path(
                exists=True, file_okay=True, dir_okay=False, resolve_path=True
            ),
            hide_input=False,
        )
        credential_id = os.path.basename(file_name)
        with open(file_name) as f:
            secret = json.load(f)
        secret = json.dumps(secret)
    elif type == "ngc":
        credential_id = "$oauthtoken"
        secret = click.prompt("Enter the access token", type=str, hide_input=True)
    else:
        raise click.UsageError("Unsupported credential type")

    return config.trainml.run(
        project.credentials.put(
            type=type, credential_id=credential_id, secret=secret, tenant=tenant
        )
    )


@credential.command()
@click.argument("name", type=click.STRING)
@pass_config
def remove(config, name):
    """
    Remove a credential.


    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    return config.trainml.run(project.credential.remove(name))
