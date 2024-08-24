import click
import os
from trainml.cli import pass_config
from trainml.cli.project import project


@project.group()
@pass_config
def secret(config):
    """trainML project secret commands."""
    pass


@secret.command()
@pass_config
def list(config):
    """List secrets."""
    data = [
        ["NAME", "CREATED BY", "UPDATED AT"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]
    project = config.trainml.run(config.trainml.client.projects.get_current())
    secrets = config.trainml.run(project.secrets.list())

    for secret in secrets:
        data.append(
            [
                secret.name,
                secret.created_by,
                secret.updated_at.isoformat(timespec="seconds"),
            ]
        )

    for row in data:
        click.echo(
            "{: >38.36} {: >30.28} {: >28.26}" "".format(*row),
            file=config.stdout,
        )


@secret.command()
@click.option(
    "--file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Load the secret value from the file at the provided path",
)
@click.argument("name", type=click.STRING)
@pass_config
def put(config, file, name):
    """
    Set a secret value.

    Secret is created with the specified NAME.
    """
    project = config.trainml.run(config.trainml.client.projects.get_current())
    if file:
        with open(os.path.expanduser(file)) as f:
            value = f.read()
    else:
        value = click.prompt("Enter the secret value", type=str, hide_input=True)

    return config.trainml.run(project.secrets.put(name=name, value=value))


@secret.command()
@click.argument("name", type=click.STRING)
@pass_config
def remove(config, name):
    """
    Remove a secret.


    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    return config.trainml.run(project.secret.remove(name))
