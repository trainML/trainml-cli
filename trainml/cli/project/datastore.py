import click
import os
import json
import base64
from pathlib import Path
from trainml.cli import pass_config
from trainml.cli.project import project


@project.group()
@pass_config
def datastore(config):
    """trainML project datastore commands."""
    pass


@datastore.command()
@pass_config
def list(config):
    """List project datastores."""
    data = [
        ["ID", "NAME", "TYPE", "REGION_UUID"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]
    project = config.trainml.run(
        config.trainml.client.projects.get(config.trainml.client.project)
    )

    datastores = config.trainml.run(project.datastores.list())

    for datastore in datastores:
        data.append(
            [
                datastore.id,
                datastore.name,
                datastore.type,
                datastore.region_uuid,
            ]
        )

    for row in data:
        click.echo(
            "{: >38.36} {: >30.28} {: >15.13} {: >38.36}" "".format(*row),
            file=config.stdout,
        )


@datastore.command()
@pass_config
def refresh(config):
    """
    Refresh project datastore list.
    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    return config.trainml.run(project.datastores.refresh())
