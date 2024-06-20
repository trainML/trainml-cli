import click
import os
import json
import base64
from pathlib import Path
from trainml.cli import pass_config
from trainml.cli.project import project


@project.group()
@pass_config
def service(config):
    """trainML project service commands."""
    pass


@service.command()
@pass_config
def list(config):
    """List project services."""
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

    services = config.trainml.run(project.services.list())

    for service in services:
        data.append(
            [
                service.id,
                service.name,
                service.hostname,
                service.region_uuid,
            ]
        )

    for row in data:
        click.echo(
            "{: >38.36} {: >30.28} {: >15.13} {: >38.36}" "".format(*row),
            file=config.stdout,
        )


@service.command()
@pass_config
def refresh(config):
    """
    Refresh project service list.
    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    return config.trainml.run(project.services.refresh())
