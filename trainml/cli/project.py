import click
from trainml.cli import cli, pass_config, search_by_id_name


@cli.group()
@pass_config
def project(config):
    """trainML project commands."""
    pass


@project.command()
@pass_config
def list(config):
    """List projects."""
    data = [
        ["ID", "NAME", "OWNER", "MINE"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    projects = config.trainml.run(config.trainml.client.projects.list())

    for project in projects:
        data.append(
            [
                project.id,
                project.name,
                project.owner_name,
                "X" if project.is_owner else "",
            ]
        )

    for row in data:
        click.echo(
            "{: >38.36} {: >30.28} {: >15.13} {: >4.4}" "".format(*row),
            file=config.stdout,
        )


@project.command()
@click.argument("name", type=click.STRING)
@pass_config
def create(config, name):
    """
    Create a project.

    Project is created with the specified NAME.
    """

    return config.trainml.run(
        config.trainml.client.projects.create(
            name=name,
        )
    )


@project.command()
@click.argument("project", type=click.STRING)
@pass_config
def remove(config, project):
    """
    Remove a project.

    PROJECT may be specified by name or ID, but ID is preferred.
    """
    projects = config.trainml.run(config.trainml.client.projects.list())

    found = search_by_id_name(project, projects)
    if None is found:
        raise click.UsageError("Cannot find specified project.")

    return config.trainml.run(found.remove())


@project.command()
@pass_config
def list_datastores(config):
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

    datastores = config.trainml.run(project.list_datastores())

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


@project.command()
@pass_config
def list_services(config):
    """List project services."""
    data = [
        ["ID", "NAME", "HOSTNAME", "REGION_UUID"],
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

    services = config.trainml.run(project.list_services())

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
            "{: >38.36} {: >30.28} {: >30.28} {: >38.36}" "".format(*row),
            file=config.stdout,
        )
