import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender


@cloudbender.group()
@pass_config
def service(config):
    """trainML CloudBender service commands."""
    pass


@service.command()
@click.option(
    "--provider",
    "-p",
    type=click.STRING,
    required=True,
    help="The provider ID of the region.",
)
@click.option(
    "--region",
    "-r",
    type=click.STRING,
    required=True,
    help="The region ID to list services for.",
)
@pass_config
def list(config, provider, region):
    """List services."""
    data = [
        ["ID", "NAME", "HOSTNAME"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    services = config.trainml.run(
        config.trainml.client.cloudbender.services.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    for service in services:
        data.append(
            [
                service.id,
                service.name,
                service.hostname,
            ]
        )

    for row in data:
        click.echo(
            "{: >25.24} {: >29.28} {: >40.39}" "".format(*row),
            file=config.stdout,
        )


@service.command()
@click.option(
    "--provider",
    "-p",
    type=click.STRING,
    required=True,
    help="The provider ID of the region.",
)
@click.option(
    "--region",
    "-r",
    type=click.STRING,
    required=True,
    help="The region ID to create the service in.",
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(
        [
            "https",
            "tcp",
            "udp",
        ],
    ),
    required=True,
    help="The type of regional service.",
)
@click.option(
    "--public/--no-public",
    default=True,
    show_default=True,
    help="Service should be accessible from the public internet.",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def create(config, provider, region, type, public, name):
    """
    Creates a service.
    """
    return config.trainml.run(
        config.trainml.client.cloudbender.services.create(
            provider_uuid=provider,
            region_uuid=region,
            name=name,
            type=type,
            public=public,
        )
    )


@service.command()
@click.option(
    "--provider",
    "-p",
    type=click.STRING,
    required=True,
    help="The provider ID of the region.",
)
@click.option(
    "--region",
    "-r",
    type=click.STRING,
    required=True,
    help="The region ID to remove the service from.",
)
@click.argument("service", type=click.STRING)
@pass_config
def remove(config, provider, region, service):
    """
    Remove a service.

    RESERVATION may be specified by name or ID, but ID is preferred.
    """
    services = config.trainml.run(
        config.trainml.client.cloudbender.services.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    found = search_by_id_name(service, services)
    if None is found:
        raise click.UsageError("Cannot find specified service.")

    return config.trainml.run(found.remove())
