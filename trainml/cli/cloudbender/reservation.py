import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender


@cloudbender.group()
@pass_config
def reservation(config):
    """trainML CloudBender reservation commands."""
    pass


@reservation.command()
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
    help="The region ID to list reservations for.",
)
@pass_config
def list(config, provider, region):
    """List reservations."""
    data = [
        ["ID", "NAME", "TYPE", "RESOURCE", "HOSTNAME"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    reservations = config.trainml.run(
        config.trainml.client.cloudbender.reservations.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    for reservation in reservations:
        data.append(
            [
                reservation.id,
                reservation.name,
                reservation.type,
                reservation.resource,
                reservation.hostname,
            ]
        )

    for row in data:
        click.echo(
            "{: >37.36} {: >29.28} {: >9.8} {: >9.8} {: >29.28}"
            "".format(*row),
            file=config.stdout,
        )


@reservation.command()
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
    help="The region ID to create the reservation in.",
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(
        [
            "port",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="The type of reservation to create.",
)
@click.option(
    "--hostname",
    "-h",
    type=click.STRING,
    required=True,
    help="The hostname to make the reservation on",
)
@click.option(
    "--resource",
    "-r",
    type=click.STRING,
    required=True,
    help="The resource to reserve",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def create(config, provider, region, type, hostname, resource, name):
    """
    Creates a reservation.
    """
    return config.trainml.run(
        config.trainml.client.cloudbender.reservations.create(
            provider_uuid=provider,
            region_uuid=region,
            name=name,
            hostname=hostname,
            resource=resource,
            type=type,
        )
    )


@reservation.command()
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
    help="The region ID to remove the reservation from.",
)
@click.argument("reservation", type=click.STRING)
@pass_config
def remove(config, provider, region, reservation):
    """
    Remove a reservation.

    RESERVATION may be specified by name or ID, but ID is preferred.
    """
    reservations = config.trainml.run(
        config.trainml.client.cloudbender.reservations.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    found = search_by_id_name(reservation, reservations)
    if None is found:
        raise click.UsageError("Cannot find specified reservation.")

    return config.trainml.run(found.remove())
