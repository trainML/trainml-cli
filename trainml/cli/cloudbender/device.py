import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender


@cloudbender.group()
@pass_config
def device(config):
    """trainML CloudBender device commands."""
    pass


@device.command()
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
    help="The region ID to list devices for.",
)
@pass_config
def list(config, provider, region):
    """List devices."""
    data = [
        [
            "ID",
            "NAME",
            "STATUS",
            "JOB STATUS",
            "ONLINE",
            "MAINTENANCE",
        ],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    devices = config.trainml.run(
        config.trainml.client.cloudbender.devices.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    for device in devices:
        data.append(
            [
                device.id,
                device.name,
                device.status,
                device.job_status,
                "X" if device.online else "",
                "X" if device.maintenance_mode else "",
            ]
        )

    for row in data:
        click.echo(
            "{: >37.36} {: >29.28} {: >9.8} {: >11.10} {: >7.6} {: >12.11}"
            "".format(*row),
            file=config.stdout,
        )


@device.command()
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
    help="The region ID to create the region in.",
)
@click.option(
    "--minion-id",
    "-m",
    type=click.STRING,
    required=True,
    help="The minion_id of the new node.",
)
@click.option(
    "--hostname",
    "-h",
    type=click.STRING,
    help="The hostname (if different from name)",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def create(config, provider, region, minion_id, hostname, name):
    """
    Creates a node.
    """
    if not hostname:
        hostname = name
    return config.trainml.run(
        config.trainml.client.cloudbender.devices.create(
            provider_uuid=provider,
            region_uuid=region,
            friendly_name=name,
            hostname=hostname,
            minion_id=minion_id,
        )
    )


@device.command()
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
    help="The region ID to delete the node from.",
)
@click.argument("device", type=click.STRING)
@pass_config
def remove(config, provider, region, node):
    """
    Remove a device.

    DEVICE may be specified by name or ID, but ID is preferred.
    """
    devices = config.trainml.run(
        config.trainml.client.cloudbender.devices.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    found = search_by_id_name(device, devices)
    if None is found:
        raise click.UsageError("Cannot find specified device.")

    return config.trainml.run(found.remove())
