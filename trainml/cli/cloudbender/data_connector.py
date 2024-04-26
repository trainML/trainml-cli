import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender


@cloudbender.group()
@pass_config
def data_connector(config):
    """trainML CloudBender data connector commands."""
    pass


@data_connector.command()
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
    help="The region ID to list data connectors for.",
)
@pass_config
def list(config, provider, region):
    """List data connectors."""
    data = [
        ["ID", "NAME", "TYPE"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    data_connectors = config.trainml.run(
        config.trainml.client.cloudbender.data_connectors.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    for data_connector in data_connectors:
        data.append(
            [
                data_connector.id,
                data_connector.name,
                data_connector.type,
            ]
        )

    for row in data:
        click.echo(
            "{: >37.36} {: >29.28} {: >9.8}" "".format(*row),
            file=config.stdout,
        )


@data_connector.command()
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
    help="The region ID to create the data_connector in.",
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(
        [
            "custom",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="The type of data connector to create.",
)
@click.option(
    "--protocol",
    "-r",
    type=click.STRING,
    help="The transport protocol of the data connector",
)
@click.option(
    "--port-range",
    "-p",
    type=click.STRING,
    help="The port range of the data connector",
)
@click.option(
    "--cidr",
    "-i",
    type=click.STRING,
    help="The IP range to allow in CIDR notation",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def create(config, provider, region, type, protocol, port_range, cidr, name):
    """
    Creates a data_connector.
    """
    return config.trainml.run(
        config.trainml.client.cloudbender.data_connectors.create(
            provider_uuid=provider,
            region_uuid=region,
            name=name,
            type=type,
            protocol=protocol,
            port_range=port_range,
            cidr=cidr,
        )
    )


@data_connector.command()
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
    help="The region ID to remove the data_connector from.",
)
@click.argument("data_connector", type=click.STRING)
@pass_config
def remove(config, provider, region, data_connector):
    """
    Remove a data_connector.

    DATASTORE may be specified by name or ID, but ID is preferred.
    """
    data_connectors = config.trainml.run(
        config.trainml.client.cloudbender.data_connectors.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    found = search_by_id_name(data_connector, data_connectors)
    if None is found:
        raise click.UsageError("Cannot find specified data_connector.")

    return config.trainml.run(found.remove())
