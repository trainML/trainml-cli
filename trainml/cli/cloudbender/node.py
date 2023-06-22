import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender


@cloudbender.group()
@pass_config
def node(config):
    """trainML CloudBender node commands."""
    pass


@node.command()
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
    help="The region ID to list nodes for.",
)
@pass_config
def list(config, provider, region):
    """List nodes."""
    data = [
        ["ID", "NAME", "SERVICE", "STATUS", "ONLINE", "MAINTENANCE"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    nodes = config.trainml.run(
        config.trainml.client.cloudbender.nodes.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    for node in nodes:
        data.append(
            [
                node.id,
                node.name,
                node.service,
                node.status,
                "X" if node.online else "",
                "X" if node.maintenance_mode else "",
            ]
        )

    for row in data:
        click.echo(
            "{: >37.36} {: >29.28} {: >9.8} {: >11.10} {: >7.6} {: >12.11}"
            "".format(*row),
            file=config.stdout,
        )


@node.command()
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
    "--type",
    "-t",
    type=click.Choice(
        [
            "permanent",
            "device",
        ],
        case_sensitive=False,
    ),
    default="permanent",
    show_default=True,
    help="The type of node to create.",
)
@click.option(
    "--service",
    "-s",
    type=click.Choice(
        ["compute", "storage"],
        case_sensitive=False,
    ),
    default="compute",
    show_default=True,
    help="The service the node will fulfill.",
)
@click.option(
    "--hostname",
    "-h",
    type=click.STRING,
    help="The hostname (if different from name)",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def create(config, provider, region, type, service, minion_id, hostname, name):
    """
    Creates a node.
    """
    if not hostname:
        hostname = name
    return config.trainml.run(
        config.trainml.client.cloudbender.nodes.create(
            provider_uuid=provider,
            region_uuid=region,
            friendly_name=name,
            hostname=hostname,
            minion_id=minion_id,
            type=type,
            service=service,
        )
    )


@node.command()
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
@click.argument("node", type=click.STRING)
@pass_config
def remove(config, provider, region, node):
    """
    Remove a node.

    NODE may be specified by name or ID, but ID is preferred.
    """
    nodes = config.trainml.run(
        config.trainml.client.cloudbender.nodes.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    found = search_by_id_name(node, nodes)
    if None is found:
        raise click.UsageError("Cannot find specified node.")

    return config.trainml.run(found.remove())
