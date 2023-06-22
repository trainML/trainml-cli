import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender


@cloudbender.group()
@pass_config
def region(config):
    """trainML CloudBender region commands."""
    pass


@region.command()
@click.option(
    "--provider",
    "-p",
    type=click.STRING,
    required=True,
    help="The provider ID to list regions for.",
)
@pass_config
def list(config, provider):
    """List regions."""
    data = [
        ["ID", "NAME", "TYPE", "STATUS"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    regions = config.trainml.run(
        config.trainml.client.cloudbender.regions.list(provider_uuid=provider)
    )

    for region in regions:
        data.append([region.id, region.name, region.type, region.status])

    for row in data:
        click.echo(
            "{: >38.36} {: >30.28} {: >15.13} {: >15.13}" "".format(*row),
            file=config.stdout,
        )


@region.command()
@click.option(
    "--provider",
    "-p",
    type=click.STRING,
    required=True,
    help="The provider ID to create the region in.",
)
@click.option(
    "--public/--no-public",
    default=True,
    show_default=True,
    help="Nodes can run other network jobs.",
)
@click.option(
    "--storage-mode",
    "-s",
    type=click.Choice(
        [
            "local",
            "central",
        ],
        case_sensitive=False,
    ),
    help="(Physical Regions Only) The type of storage configuration the region will use.",
)
@click.option(
    "--instance-class",
    "-s",
    type=click.Choice(
        ["micro", "small", "medium", "large", "xlarge"],
        case_sensitive=False,
    ),
    help="(Cloud Regions Only) The size of the storage node.",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def create(config, provider, public, storage_mode, instance_class, name):
    """
    Creates a region.
    """
    storage = dict(storage_mode=storage_mode, instance_class=instance_class)
    return config.trainml.run(
        config.trainml.client.cloudbender.regions.create(
            provider_uuid=provider, name=name, public=public, storage=storage
        )
    )


@region.command()
@click.option(
    "--provider",
    "-p",
    type=click.STRING,
    required=True,
    help="The provider ID to delete region from.",
)
@click.argument("region", type=click.STRING)
@pass_config
def remove(config, provider, region):
    """
    Remove a region.

    REGION may be specified by name or ID, but ID is preferred.
    """
    regions = config.trainml.run(
        config.trainml.client.cloudbender.regions.list(provider_uuid=provider)
    )

    found = search_by_id_name(region, regions)
    if None is found:
        raise click.UsageError("Cannot find specified region.")

    return config.trainml.run(found.remove())
