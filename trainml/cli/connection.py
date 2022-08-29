import click
from trainml.cli import cli, pass_config, search_by_id_name


@cli.group()
@pass_config
def connection(config):
    """TrainML connection commands."""
    pass


@connection.command()
@pass_config
def list(config):
    """List connections."""
    data = [["ID", "TYPE", "STATUS"], ["-" * 80, "-" * 80, "-" * 80]]

    connections = config.trainml.run(config.trainml.client.connections.list())

    for con in connections:
        data.append([con.id, con.type, con.status])
    for row in data:
        click.echo(
            "{: >38.36} {: >9.7} {: >15.13}".format(*row), file=config.stdout
        )


@connection.command()
@click.argument("id", type=click.STRING)
@pass_config
def remove(config, id):
    """Remove connection."""
    connections = config.trainml.run(config.trainml.client.connections.list())

    found = search_by_id_name(id, connections)
    if None is found:
        raise click.UsageError("Connection ID specified does not exist.")

    if found.type == "dataset":
        this = config.trainml.run(config.trainml.client.datasets.get(id))
    elif found.type == "job":
        this = config.trainml.run(config.trainml.client.jobs.get(id))
    else:
        raise click.UsageError("Unknown connection type.")

    return config.trainml.run(this.disconnect())


@connection.command()
@click.option(
    "--all-projects/--no-all-projects",
    default=False,
    show_default=True,
    help="Auto attach to dataset and show creation logs.",
)
@pass_config
def remove_all(config, all_projects):
    """Clear and clean-up all TrainML connections."""
    return config.trainml.run(
        config.trainml.client.connections.remove_all(all_projects=all_projects)
    )
