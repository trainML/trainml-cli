import asyncio
import click
from . import cli, pass_config
from trainml.trainml import TrainML


@cli.group()
@pass_config
def connection(config):
    """TrainML connection commands."""
    pass


@connection.command()
@pass_config
def list(config):
    """List connections."""
    data = [['ID', 'TYPE', 'STATUS'],
            ['-'*80, '-'*80, '-'*80]]

    try:
        trainml_client = TrainML()
        connections = asyncio.run(
            trainml_client.connections.list()
        )
    except Exception as err:
        raise click.UsageError(err)

    for con in connections:
        data.append([con.id, con.type, con.status])
    for row in data:
        click.echo("{: >38.36} {: >9.7} {: >15.13}".format(*row), file=config.output)
