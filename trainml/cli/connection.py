import click
from . import cli, pass_config


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

    connections = config.trainml.run(
        config.trainml.client.connections.list())
    
    for con in connections:
        data.append([con.id, con.type, con.status])
    for row in data:
        click.echo("{: >38.36} {: >9.7} {: >15.13}".format(*row), file=config.output)
