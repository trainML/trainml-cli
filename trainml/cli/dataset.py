import asyncio
import click
from . import cli, pass_config
from trainml.trainml import TrainML


@cli.group()
@pass_config
def dataset(config):
    """TrainML dataset commands."""
    pass

@dataset.command()
@pass_config
def list(config):
    """List datasets."""
    data = [['ID', 'STATUS', 'PROVIDER', 'NAME', 'SIZE']]

    try:
        trainml_client = TrainML()
        datasets = asyncio.run(
            trainml_client.datasets.list()
        )
    except Exception as err:
        raise click.UsageError(err)
    
    for dset in datasets:
        data.append([dset.id, dset.status, dset.provider, dset.name, str(dset.size)])
    for row in data:
        click.echo("{: >38} {: >13} {: >10} {: >40} {: >14}".format(*row), file=config.output)


@dataset.command()
@pass_config
def list_public(config):
    """List public datasets."""
    data = [['ID', 'STATUS', 'PROVIDER', 'NAME', 'SIZE']]

    try:
        trainml_client = TrainML()
        datasets = asyncio.run(
            trainml_client.datasets.list_public()
        )
    except Exception as err:
        raise click.UsageError(err)

    for dset in datasets:
        data.append([dset.id, dset.status, dset.provider, dset.name, str(dset.size)])
    for row in data:
        click.echo("{: >38} {: >13} {: >10} {: >40} {: >14}".format(*row), file=config.output)
