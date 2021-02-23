import asyncio
import click
from trainml.trainml import TrainML


class Config(object):

    def __init__(self):
        self.output = None


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--output-file', '-o', envvar='TRAINML_OUT', type=click.File('w'),
                default='-', help='Send output to file.')
@pass_config
def cli(config, output_file):
    """TrainML command-line interface."""
    config.output = output_file




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
    trainml_client = TrainML()
    datasets = asyncio.run(
        trainml_client.datasets.list()
    )
    for dset in datasets:
        data.append([dset.id, dset.status, dset.provider, dset.name, str(dset.size)])
    for row in data:
        click.echo("{: >38} {: >13} {: >10} {: >40} {: >14}".format(*row), file=config.output)


@dataset.command()
@pass_config
def list_public(config):
    """List public datasets."""
    data = [['ID', 'STATUS', 'PROVIDER', 'NAME', 'SIZE']]
    trainml_client = TrainML()
    datasets = asyncio.run(
        trainml_client.datasets.list_public()
    )
    for dset in datasets:
        data.append([dset.id, dset.status, dset.provider, dset.name, str(dset.size)])
    for row in data:
        click.echo("{: >38} {: >13} {: >10} {: >40} {: >14}".format(*row), file=config.output)




@cli.group()
@pass_config
def job(config):
    """TrainML job commands."""
    pass


@job.command()
@pass_config
def list(config):
    """List TrainML jobs."""
    data = [['ID', 'NAME', 'STATUS', 'PROVIDER', 'TYPE']]
    trainml_client = TrainML()
    jobs = asyncio.run(
        trainml_client.jobs.list()
    )
    for job in jobs:
        data.append([job.id, job.name, job.status, job.provider, job.type])
    for row in data:
        click.echo("{: >38} {: >40} {: >13} {: >10} {: >14}".format(*row), file=config.output)

    


if __name__ == '__main__':
    cli()