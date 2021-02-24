import asyncio
import click
from . import cli, pass_config
from trainml.trainml import TrainML


@cli.group()
@pass_config
def job(config):
    """TrainML job commands."""
    pass


@job.command()
@pass_config
def list(config):
    """List TrainML jobs."""
    data = [['ID', 'NAME', 'STATUS', 'PROVIDER', 'TYPE'],
            ['-'*80, '-'*80, '-'*80, '-'*80, '-'*80]]

    try:
        trainml_client = TrainML()
        jobs = asyncio.run(
            trainml_client.jobs.list()
        )
    except Exception as err:
        raise click.UsageError(err)

    for job in jobs:
        data.append([job.id, job.name, job.status, job.provider, job.type])
    for row in data:
        click.echo("{: >38.36} {: >40.38} {: >13.11} {: >10.8} {: >14.12}".format(*row), file=config.output)
