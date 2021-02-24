import asyncio
import click
from . import cli, pass_config
from trainml.trainml import TrainML


@cli.group()
@pass_config
def gpu(config):
    """TrainML GPU commands."""
    pass


@gpu.command()
@pass_config
def list(config):
    """List GPUs."""
    data = [['ID', 'NAME', 'PROVIDER', 'AVAILABLE', 'CREDITS/HR'],
            ['-'*80, '-'*80, '-'*80, '-'*80, '-'*80]]

    try:
        trainml_client = TrainML()
        gpus = asyncio.run(
            trainml_client.gpu_types.list()
        )
    except Exception as err:
        raise click.UsageError(err)

    for gpu in gpus:
        data.append([gpu.id, gpu.name, gpu.provider, str(gpu.available), str(gpu.credits_per_hour)])
    for row in data:
        click.echo("{: >36.34} {: >16.14} {: >10.8} {: >11.9} {: >12.10}".format(*row), file=config.output)
