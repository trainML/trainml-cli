import asyncio
import click
from . import cli, pass_config
from trainml.trainml import TrainML


@cli.group()
@pass_config
def environment(config):
    """TrainML environment commands."""
    pass


@environment.command()
@pass_config
def list(config):
    """List environments."""
    data = [['ID', 'NAME', 'PYTHON', 'FRAMEWORK', 'VERSION', 'CUDA'],
            ['-'*80, '-'*80, '-'*80, '-'*80, '-'*80, '-'*80]]

    try:
        trainml_client = TrainML()
        environments = asyncio.run(
            trainml_client.environments.list()
        )
    except Exception as err:
        raise click.UsageError(err)

    for env in environments:
        data.append([env.id, env.name, env.py_version, env.framework, str(env.version), env.cuda_version])
    for row in data:
        click.echo("{: >21.19} {: >30.28} {: >8.6} {: >15.13} {: >9.7} {: >6.4}".format(*row), file=config.output)
