import click


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


from .connection import connection
from .dataset import dataset
from .environment import environment
from .gpu import gpu
from .job import job
