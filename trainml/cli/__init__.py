import asyncio
import click
import logging
from trainml.trainml import TrainML


class TrainMLRunner(object):

    def __init__(self):
        self._trainml_client = None
    
    @property
    def client(self) -> TrainML:
        if self._trainml_client is None:
            try:
                self._trainml_client = TrainML()
            except Exception as err:
                raise click.UsageError(err)
        return self._trainml_client
    
    async def _run(self, *tasks):
        return await asyncio.gather(*tasks)
    
    def run(self, *tasks):
        try:
            if len(tasks) == 1:
                return_value = asyncio.run(*tasks)
            else:
                return_value = asyncio.run(self._run(*tasks))
        except Exception as err:
            raise click.UsageError(err)
        return return_value            


class Config(object):

    def __init__(self):
        self.output = None
        self.trainml = TrainMLRunner()


def search_by_id_name(term, list):
    found = None
    for item in list:
        if item.id == term:
            found = item
            break
    if None is found:
        for item in list:
            try:
                if item.name == term:
                    found = item
                    break
            except AttributeError:
                break
    return found

    
pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--output-file', '-o', envvar='TRAINML_OUT', type=click.File('w'),
                default='-', help='Send output to file.')
@pass_config
def cli(config, output_file):
    """TrainML command-line interface."""
    config.output = output_file
    logging.basicConfig(
        format='%(asctime)s  %(levelname)s  %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        stream=output_file
    )


from .connection import connection
from .dataset import dataset
from .environment import environment
from .gpu import gpu
from .job import job
