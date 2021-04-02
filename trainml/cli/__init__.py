import asyncio
import click
import logging
from os import devnull
from sys import stderr, stdout
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
        self.stderr = stderr
        self.stdout = stdout
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
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    type=click.BOOL,
    default=False,
    help="Show debug output.",
)
@click.option(
    "--output-file",
    "-o",
    type=click.File("w"),
    default="-",
    help="Send output to file.",
)
@click.option(
    "--silent",
    "-s",
    is_flag=True,
    type=click.BOOL,
    default=False,
    help="Silence all output.",
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    type=click.INT,
    default=0,
    help="Specify verbosity (repeat to increase).",
)
@pass_config
def cli(config, debug, output_file, silent, verbosity):
    """TrainML command-line interface."""
    config.stdout = output_file

    if debug or verbosity > 0:
        if silent:
            click.echo(
                "Ignoring silent flag when debug or verbosity is set.",
                file=config.stderr,
            )
        if verbosity == 1:
            verbosity = logging.INFO
        else:
            verbosity = logging.DEBUG
    elif silent:
        config.stderr = config.stdout = open(devnull, "w")
    else:
        verbosity = logging.WARNING

    if verbosity != logging.WARNING:  # default
        click.echo(
            f"Verbosity set to {logging.getLevelName(verbosity)}",
            file=config.stderr,
        )

    logging.basicConfig(
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=verbosity,
        stream=config.stderr,
        force=True,
    )


from trainml.cli.connection import connection
from trainml.cli.dataset import dataset
from trainml.cli.model import model
from trainml.cli.environment import environment
from trainml.cli.gpu import gpu
from trainml.cli.job import job
