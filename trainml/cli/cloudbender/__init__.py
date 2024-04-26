import click
from webbrowser import open as browse
from trainml.cli import cli, pass_config, search_by_id_name


@cli.group()
@pass_config
def cloudbender(config):
    """trainML CloudBender™ commands."""
    pass


from trainml.cli.cloudbender.provider import provider
from trainml.cli.cloudbender.region import region
from trainml.cli.cloudbender.node import node
from trainml.cli.cloudbender.device import device
from trainml.cli.cloudbender.datastore import datastore
from trainml.cli.cloudbender.data_connector import data_connector
from trainml.cli.cloudbender.service import service
