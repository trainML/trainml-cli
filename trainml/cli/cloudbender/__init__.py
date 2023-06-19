import click
from webbrowser import open as browse
from trainml.cli import cli, pass_config, search_by_id_name


@cli.group()
@pass_config
def cloudbender(config):
    """trainML cloudbender commands."""
    pass


from trainml.cli.cloudbender.provider import provider
