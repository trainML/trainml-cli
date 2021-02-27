import click
from . import cli, pass_config


@cli.group()
@pass_config
def dataset(config):
    """TrainML dataset commands."""
    pass


@dataset.command()
@click.option(
    '--source', '-s',
    type=click.Choice(['local'], case_sensitive=False),
    default='local',
    show_default=True,
    help='Dataset source type.'
)
@click.argument('name', type=click.STRING)
@click.argument(
    'path',
    type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@pass_config
def create(config, source, name, path):
    """
    Creates a dataset with the specified NAME using a local source at the PATH
    specified. PATH should be a local directory containing the source data for
    a local source or a URI for all other source types.
    """

    if source == 'local':
        dataset = config.trainml.run(
            config.trainml.client.datasets.create(
                name=name,
                source_type="local",
                source_uri=path
            )
        )
        
        config.trainml.run(dataset.attach(), dataset.connect())
        config.trainml.run(dataset.disconnect())


@dataset.command()
@pass_config
def list(config):
    """List datasets."""
    data = [['ID', 'STATUS', 'PROVIDER', 'NAME', 'SIZE'],
            ['-'*80, '-'*80, '-'*80, '-'*80, '-'*80]]

    datasets = config.trainml.run(
        config.trainml.client.datasets.list())
    
    for dset in datasets:
        data.append([dset.id, dset.status, dset.provider, dset.name, str(dset.size)])
    for row in data:
        click.echo("{: >38.36} {: >13.11} {: >10.8} {: >40.38} {: >14.12}".format(*row), file=config.output)


@dataset.command()
@pass_config
def list_public(config):
    """List public datasets."""
    data = [['ID', 'STATUS', 'PROVIDER', 'NAME', 'SIZE'],
            ['-'*80, '-'*80, '-'*80, '-'*80, '-'*80]]

    datasets = config.trainml.run(
        config.trainml.client.datasets.list_public())
    
    for dset in datasets:
        data.append([dset.id, dset.status, dset.provider, dset.name, str(dset.size)])
    for row in data:
        click.echo("{: >38.36} {: >13.11} {: >10.8} {: >40.38} {: >14.12}".format(*row), file=config.output)
