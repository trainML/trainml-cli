import click
from . import cli, pass_config


@cli.group()
@pass_config
def dataset(config):
    """TrainML dataset commands."""
    pass


@dataset.command()
@click.argument('dataset', type=click.STRING)
@pass_config
def connect(config, dataset):
    """
    Connect local source to dataset and begin upload.
    
    DATASET may be specified by name or ID, but ID is preferred.
    """
    datasets = config.trainml.run(
        config.trainml.client.datasets.list())
    
    found = False
    for dset in datasets:
        if dset.id == dataset:
            dataset = dset
            found = True
            break
    if not found:
        for dset in datasets:
            if dset.name == dataset:
                dataset = dset
                found = True
                break
    if not found:
        raise click.UsageError('Cannot find specified dataset.')

    return config.trainml.run(dataset.connect())

@dataset.command()
@click.option(
    '--attach/--no-attach',
    default=True,
    show_default=True,
    help='Attach to dataset and show creation logs.'
)
@click.option(
    '--connect/--no-connect',
    default=True,
    show_default=True,
    help='Auto connect source and start dataset creation.'
)
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
def create(config, attach, connect, source, name, path):
    """
    Create a dataset.

    Dataset is created with the specified NAME using a local source at the PATH
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
        
        if connect and attach:
            config.trainml.run(dataset.attach(), dataset.connect())
            return config.trainml.run(dataset.disconnect())
        elif connect:
            return config.trainml.run(dataset.connect())
        else:
            raise click.UsageError('Cannot attach without connect.')


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
