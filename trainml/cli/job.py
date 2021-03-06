import click
from webbrowser import open as browse
from . import cli, pass_config, search_by_id_name


@cli.group()
@pass_config
def job(config):
    """TrainML job commands."""
    pass


@job.command()
@click.argument('job', type=click.STRING)
@pass_config
def attach(config, job):
    """
    Attach to job and show logs.
    
    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(
        config.trainml.client.jobs.list())
    
    found = search_by_id_name(job, jobs)
    if None is found:
        raise click.UsageError('Cannot find specified job.')

    try:
        config.trainml.run(found.attach())
        return config.trainml.run(found.disconnect())
    except:
        try:
            config.trainml.run(found.disconnect())
        except:
            pass
        raise


@job.command()
@click.option(
    '--attach/--no-attach',
    default=True,
    show_default=True,
    help='Auto attach to job.'
)
@click.argument('job', type=click.STRING)
@pass_config
def connect(config, job, attach):
    """
    Connect to job.
    
    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(
        config.trainml.client.jobs.list())
    
    found = search_by_id_name(job, jobs)
    if None is found:
        raise click.UsageError('Cannot find specified job.')

    try:
        if attach:
            config.trainml.run(found.connect(), found.attach())
            return config.trainml.run(found.disconnect())
        else:
            return config.trainml.run(found.connect())
    except:
        try:
            config.trainml.run(found.disconnect())
        except:
            pass
        raise


@job.command()
@click.option(
    '--attach/--no-attach',
    default=True,
    show_default=True,
    help='Auto attach to job.'
)
@click.option(
    '--connect/--no-connect',
    default=True,
    show_default=True,
    help='Auto connect to job.'
)
@click.option(
    '--disk-size', '-ds',
    type=click.INT,
    default=10,
    show_default=True,
    help='Disk size (GiB).'
)
@click.option(
    '--gpu-count', '-gc',
    type=click.INT,
    default=1,
    show_default=True,
    help='GPU Count (per Worker.)'
)
@click.option(
    '--gpu-type', '-gt',
    type=click.Choice(['GTX 1060'], case_sensitive=False),
    default='GTX 1060',
    show_default=True,
    help='GPU type.'
)
@click.option(
    '--type', '-t',
    type=click.Choice(['interactive'], case_sensitive=False),
    default='interactive',
    show_default=True,
    help='Job type.'
)
@click.argument('name', type=click.STRING)
@pass_config
def create(config, attach, connect, disk_size, gpu_count, gpu_type, type, name):
    """
    Create job.
    """
    if type == 'interactive':
        job = config.trainml.run(
            config.trainml.client.jobs.create(
                name=name,
                type=type,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                disk_size=disk_size,
            )
        )
        click.echo('Created.', file=config.output)
        if attach or connect:
            click.echo('Waiting for job to start...', file=config.output)
            config.trainml.run(job.wait_for('running'))
            click.echo('Launching...', file=config.output)
            browse(job.notebook_url)


@job.command()
@click.argument('job', type=click.STRING)
@pass_config
def disconnect(config, job):
    """
    Disconnect and clean-up job.
    
    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(
        config.trainml.client.jobs.list())
    
    found = search_by_id_name(job, jobs)
    if None is found:
        raise click.UsageError('Cannot find specified job.')

    return config.trainml.run(found.disconnect())


@job.command()
@pass_config
def list(config):
    """List TrainML jobs."""
    data = [['ID', 'NAME', 'STATUS', 'PROVIDER', 'TYPE'],
            ['-'*80, '-'*80, '-'*80, '-'*80, '-'*80]]

    jobs = config.trainml.run(
        config.trainml.client.jobs.list())
    
    for job in jobs:
        data.append([job.id, job.name, job.status, job.provider, job.type])
    for row in data:
        click.echo("{: >38.36} {: >40.38} {: >13.11} {: >10.8} {: >14.12}".format(*row), file=config.output)
