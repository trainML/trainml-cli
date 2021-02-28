import click
from . import cli, pass_config


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
    
    found = False
    for j in jobs:
        if j.id == job:
            job = j
            found = True
            break
    if not found:
        for j in jobs:
            if j.name == job:
                job = j
                found = True
                break
    if not found:
        raise click.UsageError('Cannot find specified job.')

    return config.trainml.run(job.attach())


@job.command()
@click.option(
    '--attach/--no-attach',
    default=True,
    show_default=True,
    help='Auto attach to job and show logs.'
)
@click.argument('job', type=click.STRING)
@pass_config
def connect(config, job, attach):
    """
    Connect to job to download output locally.
    
    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(
        config.trainml.client.jobs.list())
    
    found = False
    for j in jobs:
        if j.id == job:
            job = j
            found = True
            break
    if not found:
        for j in jobs:
            if j.name == job:
                job = j
                found = True
                break
    if not found:
        raise click.UsageError('Cannot find specified job.')

    if attach:
        config.trainml.run(job.connect(), job.attach())
        return config.trainml.run(job.disconnect())
    else:
        return config.trainml.run(job.connect())


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
    
    found = False
    for j in jobs:
        if j.id == job:
            job = j
            found = True
            break
    if not found:
        for j in jobs:
            if j.name == job:
                job = j
                found = True
                break
    if not found:
        raise click.UsageError('Cannot find specified job.')

    return config.trainml.run(job.disconnect())


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
