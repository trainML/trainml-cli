import asyncio
import click
from webbrowser import open as browse
from trainml.cli import cli, pass_config, search_by_id_name


@cli.group()
@pass_config
def job(config):
    """trainML job commands."""
    pass


@job.command()
@click.argument("job", type=click.STRING)
@pass_config
def attach(config, job):
    """
    Attach to job and show logs.

    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(config.trainml.client.jobs.list())

    found = search_by_id_name(job, jobs)
    if None is found:
        raise click.UsageError("Cannot find specified job.")

    config.trainml.run(found.attach())


async def _connect_job(job, attach, config):
    """
    Async helper function to handle job connection with proper
    handling of local input/output types and attach task management.
    """
    # Get job properties
    model = job._job.get("model", {})
    data = job._job.get("data", {})
    model_local = model.get("source_type") == "local"
    data_local = data.get("input_type") == "local"
    output_local = data.get("output_type") == "local"
    early_statuses = [
        "new",
        "waiting for data/model download",
        "waiting for GPUs",
        "waiting for resources",
    ]

    # Check if we need to wait for data/model download
    # Only wait if status is early AND (data or model is local)
    needs_upload_wait = job.status in early_statuses and (
        model_local or data_local
    )

    if needs_upload_wait:
        # Wait for job to reach data/model download status
        await job.wait_for("waiting for data/model download", 3600)
        await job.refresh()

    # Start attach task early if requested
    attach_task = None
    if attach:
        attach_task = asyncio.create_task(job.attach())

    # Run first connect (upload if needed)
    await job.connect()

    # For notebook jobs, handle opening
    if job.type == "notebook":
        # Refresh to get latest status after connect
        await job.refresh()

        if job.status in early_statuses:
            if attach_task:
                await attach_task
            click.echo("Launching...", file=config.stdout)
            browse(job.notebook_url)
            return
        elif job.status not in [
            "starting",
            "running",
            "reinitializing",
            "copying",
        ]:
            if attach_task:
                attach_task.cancel()
            raise click.UsageError("Notebook job not running.")
        else:
            await job.wait_for("running")
            if attach_task:
                await attach_task
            click.echo("Launching...", file=config.stdout)
            browse(job.notebook_url)
            return

    # For non-notebook jobs, check if we need second connect (download)
    # Refresh to get latest status after first connect
    await job.refresh()

    # Run second connect if output_type is local
    # (as per user's requirement: "if the output_type is 'local'")
    if output_local:
        # Always wait for running status before second connect
        # (as shown in user's example)
        await job.wait_for("running", 3600)
        await job.refresh()

        # Create second connect task (download)
        connect_task = asyncio.create_task(job.connect())

        # Gather both attach and second connect tasks
        if attach_task:
            await asyncio.gather(attach_task, connect_task)
        else:
            await connect_task
    elif attach_task:
        # Just wait for attach if no second connect needed
        await attach_task


@job.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to job.",
)
@click.argument("job", type=click.STRING)
@pass_config
def connect(config, job, attach):
    """
    Connect to job.

    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(config.trainml.client.jobs.list())

    found = search_by_id_name(job, jobs)
    if None is found:
        raise click.UsageError("Cannot find specified job.")

    config.trainml.run(_connect_job(found, attach, config))


@job.command()
@click.option(
    "--wait/--no-wait",
    default=True,
    show_default=True,
    help="Wait until job is stopped before returning.",
)
@click.argument("job", type=click.STRING)
@pass_config
def stop(config, job, wait):
    """
    Stop a running job.

    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(config.trainml.client.jobs.list())

    found = search_by_id_name(job, jobs)
    if None is found:
        raise click.UsageError("Cannot find specified job.")

    if wait:
        config.trainml.run(found.stop())
        click.echo("Waiting for job to stop...", file=config.stdout)
        return config.trainml.run(found.wait_for("stopped"))
    else:
        return config.trainml.run(found.stop())


@job.command()
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect to job.",
)
@click.argument("job", type=click.STRING)
@pass_config
def start(config, job, connect):
    """
    Start a previously stopped job.

    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(config.trainml.client.jobs.list())

    found = search_by_id_name(job, jobs)
    if None is found:
        raise click.UsageError("Cannot find specified job.")

    if connect:
        config.trainml.run(found.start())
        click.echo("Waiting for job to start...", file=config.stdout)
        config.trainml.run(found.wait_for("running"))
        click.echo("Launching...", file=config.stdout)
        browse(found.notebook_url)
    else:
        return config.trainml.run(found.start())


@job.command()
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Force removal.",
)
@click.argument("job", type=click.STRING)
@pass_config
def remove(config, job, force):
    """
    Remove a job.

    JOB may be specified by name or ID, but ID is preferred.
    """
    jobs = config.trainml.run(config.trainml.client.jobs.list())

    found = search_by_id_name(job, jobs)
    if None is found:
        if force:
            config.trainml.run(config.trainml.client.jobs.remove(job))
        else:
            raise click.UsageError("Cannot find specified job.")

    return config.trainml.run(found.remove(force=force))


@job.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Choose output format.",
)
@pass_config
def list(config, format):
    """List trainML jobs."""
    jobs = config.trainml.run(config.trainml.client.jobs.list())

    if format == "text":
        data = [
            ["ID", "NAME", "STATUS", "TYPE"],
            ["-" * 80, "-" * 80, "-" * 80, "-" * 80],
        ]

        for job in jobs:
            data.append([job.id, job.name, job.status, job.type])
        for row in data:
            click.echo(
                "{: >38.36} {: >40.38} {: >13.11} {: >14.12}" "".format(*row),
                file=config.stdout,
            )
    elif format == "json":
        output = []
        for job in jobs:
            output.append(job.dict)
        click.echo(output, file=config.stdout)


from trainml.cli.job.create import create
