import click
from trainml.cli import cli, pass_config, search_by_id_name


def pretty_size(num):
    if not num:
        num = 0.0
    s = ("  B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    n = 0
    while num > 1023:
        num = num / 1024
        n += 1
    return f"{num:.2f} {s[n]}"


@cli.group()
@pass_config
def checkpoint(config):
    """trainML checkpoint commands."""
    pass


@checkpoint.command()
@click.argument("checkpoint", type=click.STRING)
@pass_config
def attach(config, checkpoint):
    """
    Attach to checkpoint and show creation logs.

    CHECKPOINT may be specified by name or ID, but ID is preferred.
    """
    checkpoints = config.trainml.run(config.trainml.client.checkpoints.list())

    found = search_by_id_name(checkpoint, checkpoints)
    if None is found:
        raise click.UsageError("Cannot find specified checkpoint.")

    try:
        config.trainml.run(found.attach())
        return config.trainml.run(found.disconnect())
    except:
        try:
            config.trainml.run(found.disconnect())
        except:
            pass
        raise


@checkpoint.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to checkpoint and show creation logs.",
)
@click.argument("checkpoint", type=click.STRING)
@pass_config
def connect(config, checkpoint, attach):
    """
    Connect local source to checkpoint and begin upload.

    CHECKPOINT may be specified by name or ID, but ID is preferred.
    """
    checkpoints = config.trainml.run(config.trainml.client.checkpoints.list())

    found = search_by_id_name(checkpoint, checkpoints)
    if None is found:
        raise click.UsageError("Cannot find specified checkpoint.")

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


@checkpoint.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to checkpoint and show creation logs.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect source and start checkpoint creation.",
)
@click.option(
    "--source",
    "-s",
    type=click.Choice(["local"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Dataset source type.",
)
@click.argument("name", type=click.STRING)
@click.argument(
    "path", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@pass_config
def create(config, attach, connect, source, name, path):
    """
    Create a checkpoint.

    A checkpoint is created with the specified NAME using a local source at the PATH
    specified. PATH should be a local directory containing the source data for
    a local source or a URI for all other source types.
    """

    if source == "local":
        checkpoint = config.trainml.run(
            config.trainml.client.checkpoints.create(
                name=name, source_type="local", source_uri=path
            )
        )

        try:
            if connect and attach:
                config.trainml.run(checkpoint.attach(), checkpoint.connect())
                return config.trainml.run(checkpoint.disconnect())
            elif connect:
                return config.trainml.run(checkpoint.connect())
            else:
                raise click.UsageError(
                    "Abort!\n"
                    "No logs to show for local sourced checkpoint without connect."
                )
        except:
            try:
                config.trainml.run(checkpoint.disconnect())
            except:
                pass
            raise


@checkpoint.command()
@click.argument("checkpoint", type=click.STRING)
@pass_config
def disconnect(config, checkpoint):
    """
    Disconnect and clean-up checkpoint upload.

    CHECKPOINT may be specified by name or ID, but ID is preferred.
    """
    checkpoints = config.trainml.run(config.trainml.client.checkpoints.list())

    found = search_by_id_name(checkpoint, checkpoints)
    if None is found:
        raise click.UsageError("Cannot find specified checkpoint.")

    return config.trainml.run(found.disconnect())


@checkpoint.command()
@pass_config
def list(config):
    """List checkpoints."""
    data = [
        ["ID", "STATUS", "NAME", "SIZE"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    checkpoints = config.trainml.run(config.trainml.client.checkpoints.list())

    for checkpoint in checkpoints:
        data.append(
            [
                checkpoint.id,
                checkpoint.status,
                checkpoint.name,
                pretty_size(checkpoint.size),
            ]
        )
    for row in data:
        click.echo(
            "{: >38.36} {: >13.11} {: >40.38} {: >14.12}" "".format(*row),
            file=config.stdout,
        )


@checkpoint.command()
@pass_config
def list_public(config):
    """List public checkpoints."""
    data = [
        ["ID", "STATUS", "NAME", "SIZE"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    checkpoints = config.trainml.run(
        config.trainml.client.checkpoints.list_public()
    )

    for ckpt in checkpoints:
        data.append(
            [
                ckpt.id,
                ckpt.status,
                ckpt.name,
                pretty_size(ckpt.size),
            ]
        )
    for row in data:
        click.echo(
            "{: >38.36} {: >13.11} {: >40.38} {: >14.12}" "".format(*row),
            file=config.stdout,
        )


@checkpoint.command()
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Force removal.",
)
@click.argument("checkpoint", type=click.STRING)
@pass_config
def remove(config, checkpoint, force):
    """
    Remove a checkpoint.

    CHECKPOINT may be specified by name or ID, but ID is preferred.
    """
    checkpoints = config.trainml.run(config.trainml.client.checkpoints.list())

    found = search_by_id_name(checkpoint, checkpoints)
    if None is found:
        if force:
            config.trainml.run(found.client.checkpoints.remove(checkpoint))
        else:
            raise click.UsageError("Cannot find specified checkpoint.")

    return config.trainml.run(found.remove(force=force))


@checkpoint.command()
@click.argument("checkpoint", type=click.STRING)
@click.argument("name", type=click.STRING)
@pass_config
def rename(config, checkpoint, name):
    """
    Renames a checkpoint.

    CHECKPOINT may be specified by name or ID, but ID is preferred.
    """
    try:
        checkpoint = config.trainml.run(
            config.trainml.client.checkpoints.get(checkpoint)
        )
        if checkpoint is None:
            raise click.UsageError("Cannot find specified checkpoint.")
    except:
        raise click.UsageError("Cannot find specified checkpoint.")

    return config.trainml.run(checkpoint.rename(name=name))
