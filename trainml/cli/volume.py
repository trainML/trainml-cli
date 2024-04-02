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
def volume(config):
    """trainML volume commands."""
    pass


@volume.command()
@click.argument("volume", type=click.STRING)
@pass_config
def attach(config, volume):
    """
    Attach to volume and show creation logs.

    VOLUME may be specified by name or ID, but ID is preferred.
    """
    volumes = config.trainml.run(config.trainml.client.volumes.list())

    found = search_by_id_name(volume, volumes)
    if None is found:
        raise click.UsageError("Cannot find specified volume.")

    try:
        config.trainml.run(found.attach())
        return config.trainml.run(found.disconnect())
    except:
        try:
            config.trainml.run(found.disconnect())
        except:
            pass
        raise


@volume.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to volume and show creation logs.",
)
@click.argument("volume", type=click.STRING)
@pass_config
def connect(config, volume, attach):
    """
    Connect local source to volume and begin upload.

    VOLUME may be specified by name or ID, but ID is preferred.
    """
    volumes = config.trainml.run(config.trainml.client.volumes.list())

    found = search_by_id_name(volume, volumes)
    if None is found:
        raise click.UsageError("Cannot find specified volume.")

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


@volume.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to volume and show creation logs.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect source and start volume creation.",
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
@click.argument("capacity", type=click.INT)
@click.argument(
    "path", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@pass_config
def create(config, attach, connect, source, name, capacity, path):
    """
    Create a volume.

    A volume with maximum size CAPACITY is created with the specified NAME using a local source at the PATH
    specified. PATH should be a local directory containing the source data for
    a local source or a URI for all other source types.
    """

    if source == "local":
        volume = config.trainml.run(
            config.trainml.client.volumes.create(
                name=name, source_type="local", source_uri=path, capacity=capacity
            )
        )

        try:
            if connect and attach:
                config.trainml.run(volume.attach(), volume.connect())
                return config.trainml.run(volume.disconnect())
            elif connect:
                return config.trainml.run(volume.connect())
            else:
                raise click.UsageError(
                    "Abort!\n"
                    "No logs to show for local sourced volume without connect."
                )
        except:
            try:
                config.trainml.run(volume.disconnect())
            except:
                pass
            raise


@volume.command()
@click.argument("volume", type=click.STRING)
@pass_config
def disconnect(config, volume):
    """
    Disconnect and clean-up volume upload.

    VOLUME may be specified by name or ID, but ID is preferred.
    """
    volumes = config.trainml.run(config.trainml.client.volumes.list())

    found = search_by_id_name(volume, volumes)
    if None is found:
        raise click.UsageError("Cannot find specified volume.")

    return config.trainml.run(found.disconnect())


@volume.command()
@pass_config
def list(config):
    """List volumes."""
    data = [
        ["ID", "STATUS", "NAME", "CAPACITY"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    volumes = config.trainml.run(config.trainml.client.volumes.list())

    for volume in volumes:
        data.append(
            [
                volume.id,
                volume.status,
                volume.name,
                volume.capacity,
            ]
        )
    for row in data:
        click.echo(
            "{: >38.36} {: >13.11} {: >40.38} {: >14.12}" "".format(*row),
            file=config.stdout,
        )


@volume.command()
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Force removal.",
)
@click.argument("volume", type=click.STRING)
@pass_config
def remove(config, volume, force):
    """
    Remove a volume.

    VOLUME may be specified by name or ID, but ID is preferred.
    """
    volumes = config.trainml.run(config.trainml.client.volumes.list())

    found = search_by_id_name(volume, volumes)
    if None is found:
        if force:
            config.trainml.run(found.client.volumes.remove(volume))
        else:
            raise click.UsageError("Cannot find specified volume.")

    return config.trainml.run(found.remove(force=force))


@volume.command()
@click.argument("volume", type=click.STRING)
@click.argument("name", type=click.STRING)
@pass_config
def rename(config, volume, name):
    """
    Renames a volume.

    VOLUME may be specified by name or ID, but ID is preferred.
    """
    try:
        volume = config.trainml.run(config.trainml.client.volumes.get(volume))
        if volume is None:
            raise click.UsageError("Cannot find specified volume.")
    except:
        raise click.UsageError("Cannot find specified volume.")

    return config.trainml.run(volume.rename(name=name))
