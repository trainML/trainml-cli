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
def dataset(config):
    """TrainML dataset commands."""
    pass


@dataset.command()
@click.argument("dataset", type=click.STRING)
@pass_config
def attach(config, dataset):
    """
    Attach to dataset and show creation logs.

    DATASET may be specified by name or ID, but ID is preferred.
    """
    datasets = config.trainml.run(config.trainml.client.datasets.list())

    found = search_by_id_name(dataset, datasets)
    if None is found:
        raise click.UsageError("Cannot find specified dataset.")

    try:
        config.trainml.run(found.attach())
        return config.trainml.run(found.disconnect())
    except:
        try:
            config.trainml.run(found.disconnect())
        except:
            pass
        raise


@dataset.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to dataset and show creation logs.",
)
@click.argument("dataset", type=click.STRING)
@pass_config
def connect(config, dataset, attach):
    """
    Connect local source to dataset and begin upload.

    DATASET may be specified by name or ID, but ID is preferred.
    """
    datasets = config.trainml.run(config.trainml.client.datasets.list())

    found = search_by_id_name(dataset, datasets)
    if None is found:
        raise click.UsageError("Cannot find specified dataset.")

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


@dataset.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to dataset and show creation logs.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect source and start dataset creation.",
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
    Create a dataset.

    Dataset is created with the specified NAME using a local source at the PATH
    specified. PATH should be a local directory containing the source data for
    a local source or a URI for all other source types.
    """

    if source == "local":
        dataset = config.trainml.run(
            config.trainml.client.datasets.create(
                name=name, source_type="local", source_uri=path
            )
        )

        try:
            if connect and attach:
                config.trainml.run(dataset.attach(), dataset.connect())
                return config.trainml.run(dataset.disconnect())
            elif connect:
                return config.trainml.run(dataset.connect())
            else:
                raise click.UsageError(
                    "Abort!\n"
                    "No logs to show for local sourced dataset without connect."
                )
        except:
            try:
                config.trainml.run(dataset.disconnect())
            except:
                pass
            raise


@dataset.command()
@click.argument("dataset", type=click.STRING)
@pass_config
def disconnect(config, dataset):
    """
    Disconnect and clean-up dataset upload.

    DATASET may be specified by name or ID, but ID is preferred.
    """
    datasets = config.trainml.run(config.trainml.client.datasets.list())

    found = search_by_id_name(dataset, datasets)
    if None is found:
        raise click.UsageError("Cannot find specified dataset.")

    return config.trainml.run(found.disconnect())


@dataset.command()
@pass_config
def list(config):
    """List datasets."""
    data = [
        ["ID", "STATUS", "NAME", "SIZE"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    datasets = config.trainml.run(config.trainml.client.datasets.list())

    for dset in datasets:
        data.append(
            [
                dset.id,
                dset.status,
                dset.name,
                pretty_size(dset.size),
            ]
        )
    for row in data:
        click.echo(
            "{: >38.36} {: >13.11} {: >40.38} {: >14.12}" "".format(*row),
            file=config.stdout,
        )


@dataset.command()
@pass_config
def list_public(config):
    """List public datasets."""
    data = [
        ["ID", "STATUS", "NAME", "SIZE"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    datasets = config.trainml.run(config.trainml.client.datasets.list_public())

    for dset in datasets:
        data.append(
            [
                dset.id,
                dset.status,
                dset.name,
                pretty_size(dset.size),
            ]
        )
    for row in data:
        click.echo(
            "{: >38.36} {: >13.11} {: >40.38} {: >14.12}" "".format(*row),
            file=config.stdout,
        )


@dataset.command()
@click.argument("dataset", type=click.STRING)
@pass_config
def remove(config, dataset):
    """
    Remove a dataset.

    DATASET may be specified by name or ID, but ID is preferred.
    """
    datasets = config.trainml.run(config.trainml.client.datasets.list())

    found = search_by_id_name(dataset, datasets)
    if None is found:
        raise click.UsageError("Cannot find specified dataset.")

    return config.trainml.run(found.remove())
