import click
from trainml.cli import cli, pass_config, search_by_id_name


def pretty_size(num):
    s = ("  B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    n = 0
    while num > 1023:
        num = num / 1024
        n += 1
    return f"{num:.2f} {s[n]}"


@cli.group()
@pass_config
def model(config):
    """TrainML model commands."""
    pass


@model.command()
@click.argument("model", type=click.STRING)
@pass_config
def attach(config, model):
    """
    Attach to model and show creation logs.

    MODEL may be specified by name or ID, but ID is preferred.
    """
    models = config.trainml.run(config.trainml.client.models.list())

    found = search_by_id_name(model, models)
    if None is found:
        raise click.UsageError("Cannot find specified model.")

    try:
        config.trainml.run(found.attach())
        return config.trainml.run(found.disconnect())
    except:
        try:
            config.trainml.run(found.disconnect())
        except:
            pass
        raise


@model.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to model and show creation logs.",
)
@click.argument("model", type=click.STRING)
@pass_config
def connect(config, model, attach):
    """
    Connect local source to model and begin upload.

    MODEL may be specified by name or ID, but ID is preferred.
    """
    models = config.trainml.run(config.trainml.client.models.list())

    found = search_by_id_name(model, models)
    if None is found:
        raise click.UsageError("Cannot find specified model.")

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


@model.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to model and show creation logs.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect source and start model creation.",
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
    Create a model.

    A model is created with the specified NAME using a local source at the PATH
    specified. PATH should be a local directory containing the source data for
    a local source or a URI for all other source types.
    """

    if source == "local":
        model = config.trainml.run(
            config.trainml.client.models.create(
                name=name, source_type="local", source_uri=path
            )
        )

        try:
            if connect and attach:
                config.trainml.run(model.attach(), model.connect())
                return config.trainml.run(model.disconnect())
            elif connect:
                return config.trainml.run(model.connect())
            else:
                raise click.UsageError(
                    "Abort!\n"
                    "No logs to show for local sourced model without connect."
                )
        except:
            try:
                config.trainml.run(model.disconnect())
            except:
                pass
            raise


@model.command()
@click.argument("model", type=click.STRING)
@pass_config
def disconnect(config, model):
    """
    Disconnect and clean-up model upload.

    MODEL may be specified by name or ID, but ID is preferred.
    """
    models = config.trainml.run(config.trainml.client.models.list())

    found = search_by_id_name(model, models)
    if None is found:
        raise click.UsageError("Cannot find specified model.")

    return config.trainml.run(found.disconnect())


@model.command()
@pass_config
def list(config):
    """List models."""
    data = [
        ["ID", "STATUS", "PROVIDER", "NAME", "SIZE"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    models = config.trainml.run(config.trainml.client.models.list())

    for model in models:
        data.append(
            [
                model.id,
                model.status,
                model.provider,
                model.name,
                pretty_size(model.size),
            ]
        )
    for row in data:
        click.echo(
            "{: >38.36} {: >13.11} {: >10.8} {: >40.38} {: >14.12}"
            "".format(*row),
            file=config.stdout,
        )
