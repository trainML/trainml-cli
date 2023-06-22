import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender


@cloudbender.group()
@pass_config
def provider(config):
    """trainML CloudBender provider commands."""
    pass


@provider.command()
@pass_config
def list(config):
    """List providers."""
    data = [
        ["ID", "TYPE", "CREDITS"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    providers = config.trainml.run(
        config.trainml.client.cloudbender.providers.list()
    )

    for provider in providers:
        data.append(
            [
                provider.id,
                provider.type,
                f"{provider.credits}",
            ]
        )

    for row in data:
        click.echo(
            "{: >38.36} {: >30.28} {: >15.13}" "".format(*row),
            file=config.stdout,
        )


@provider.command()
@click.argument("type", type=click.STRING)
@pass_config
def enable(config, type):
    """
    Enables a provider.

    Provider is created of the specified type.
    """

    return config.trainml.run(
        config.trainml.client.cloudbender.providers.enable(
            type=type,
        )
    )


@provider.command()
@click.argument("provider", type=click.STRING)
@pass_config
def remove(config, provider):
    """
    Remove a provider.

    PROVIDER may be specified by name or ID, but ID is preferred.
    """
    providers = config.trainml.run(
        config.trainml.client.cloudbender.providers.list()
    )

    found = search_by_id_name(provider, providers)
    if None is found:
        raise click.UsageError("Cannot find specified provider.")

    return config.trainml.run(found.remove())
