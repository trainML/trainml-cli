import click
from trainml.cli import cli, pass_config


@cli.group()
@pass_config
def environment(config):
    """TrainML environment commands."""
    pass


@environment.command()
@pass_config
def list(config):
    """List environments."""
    data = [
        ["ID", "NAME", "PYTHON", "FRAMEWORK", "VERSION", "CUDA"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    environments = config.trainml.run(
        config.trainml.client.environments.list()
    )

    for env in environments:
        if env.id != "CUSTOM":
            data.append(
                [
                    env.id,
                    env.name,
                    env.py_version,
                    env.framework,
                    str(env.version),
                    env.cuda_version,
                ]
            )
    for row in data:
        click.echo(
            "{: >21.19} {: >30.28} {: >8.6} {: >15.13} {: >9.7} {: >6.4}"
            "".format(*row),
            file=config.stdout,
        )
