import asyncio
import click
from trainml.cli import cli, pass_config
from trainml.trainml import TrainML


@cli.group()
@pass_config
def gpu(config):
    """TrainML GPU commands."""
    pass


@gpu.command()
@pass_config
def list(config):
    """List GPUs."""
    data = [
        ["ID", "NAME", "ABBRV", "CREDITS/HR MIN", "CREDITS/HR MAX"],
        ["-" * 80, "-" * 80, "-" * 80, "-" * 80, "-" * 80],
    ]

    gpus = config.trainml.run(config.trainml.client.gpu_types.list())

    for gpu in gpus:
        data.append(
            [
                gpu.id,
                gpu.name,
                gpu.abbrv,
                f"{gpu.credits_per_hour_min:.2f}",
                f"{gpu.credits_per_hour_max:.2f}",
            ]
        )
    for row in data:
        click.echo(
            "{: >36.34} {: >16.14} {: >10.8} {: >16.14} {: >16.14}"
            "".format(*row),
            file=config.stdout,
        )
