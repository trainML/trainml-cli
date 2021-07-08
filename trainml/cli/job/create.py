import click
import json
from webbrowser import open as browse
from trainml.cli import pass_config
from trainml.cli.job import job


@job.group()
@pass_config
def create(config):
    """TrainML job create."""
    pass


@create.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to job.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect to job.",
)
@click.option(
    "--disk-size",
    "-ds",
    type=click.INT,
    default=10,
    show_default=True,
    help="Disk size (GiB).",
)
@click.option(
    "--gpu-count",
    "-gc",
    type=click.INT,
    default=1,
    show_default=True,
    help="GPU Count (per Worker).",
)
@click.option(
    "--gpu-type",
    "-gt",
    type=click.Choice(
        [
            "GTX 1060",
            "RTX 2060 Super",
            "RTX 2070 Super",
            "RTX 2080 Ti",
            "RTX 3090",
            "K80",
            "P100",
            "T4",
            "V100",
            "A100",
        ],
        case_sensitive=False,
    ),
    default="RTX 2080 Ti",
    show_default=True,
    help="GPU type.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Local file path to copy as the input data",
)
@click.option(
    "--dataset",
    type=click.STRING,
    help="ID or Name of a dataset to add to the job",
    multiple=True,
)
@click.option(
    "--public-dataset",
    type=click.STRING,
    help="ID or Name of a public dataset to add to the job",
    multiple=True,
)
@click.option(
    "--environment",
    type=click.Choice(
        [
            "DEEPLEARNING_PY38",
            "DEEPLEARNING_PY37",
            "PYTORCH_PY38_18",
            "PYTORCH_PY38_17",
            "PYTORCH_PY37_17",
            "PYTORCH_PY37_16",
            "PYTORCH_PY37_15",
            "TENSORFLOW_PY38_24",
            "TENSORFLOW_PY37_23",
            "TENSORFLOW_PY37_22",
            "TENSORFLOW_PY37_114",
            "MXNET_PY38_18",
            "MXNET_PY38_17",
            "MXNET_PY37_16",
        ],
        case_sensitive=False,
    ),
    default="DEEPLEARNING_PY38",
    show_default=True,
    help="Job environment to use",
)
@click.option(
    "--env",
    type=click.STRING,
    help="Environment variables to set in the job environment in 'KEY=VALUE' format",
    multiple=True,
)
@click.option(
    "--key",
    type=click.Choice(
        [
            "aws",
            "gcp",
            "kaggle",
        ],
        case_sensitive=False,
    ),
    help="Third Party Keys to add to the job environment",
    multiple=True,
)
@click.option(
    "--git-uri",
    type=click.STRING,
    help="Git repository to use as the model data",
)
@click.option(
    "--model-id",
    type=click.STRING,
    help="trainML Model ID to use as the model data",
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Local file path to copy as the model data",
)
@click.argument("name", type=click.STRING)
@pass_config
def notebook(
    config,
    attach,
    connect,
    disk_size,
    gpu_count,
    gpu_type,
    data_dir,
    dataset,
    public_dataset,
    environment,
    env,
    key,
    model_dir,
    git_uri,
    model_id,
    name,
):
    """
    Create a notebook.
    """

    datasets = [dict(id=item, type="existing") for item in dataset] + [
        dict(id=item, type="public") for item in public_dataset
    ]

    options = dict(
        data=dict(datasets=datasets),
        environment=dict(type=environment, worker_key_types=[k for k in key]),
    )

    try:
        envs = [
            {"key": e.split("=")[0], "value": e.split("=")[1]} for e in env
        ]
        options["environment"]["env"] = envs
    except IndexError:
        raise click.UsageError(
            "Invalid environment variable format.  Must be in 'KEY=VALUE' format."
        )

    if data_dir:
        click.echo("Creating Dataset..", file=config.stdout)
        new_dataset = config.trainml.run(
            config.trainml.client.datasets.create(
                f"Job - {name}", "local", data_dir
            )
        )
        if attach:
            config.trainml.run(new_dataset.attach(), new_dataset.connect())
            config.trainml.run(new_dataset.disconnect())
        else:
            config.trainml.run(new_dataset.connect())
            config.trainml.run(new_dataset.wait_for("ready"))
            config.trainml.run(new_dataset.disconnect())
        options["data"]["datasets"].append(
            dict(id=new_dataset.id, type="existing")
        )

    if git_uri:
        options["model"] = dict(source_type="git", source_uri=git_uri)
    if model_id:
        options["model"] = dict(source_type="trainml", source_uri=model_id)
    if model_dir:
        options["model"] = dict(source_type="local", source_uri=model_dir)
    job = config.trainml.run(
        config.trainml.client.jobs.create(
            name=name,
            type="notebook",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            disk_size=disk_size,
            **options,
        )
    )
    click.echo("Created Job.", file=config.stdout)
    if model_dir:
        config.trainml.run(job.wait_for("waiting for data/model download"))
        if attach or connect:
            click.echo("Waiting for job to start...", file=config.stdout)
            config.trainml.run(job.connect(), job.attach())
            config.trainml.run(job.disconnect())
            click.echo("Launching...", file=config.stdout)
            browse(job.notebook_url)
        else:
            config.trainml.run(job.connect())
            config.trainml.run(job.wait_for("running"))
            config.trainml.run(job.disconnect())
    elif attach or connect:
        click.echo("Waiting for job to start...", file=config.stdout)
        config.trainml.run(job.wait_for("running"))
        click.echo("Launching...", file=config.stdout)
        browse(job.notebook_url)


@create.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to job.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect to job.",
)
@click.option(
    "--disk-size",
    "-ds",
    type=click.INT,
    default=10,
    show_default=True,
    help="Disk size (GiB).",
)
@click.option(
    "--gpu-count",
    "-gc",
    type=click.INT,
    default=1,
    show_default=True,
    help="GPU Count (per Worker.)",
)
@click.option(
    "--gpu-type",
    "-gt",
    type=click.Choice(
        [
            "GTX 1060",
            "RTX 2060 Super",
            "RTX 2070 Super",
            "RTX 2080 Ti",
            "RTX 3090",
            "K80",
            "P100",
            "T4",
            "V100",
            "A100",
        ],
        case_sensitive=False,
    ),
    default="RTX 2080 Ti",
    show_default=True,
    help="GPU type.",
)
@click.argument("name", type=click.STRING)
@pass_config
def training(config, attach, connect, disk_size, gpu_count, gpu_type, name):
    """
    Create a training job.
    """

    job = config.trainml.run(
        config.trainml.client.jobs.create(
            name=name,
            type="training",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            disk_size=disk_size,
        )
    )
    click.echo("Created.", file=config.stdout)
    if connect:
        config.trainml.run(job.connect())
    if attach:
        config.trainml.run(job.attach())


@create.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to job.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect to job.",
)
@click.option(
    "--disk-size",
    "-ds",
    type=click.INT,
    default=10,
    show_default=True,
    help="Disk size (GiB).",
)
@click.option(
    "--gpu-count",
    "-gc",
    type=click.INT,
    default=1,
    show_default=True,
    help="GPU Count (per Worker.)",
)
@click.option(
    "--gpu-type",
    "-gt",
    type=click.Choice(
        [
            "GTX 1060",
            "RTX 2060 Super",
            "RTX 2070 Super",
            "RTX 2080 Ti",
            "RTX 3090",
            "K80",
            "P100",
            "T4",
            "V100",
            "A100",
        ],
        case_sensitive=False,
    ),
    default="RTX 2080 Ti",
    show_default=True,
    help="GPU type.",
)
@click.argument("name", type=click.STRING)
@pass_config
def inference(config, attach, connect, disk_size, gpu_count, gpu_type, name):
    """
    Create an inference job.
    """

    job = config.trainml.run(
        config.trainml.client.jobs.create(
            name=name,
            type="inference",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            disk_size=disk_size,
        )
    )
    click.echo("Created.", file=config.stdout)
    if connect:
        config.trainml.run(job.connect())
    if attach:
        config.trainml.run(job.attach())


@create.command()
@click.option(
    "--attach/--no-attach",
    default=True,
    show_default=True,
    help="Auto attach to job.",
)
@click.option(
    "--connect/--no-connect",
    default=True,
    show_default=True,
    help="Auto connect to job.",
)
@click.argument("file", type=click.File("rb"))
@pass_config
def from_json(config, attach, connect, file):
    """
    Create an job from json file representation.
    """
    payload = json.loads(file.read())

    job = config.trainml.run(config.trainml.client.jobs.create_json(payload))
    click.echo("Created.", file=config.stdout)
    if attach or connect:
        if job.type == "notebook":
            click.echo("Waiting for job to start...", file=config.stdout)
            config.trainml.run(job.wait_for("running"))
            click.echo("Launching...", file=config.stdout)
            browse(job.notebook_url)
        else:
            if connect:
                config.trainml.run(job.connect())
            if attach:
                config.trainml.run(job.attach())
