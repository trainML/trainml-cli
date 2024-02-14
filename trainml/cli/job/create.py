import click
import json
from webbrowser import open as browse
from trainml.cli import pass_config
from trainml.cli.job import job


def validate_cpu_count(ctx, param, value):
    if value is None:
        return
    try:
        count = int(value)
    except:
        raise click.BadParameter("must be integer")
    if count % 4 != 0:
        raise click.BadParameter("must be multiple of 4")

    return count


@job.group()
@pass_config
def create(config):
    """trainML job create."""
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
            "gtx1060",
            "gtx1080ti",
            "rtx2060s",
            "rtx2070s",
            "rtx2080ti",
            "rtx3090",
            "p100",
            "t4",
            "v100",
            "a100",
            "a100xl",
            "a6000",
            "a4000",
            "cpu",
        ],
        case_sensitive=False,
    ),
    default=["rtx3090"],
    show_default=True,
    multiple=True,
    help="GPU type.",
)
@click.option(
    "--cpu-count",
    "-cc",
    type=click.UNPROCESSED,
    callback=validate_cpu_count,
    help="CPU Count (per Worker).",
)
@click.option(
    "--max-price",
    "-mp",
    type=click.FLOAT,
    default=10.0,
    show_default=True,
    help="Max Price (per GPU).",
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
    "--checkpoint",
    type=click.STRING,
    help="ID or Name of a checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--public-checkpoint",
    type=click.STRING,
    help="ID or Name of a public checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--environment",
    type=click.Choice(
        [
            "DEEPLEARNING_PY39",
            "DEEPLEARNING_PY310",
            "PYTORCH_PY39_20",
            "PYTORCH_PY39_113",
            "PYTORCH_PY39_112",
            "TENSORFLOW_PY39_212",
            "TENSORFLOW_PY39_211",
            "TENSORFLOW_PY39_210",
            "MXNET_PY39_19",
        ],
        case_sensitive=False,
    ),
    default="DEEPLEARNING_PY310",
    show_default=True,
    help="Job environment to use",
)
@click.option(
    "--custom-image",
    type=click.STRING,
    help="Docker Image to use for the job.  Implies 'CUSTOM' environment type.",
)
@click.option(
    "--env",
    type=click.STRING,
    help="Environment variables to set in the job environment in 'KEY=VALUE' format",
    multiple=True,
)
@click.option(
    "--apt-packages",
    type=click.STRING,
    help="Apt packages to install as a comma separated list 'p1,p2=v2,p3'",
)
@click.option(
    "--pip-packages",
    type=click.STRING,
    help="Pip packages to install as a comma separated list 'p1,p2==v2,p3'",
)
@click.option(
    "--conda-packages",
    type=click.STRING,
    help="Conda packages to install as a comma separated list 'p1,\"p2=v2\",p3'",
)
@click.option(
    "--key",
    type=click.Choice(
        ["aws", "gcp", "kaggle", "azure", "wasabi"],
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
@click.option(
    "--timeout",
    type=click.INT,
    default=300,
    show_default=True,
    help="Seconds to wait for job to start",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def notebook(
    config,
    attach,
    connect,
    disk_size,
    gpu_count,
    gpu_type,
    cpu_count,
    max_price,
    data_dir,
    dataset,
    public_dataset,
    checkpoint,
    public_checkpoint,
    environment,
    custom_image,
    env,
    key,
    apt_packages,
    pip_packages,
    conda_packages,
    model_dir,
    git_uri,
    model_id,
    timeout,
    name,
):
    """
    Create a notebook.
    """

    datasets = [
        dict(
            id=item,
        )
        for item in dataset
    ] + [dict(id=item, public=True) for item in public_dataset]

    checkpoints = [
        dict(
            id=item,
        )
        for item in checkpoint
    ] + [dict(id=item, public=True) for item in public_checkpoint]

    options = dict(
        max_price=max_price,
        data=dict(datasets=datasets),
        model=dict(checkpoints=checkpoints),
        environment=dict(
            worker_key_types=[k for k in key],
        ),
    )

    if not (len(gpu_type) == 1 and gpu_type[0] == "cpu"):
        options["gpu_count"] = gpu_count

    if custom_image:
        options["environment"]["type"] = "CUSTOM"
        options["environment"]["custom_image"] = custom_image
    else:
        options["environment"]["type"] = environment

    try:
        envs = [{"key": e.split("=")[0], "value": e.split("=")[1]} for e in env]
        options["environment"]["env"] = envs
    except IndexError:
        raise click.UsageError(
            "Invalid environment variable format.  Must be in 'KEY=VALUE' format."
        )

    if apt_packages or pip_packages or conda_packages:
        options["environment"]["packages"] = dict()
        if apt_packages:
            options["environment"]["packages"]["apt"] = apt_packages.split(",")
        if pip_packages:
            options["environment"]["packages"]["pip"] = pip_packages.split(",")
        if conda_packages:
            options["environment"]["packages"]["conda"] = conda_packages.split(",")

    if data_dir:
        click.echo("Creating Dataset..", file=config.stdout)
        new_dataset = config.trainml.run(
            config.trainml.client.datasets.create(f"Job - {name}", "local", data_dir)
        )
        if attach:
            config.trainml.run(new_dataset.attach(), new_dataset.connect())
            config.trainml.run(new_dataset.disconnect())
        else:
            config.trainml.run(new_dataset.connect())
            config.trainml.run(new_dataset.wait_for("ready"))
            config.trainml.run(new_dataset.disconnect())
        options["data"]["datasets"].append(dict(id=new_dataset.id, type="existing"))

    if git_uri:
        options["model"]["source_type"] = "git"
        options["model"]["source_uri"] = git_uri
    if model_id:
        options["model"]["source_type"] = "trainml"
        options["model"]["source_uri"] = model_id
    if model_dir:
        options["model"]["source_type"] = "local"
        options["model"]["source_uri"] = model_dir

    job = config.trainml.run(
        config.trainml.client.jobs.create(
            name=name,
            type="notebook",
            gpu_types=gpu_type,
            cpu_count=cpu_count,
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
        config.trainml.run(job.wait_for("running", timeout))
        click.echo("Launching...", file=config.stdout)
        config.trainml.run(job.open())


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
            "gtx1060",
            "gtx1080ti",
            "rtx2060s",
            "rtx2070s",
            "rtx2080ti",
            "rtx3090",
            "p100",
            "t4",
            "v100",
            "a100",
            "a100xl",
            "a6000",
            "a4000",
            "cpu",
        ],
        case_sensitive=False,
    ),
    default=["rtx3090"],
    multiple=True,
    show_default=True,
    help="GPU type.",
)
@click.option(
    "--cpu-count",
    "-cc",
    type=click.UNPROCESSED,
    callback=validate_cpu_count,
    help="CPU Count (per Worker).",
)
@click.option(
    "--max-price",
    "-mp",
    type=click.FLOAT,
    default=10.0,
    show_default=True,
    help="Max Price (per GPU).",
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
    "--checkpoint",
    type=click.STRING,
    help="ID or Name of a checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--public-checkpoint",
    type=click.STRING,
    help="ID or Name of a public checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Local file path to save the output results to",
)
@click.option(
    "--output-type",
    type=click.Choice(
        ["aws", "gcp", "local", "trainml", "azure", "wasabi"],
        case_sensitive=False,
    ),
    help="Service provider to store the output data",
)
@click.option(
    "--output-uri",
    type=click.STRING,
    help="Location in the output-type provider to store the output data",
)
@click.option(
    "--archive/--no-archive",
    default=False,
    show_default=True,
    help="Zip the output contents before uploading.",
)
@click.option(
    "--save-model/--no-save-model",
    default=False,
    show_default=True,
    help="Include the model directory in the job output",
)
@click.option(
    "--environment",
    type=click.Choice(
        [
            "DEEPLEARNING_PY39",
            "DEEPLEARNING_PY310",
            "PYTORCH_PY39_20",
            "PYTORCH_PY39_113",
            "PYTORCH_PY39_112",
            "TENSORFLOW_PY39_212",
            "TENSORFLOW_PY39_211",
            "TENSORFLOW_PY39_210",
            "MXNET_PY39_19",
        ],
        case_sensitive=False,
    ),
    default="DEEPLEARNING_PY310",
    show_default=True,
    help="Job environment to use",
)
@click.option(
    "--custom-image",
    type=click.STRING,
    help="Docker Image to use for the job.  Implies 'CUSTOM' environment type.",
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
        ["aws", "gcp", "kaggle", "azure", "wasabi"],
        case_sensitive=False,
    ),
    help="Third Party Keys to add to the job environment",
    multiple=True,
)
@click.option(
    "--apt-packages",
    type=click.STRING,
    help="Apt packages to install as a comma separated list 'p1,p2=v2,p3'",
)
@click.option(
    "--pip-packages",
    type=click.STRING,
    help="Pip packages to install as a comma separated list 'p1,p2==v2,p3'",
)
@click.option(
    "--conda-packages",
    type=click.STRING,
    help="Conda packages to install as a comma separated list 'p1,\"p2=v2\",p3'",
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
@click.argument("name", type=click.STRING, required=True)
@click.argument("commands", type=click.STRING, nargs=-1, required=True)
@pass_config
def training(
    config,
    attach,
    connect,
    disk_size,
    gpu_count,
    gpu_type,
    cpu_count,
    max_price,
    data_dir,
    dataset,
    public_dataset,
    checkpoint,
    public_checkpoint,
    output_dir,
    output_type,
    output_uri,
    archive,
    save_model,
    environment,
    custom_image,
    env,
    key,
    apt_packages,
    pip_packages,
    conda_packages,
    model_dir,
    git_uri,
    model_id,
    name,
    commands,
):
    """
    Create a training job.
    """

    datasets = [
        dict(
            id=item,
        )
        for item in dataset
    ] + [dict(id=item, public=True) for item in public_dataset]

    checkpoints = [
        dict(
            id=item,
        )
        for item in checkpoint
    ] + [dict(id=item, public=True) for item in public_checkpoint]

    options = dict(
        max_price=max_price,
        data=dict(datasets=datasets),
        model=dict(checkpoints=checkpoints),
        environment=dict(
            worker_key_types=[k for k in key],
        ),
    )

    if not (len(gpu_type) == 1 and gpu_type[0] == "cpu"):
        options["gpu_count"] = gpu_count

    if custom_image:
        options["environment"]["type"] = "CUSTOM"
        options["environment"]["custom_image"] = custom_image
    else:
        options["environment"]["type"] = environment

    if output_type:
        options["data"]["output_type"] = output_type
        options["data"]["output_uri"] = output_uri
        options["data"]["output_options"] = dict(archive=archive, save_model=save_model)

    if output_dir:
        options["data"]["output_type"] = "local"
        options["data"]["output_uri"] = output_dir
        options["data"]["output_options"] = dict(archive=archive, save_model=save_model)

    try:
        envs = [{"key": e.split("=")[0], "value": e.split("=")[1]} for e in env]
        options["environment"]["env"] = envs
    except IndexError:
        raise click.UsageError(
            "Invalid environment variable format.  Must be in 'KEY=VALUE' format."
        )

    if apt_packages or pip_packages or conda_packages:
        options["environment"]["packages"] = dict()
        if apt_packages:
            options["environment"]["packages"]["apt"] = apt_packages.split(",")
        if pip_packages:
            options["environment"]["packages"]["pip"] = pip_packages.split(",")
        if conda_packages:
            options["environment"]["packages"]["conda"] = conda_packages.split(",")

    if data_dir:
        click.echo("Creating Dataset..", file=config.stdout)
        new_dataset = config.trainml.run(
            config.trainml.client.datasets.create(f"Job - {name}", "local", data_dir)
        )
        if attach:
            config.trainml.run(new_dataset.attach(), new_dataset.connect())
            config.trainml.run(new_dataset.disconnect())
        else:
            config.trainml.run(new_dataset.connect())
            config.trainml.run(new_dataset.wait_for("ready"))
            config.trainml.run(new_dataset.disconnect())
        options["data"]["datasets"].append(dict(id=new_dataset.id, type="existing"))

    if git_uri:
        options["model"]["source_type"] = "git"
        options["model"]["source_uri"] = git_uri
    if model_id:
        options["model"]["source_type"] = "trainml"
        options["model"]["source_uri"] = model_id
    if model_dir:
        options["model"]["source_type"] = "local"
        options["model"]["source_uri"] = model_dir

    job = config.trainml.run(
        config.trainml.client.jobs.create(
            name=name,
            type="training",
            gpu_types=gpu_type,
            cpu_count=cpu_count,
            disk_size=disk_size,
            workers=[command for command in commands],
            **options,
        )
    )
    click.echo("Created Job.", file=config.stdout)

    if connect or attach:
        config.trainml.run(job.wait_for("waiting for data/model download"))
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
            "gtx1060",
            "gtx1080ti",
            "rtx2060s",
            "rtx2070s",
            "rtx2080ti",
            "rtx3090",
            "p100",
            "t4",
            "v100",
            "a100",
            "a100xl",
            "a6000",
            "a4000",
            "cpu",
        ],
        case_sensitive=False,
    ),
    default=["rtx3090"],
    show_default=True,
    multiple=True,
    help="GPU type.",
)
@click.option(
    "--cpu-count",
    "-cc",
    type=click.UNPROCESSED,
    callback=validate_cpu_count,
    help="CPU Count (per Worker).",
)
@click.option(
    "--max-price",
    "-mp",
    type=click.FLOAT,
    default=10.0,
    show_default=True,
    help="Max Price (per GPU).",
)
@click.option(
    "--checkpoint",
    type=click.STRING,
    help="ID or Name of a checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--public-checkpoint",
    type=click.STRING,
    help="ID or Name of a public checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Local file path to copy as the input data",
)
@click.option(
    "--input-type",
    type=click.Choice(
        ["aws", "gcp", "local", "kaggle", "web", "azure", "wasabi"],
        case_sensitive=False,
    ),
    help="Service provider to load the input data",
)
@click.option(
    "--input-uri",
    type=click.STRING,
    help="Location in the input-type provider of the input data",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Local file path to save the output results to",
)
@click.option(
    "--output-type",
    type=click.Choice(
        ["aws", "gcp", "local", "trainml", "azure", "wasabi"],
        case_sensitive=False,
    ),
    help="Service provider to store the output data",
)
@click.option(
    "--output-uri",
    type=click.STRING,
    help="Location in the output-type provider to store the output data",
)
@click.option(
    "--archive/--no-archive",
    default=False,
    show_default=True,
    help="Zip the output contents before uploading.",
)
@click.option(
    "--save-model/--no-save-model",
    default=False,
    show_default=True,
    help="Include the model directory in the job output",
)
@click.option(
    "--environment",
    type=click.Choice(
        [
            "DEEPLEARNING_PY39",
            "DEEPLEARNING_PY310",
            "PYTORCH_PY39_20",
            "PYTORCH_PY39_113",
            "PYTORCH_PY39_112",
            "TENSORFLOW_PY39_212",
            "TENSORFLOW_PY39_211",
            "TENSORFLOW_PY39_210",
            "MXNET_PY39_19",
        ],
        case_sensitive=False,
    ),
    default="DEEPLEARNING_PY310",
    show_default=True,
    help="Job environment to use",
)
@click.option(
    "--custom-image",
    type=click.STRING,
    help="Docker Image to use for the job.  Implies 'CUSTOM' environment type.",
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
        ["aws", "gcp", "kaggle", "azure", "wasabi"],
        case_sensitive=False,
    ),
    help="Third Party Keys to add to the job environment",
    multiple=True,
)
@click.option(
    "--apt-packages",
    type=click.STRING,
    help="Apt packages to install as a comma separated list 'p1,p2=v2,p3'",
)
@click.option(
    "--pip-packages",
    type=click.STRING,
    help="Pip packages to install as a comma separated list 'p1,p2==v2,p3'",
)
@click.option(
    "--conda-packages",
    type=click.STRING,
    help="Conda packages to install as a comma separated list 'p1,\"p2=v2\",p3'",
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
@click.argument("name", type=click.STRING, required=True)
@click.argument("command", type=click.STRING, required=True)
@pass_config
def inference(
    config,
    attach,
    connect,
    disk_size,
    gpu_count,
    gpu_type,
    cpu_count,
    max_price,
    checkpoint,
    public_checkpoint,
    input_dir,
    input_type,
    input_uri,
    output_dir,
    output_type,
    output_uri,
    archive,
    save_model,
    environment,
    custom_image,
    env,
    key,
    apt_packages,
    pip_packages,
    conda_packages,
    model_dir,
    git_uri,
    model_id,
    name,
    command,
):
    """
    Create an inference job.
    """

    checkpoints = [
        dict(
            id=item,
        )
        for item in checkpoint
    ] + [dict(id=item, public=True) for item in public_checkpoint]

    options = dict(
        max_price=max_price,
        data=dict(datasets=[]),
        model=dict(checkpoints=checkpoints),
        environment=dict(
            worker_key_types=[k for k in key],
        ),
    )

    if not (len(gpu_type) == 1 and gpu_type[0] == "cpu"):
        options["gpu_count"] = gpu_count

    if custom_image:
        options["environment"]["type"] = "CUSTOM"
        options["environment"]["custom_image"] = custom_image
    else:
        options["environment"]["type"] = environment

    if input_type:
        options["data"]["input_type"] = input_type
        options["data"]["input_uri"] = input_uri

    if input_dir:
        options["data"]["input_type"] = "local"
        options["data"]["input_uri"] = input_dir

    if output_type:
        options["data"]["output_type"] = output_type
        options["data"]["output_uri"] = output_uri
        options["data"]["output_options"] = dict(archive=archive, save_model=save_model)

    if output_dir:
        options["data"]["output_type"] = "local"
        options["data"]["output_uri"] = output_dir
        options["data"]["output_options"] = dict(archive=archive, save_model=save_model)

    try:
        envs = [{"key": e.split("=")[0], "value": e.split("=")[1]} for e in env]
        options["environment"]["env"] = envs
    except IndexError:
        raise click.UsageError(
            "Invalid environment variable format.  Must be in 'KEY=VALUE' format."
        )

    if apt_packages or pip_packages or conda_packages:
        options["environment"]["packages"] = dict()
        if apt_packages:
            options["environment"]["packages"]["apt"] = apt_packages.split(",")
        if pip_packages:
            options["environment"]["packages"]["pip"] = pip_packages.split(",")
        if conda_packages:
            options["environment"]["packages"]["conda"] = conda_packages.split(",")

    if git_uri:
        options["model"]["source_type"] = "git"
        options["model"]["source_uri"] = git_uri
    if model_id:
        options["model"]["source_type"] = "trainml"
        options["model"]["source_uri"] = model_id
    if model_dir:
        options["model"]["source_type"] = "local"
        options["model"]["source_uri"] = model_dir

    job = config.trainml.run(
        config.trainml.client.jobs.create(
            name=name,
            type="inference",
            gpu_types=gpu_type,
            cpu_count=cpu_count,
            disk_size=disk_size,
            workers=[command],
            **options,
        )
    )
    click.echo("Created Job.", file=config.stdout)

    if connect or attach:
        config.trainml.run(job.wait_for("waiting for data/model download"))
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
            "gtx1060",
            "gtx1080ti",
            "rtx2060s",
            "rtx2070s",
            "rtx2080ti",
            "rtx3090",
            "p100",
            "t4",
            "v100",
            "a100",
            "a100xl",
            "a6000",
            "a4000",
            "cpu",
        ],
        case_sensitive=False,
    ),
    default=["rtx3090"],
    show_default=True,
    multiple=True,
    help="GPU type.",
)
@click.option(
    "--cpu-count",
    "-cc",
    type=click.UNPROCESSED,
    callback=validate_cpu_count,
    help="CPU Count (per Worker).",
)
@click.option(
    "--max-price",
    "-mp",
    type=click.FLOAT,
    default=10.0,
    show_default=True,
    help="Max Price (per GPU).",
)
@click.option(
    "--checkpoint",
    type=click.STRING,
    help="ID or Name of a checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--public-checkpoint",
    type=click.STRING,
    help="ID or Name of a public checkpoint to add to the job",
    multiple=True,
)
@click.option(
    "--environment",
    type=click.Choice(
        [
            "DEEPLEARNING_PY39",
            "DEEPLEARNING_PY310",
            "PYTORCH_PY39_20",
            "PYTORCH_PY39_113",
            "PYTORCH_PY39_112",
            "TENSORFLOW_PY39_212",
            "TENSORFLOW_PY39_211",
            "TENSORFLOW_PY39_210",
            "MXNET_PY39_19",
        ],
        case_sensitive=False,
    ),
    default="DEEPLEARNING_PY310",
    show_default=True,
    help="Job environment to use",
)
@click.option(
    "--custom-image",
    type=click.STRING,
    help="Docker Image to use for the job.  Implies 'CUSTOM' environment type.",
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
        ["aws", "gcp", "kaggle", "azure", "wasabi"],
        case_sensitive=False,
    ),
    help="Third Party Keys to add to the job environment",
    multiple=True,
)
@click.option(
    "--apt-packages",
    type=click.STRING,
    help="Apt packages to install as a comma separated list 'p1,p2=v2,p3'",
)
@click.option(
    "--pip-packages",
    type=click.STRING,
    help="Pip packages to install as a comma separated list 'p1,p2==v2,p3'",
)
@click.option(
    "--conda-packages",
    type=click.STRING,
    help="Conda packages to install as a comma separated list 'p1,\"p2=v2\",p3'",
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
@click.option(
    "--route",
    type=click.STRING,
    help="Routes to configure in endpoint (in JSON).",
    multiple=True,
)
@click.option(
    "--timeout",
    type=click.INT,
    default=300,
    show_default=True,
    help="Seconds to wait for job to start",
)
@click.argument("name", type=click.STRING, required=True)
@pass_config
def endpoint(
    config,
    attach,
    connect,
    disk_size,
    gpu_count,
    gpu_type,
    cpu_count,
    max_price,
    checkpoint,
    public_checkpoint,
    environment,
    custom_image,
    env,
    key,
    apt_packages,
    pip_packages,
    conda_packages,
    model_dir,
    git_uri,
    model_id,
    route,
    timeout,
    name,
):
    """
    Create an endpoint.
    """
    checkpoints = [
        dict(
            id=item,
        )
        for item in checkpoint
    ] + [dict(id=item, public=True) for item in public_checkpoint]

    routes = [json.loads(item) for item in route]

    options = dict(
        max_price=max_price,
        model=dict(checkpoints=checkpoints),
        environment=dict(
            worker_key_types=[k for k in key],
        ),
    )

    if not (len(gpu_type) == 1 and gpu_type[0] == "cpu"):
        options["gpu_count"] = gpu_count

    if custom_image:
        options["environment"]["type"] = "CUSTOM"
        options["environment"]["custom_image"] = custom_image
    else:
        options["environment"]["type"] = environment

    try:
        envs = [{"key": e.split("=")[0], "value": e.split("=")[1]} for e in env]
        options["environment"]["env"] = envs
    except IndexError:
        raise click.UsageError(
            "Invalid environment variable format.  Must be in 'KEY=VALUE' format."
        )

    if apt_packages or pip_packages or conda_packages:
        options["environment"]["packages"] = dict()
        if apt_packages:
            options["environment"]["packages"]["apt"] = apt_packages.split(",")
        if pip_packages:
            options["environment"]["packages"]["pip"] = pip_packages.split(",")
        if conda_packages:
            options["environment"]["packages"]["conda"] = conda_packages.split(",")

    if git_uri:
        options["model"]["source_type"] = "git"
        options["model"]["source_uri"] = git_uri
    if model_id:
        options["model"]["source_type"] = "trainml"
        options["model"]["source_uri"] = model_id
    if model_dir:
        options["model"]["source_type"] = "local"
        options["model"]["source_uri"] = model_dir

    job = config.trainml.run(
        config.trainml.client.jobs.create(
            name=name,
            type="endpoint",
            gpu_types=gpu_type,
            cpu_count=cpu_count,
            disk_size=disk_size,
            endpoint=dict(routes=routes),
            **options,
        )
    )
    click.echo("Created Job.", file=config.stdout)

    if model_dir:
        click.echo("Uploading model...", file=config.stdout)
        config.trainml.run(job.wait_for("waiting for data/model download"))
        if attach or connect:
            config.trainml.run(job.connect(), job.attach())
        else:
            config.trainml.run(job.connect())
        click.echo("Waiting for job to start...", file=config.stdout)
        config.trainml.run(job.wait_for("running", timeout))
        config.trainml.run(job.disconnect())
        config.trainml.run(job.refresh())
        click.echo(f"Endpoint is running at:  {job.url}", file=config.stdout)
    else:
        if connect:
            click.echo("Waiting for job to start...", file=config.stdout)
            config.trainml.run(job.wait_for("running", timeout))
            config.trainml.run(job.refresh())
            click.echo(f"Endpoint is running at:  {job.url}", file=config.stdout)
