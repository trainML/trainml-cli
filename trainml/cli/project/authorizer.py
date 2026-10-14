import json
import click
from trainml.cli import pass_config
from trainml.cli.project import project


@project.group()
@pass_config
def authorizer(config):
    """trainML project authorizer commands."""
    pass


@authorizer.command()
@pass_config
def list(config):
    """List authorizers."""
    data = [
        ["ID", "NAME", "TYPE", "UPDATED AT"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]
    project = config.trainml.run(config.trainml.client.projects.get_current())
    authorizers = config.trainml.run(project.authorizers.list())

    for authorizer in authorizers:
        data.append(
            [
                authorizer.id,
                authorizer.name,
                authorizer.type,
                authorizer.updated_at.isoformat(timespec="seconds"),
            ]
        )

    for row in data:
        click.echo(
            "{: >38.36} {: >38.36} {: >30.28} {: >28.26}" "".format(*row),
            file=config.stdout,
        )


@authorizer.command()
@click.argument("name", type=click.STRING)
@click.option(
    "--type",
    "authorizer_type",
    type=click.Choice(["api_key", "oidc"], case_sensitive=False),
    required=True,
    help="The authorizer type",
)
@click.option(
    "--client-id",
    type=click.STRING,
    default="",
    show_default=True,
    help="The client ID for the API key (api_key type only)",
)
@click.option(
    "--issuer",
    type=click.STRING,
    help="The OIDC issuer URL (oidc type only)",
)
@click.option(
    "--audience",
    type=click.STRING,
    help="The OIDC audience (oidc type only)",
)
@click.option(
    "--algorithms",
    type=click.STRING,
    help="Comma separated list of OIDC token signing algorithms (oidc type only)",
)
@click.option(
    "--jwks-uri",
    type=click.STRING,
    help="The JWKS URI for OIDC token validation (oidc type only)",
)
@click.option(
    "--required-claims",
    type=click.STRING,
    help="JSON object of required claims for the OIDC token (oidc type only)",
)
@pass_config
def add(
    config,
    name,
    authorizer_type,
    client_id,
    issuer,
    audience,
    algorithms,
    jwks_uri,
    required_claims,
):
    """
    Add an authorizer.

    Authorizer is created with the specified NAME.
    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    keys = None
    oidc = None

    if authorizer_type == "api_key":
        key = click.prompt("Enter the API key", type=str, hide_input=True)
        keys = [dict(client_id=client_id, key=key)]
    else:
        if not issuer:
            raise click.UsageError("--issuer is required for oidc authorizers")
        if not audience:
            raise click.UsageError("--audience is required for oidc authorizers")
        if not algorithms:
            raise click.UsageError("--algorithms is required for oidc authorizers")
        data = dict(
            issuer=issuer,
            audience=audience,
            algorithms=[algorithm.strip() for algorithm in algorithms.split(",")],
        )
        if jwks_uri:
            data["jwks_uri"] = jwks_uri
        if required_claims:
            try:
                data["required_claims"] = json.loads(required_claims)
            except json.JSONDecodeError:
                raise click.UsageError("--required-claims must be valid JSON")
        oidc = data

    return config.trainml.run(
        project.authorizers.create(
            name=name,
            type=authorizer_type,
            keys=keys,
            oidc=oidc,
        )
    )


@authorizer.command()
@click.argument("name_or_id", type=click.STRING)
@pass_config
def remove(config, name_or_id):
    """
    Remove an authorizer.

    AUTHORIZER may be specified by name or ID, but ID is preferred.
    """
    project = config.trainml.run(config.trainml.client.projects.get_current())

    authorizer = config.trainml.run(project.authorizers.get(name_or_id))
    if not authorizer:
        raise click.UsageError("Cannot find specified authorizer.")

    return config.trainml.run(project.authorizers.remove(authorizer.id))
