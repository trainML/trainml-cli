import click
from trainml.cli import cli, pass_config, search_by_id_name
from trainml.cli.cloudbender import cloudbender

CONNECTION_TYPES = ["custom", "http", "pg", "redis"]
AUTH_TYPES = [
    "password",
    "api_key",
    "certificate",
    "oauth_client_secret",
    "oauth_private_key",
]


def _read_file_option(value, file_value, option_name):
    """Resolve a credential that can be given inline or as a file path."""
    if value is not None and file_value is not None:
        raise click.UsageError(
            f"{option_name} and {option_name}-file may not both be specified."
        )
    if file_value is not None:
        try:
            with open(file_value) as file:
                return file.read()
        except OSError as e:
            raise click.UsageError(f"Unable to read {option_name}-file: {e}")
    return value


def _connector_fields(
    host,
    port,
    database,
    protocol,
    port_range,
    cidr,
    auth_type,
    username,
    password,
    api_key,
    client_id,
    client_secret,
    token_url,
    scopes,
    ca_certificate,
    client_certificate,
    client_private_key,
    tls,
    authorization_header,
    authorization_prefix,
    ca_certificate_file=None,
    client_certificate_file=None,
    client_private_key_file=None,
):
    return dict(
        host=host,
        port=port,
        tls=tls,
        database=database,
        protocol=protocol,
        port_range=port_range,
        cidr=cidr,
        auth_type=auth_type,
        authorization_header=authorization_header,
        authorization_prefix=authorization_prefix,
        username=username,
        password=password,
        api_key=api_key,
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url,
        scopes=scopes,
        ca_certificate=_read_file_option(
            ca_certificate, ca_certificate_file, "ca_certificate"
        ),
        client_certificate=_read_file_option(
            client_certificate, client_certificate_file, "client_certificate"
        ),
        client_private_key=_read_file_option(
            client_private_key, client_private_key_file, "client_private_key"
        ),
    )


def _common_decorators():
    def decorator(func):
        func = click.option(
            "--region",
            "-r",
            type=click.STRING,
            required=True,
            help="The region ID.",
        )(func)
        func = click.option(
            "--provider",
            "-p",
            type=click.STRING,
            required=True,
            help="The provider ID of the region.",
        )(func)
        return func

    return decorator


def _connection_decorators(include_auth_type):
    def decorator(func):
        func = click.option(
            "--client-private-key-file",
            type=click.STRING,
            help="A path to a file containing the client private key, PEM encoded.",
        )(func)
        func = click.option(
            "--client-private-key",
            type=click.STRING,
            help="The client private key, PEM encoded.",
        )(func)
        func = click.option(
            "--client-certificate-file",
            type=click.STRING,
            help="A path to a file containing the client certificate, PEM encoded.",
        )(func)
        func = click.option(
            "--client-certificate",
            type=click.STRING,
            help="The client certificate, PEM encoded (certificate auth).",
        )(func)
        func = click.option(
            "--ca-certificate-file",
            type=click.STRING,
            help="A path to a file containing the CA certificate, PEM encoded.",
        )(func)
        func = click.option(
            "--ca-certificate",
            type=click.STRING,
            help="The CA certificate, PEM encoded (certificate auth).",
        )(func)
        func = click.option(
            "--scopes",
            type=click.STRING,
            help="The OAuth scopes, space- or comma-separated.",
        )(func)
        func = click.option(
            "--token-url",
            type=click.STRING,
            help="The OAuth token endpoint URL.",
        )(func)
        func = click.option(
            "--client-secret",
            type=click.STRING,
            help="The OAuth client secret (oauth_client_secret auth).",
        )(func)
        func = click.option(
            "--client-id",
            type=click.STRING,
            help="The OAuth client ID.",
        )(func)
        func = click.option(
            "--authorization-prefix",
            type=click.STRING,
            help='The prefix for the credential value, e.g. "Bearer " (api_key, OAuth auth).',
        )(func)
        func = click.option(
            "--authorization-header",
            type=click.STRING,
            help="The header name carrying the credential (api_key, OAuth auth).",
        )(func)
        func = click.option(
            "--api-key",
            type=click.STRING,
            help="The API key (api_key auth).",
        )(func)
        func = click.option(
            "--password",
            type=click.STRING,
            help="The password (password auth).",
        )(func)
        func = click.option(
            "--username",
            type=click.STRING,
            help="The username (password auth).",
        )(func)
        if include_auth_type:
            func = click.option(
                "--auth-type",
                type=click.Choice(AUTH_TYPES, case_sensitive=False),
                help="The authentication type of the data connector.",
            )(func)
        func = click.option(
            "--cidr",
            type=click.STRING,
            help="The IP range to allow, in CIDR notation (custom connectors).",
        )(func)
        func = click.option(
            "--port-range",
            type=click.STRING,
            help="The port range (custom connectors).",
        )(func)
        func = click.option(
            "--protocol",
            type=click.STRING,
            help="The transport protocol (custom connectors).",
        )(func)
        func = click.option(
            "--database",
            type=click.STRING,
            help="The database name or logical database index.",
        )(func)
        func = click.option(
            "--port",
            type=click.INT,
            help="The port of the data connector target.",
        )(func)
        func = click.option(
            "--host",
            type=click.STRING,
            help="The hostname or IP address of the data connector target.",
        )(func)
        func = click.option(
            "--tls/--no-tls",
            "tls",
            default=None,
            help="Enable TLS (default: enabled). Use --no-tls to disable for http connectors.",
        )(func)
        return func

    return decorator


@cloudbender.group()
@pass_config
def data_connector(config):
    """trainML CloudBender data connector commands."""
    pass


@data_connector.command()
@_common_decorators()
@pass_config
def list(config, provider, region):
    """List data connectors."""
    data = [
        ["ID", "NAME", "TYPE", "HOST", "AUTH_TYPE", "PROTOCOL", "TLS"],
        [
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
            "-" * 80,
        ],
    ]

    data_connectors = config.trainml.run(
        config.trainml.client.cloudbender.data_connectors.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    for data_connector in data_connectors:
        tls = data_connector.tls
        data.append(
            [
                data_connector.id or "",
                data_connector.name or "",
                data_connector.type or "",
                data_connector.host or "",
                data_connector.auth_type or "",
                data_connector.protocol or "",
                "" if tls is None else str(tls).lower(),
            ]
        )

    for row in data:
        click.echo(
            "{: >37.36} {: >29.28} {: >9.8} {: >30.29} {: >16.15} {: >9.8} {: >5.5}".format(
                *row
            ),
            file=config.stdout,
        )


@data_connector.command()
@_common_decorators()
@click.option(
    "--type",
    "-t",
    type=click.Choice(CONNECTION_TYPES, case_sensitive=False),
    required=True,
    help="The type of data connector to create.",
)
@click.argument("name", type=click.STRING, required=True)
@_connection_decorators(True)
@pass_config
def create(
    config,
    provider,
    region,
    type,
    name,
    host,
    port,
    tls,
    database,
    protocol,
    port_range,
    cidr,
    auth_type,
    username,
    password,
    api_key,
    authorization_header,
    authorization_prefix,
    client_id,
    client_secret,
    token_url,
    scopes,
    ca_certificate,
    ca_certificate_file,
    client_certificate,
    client_certificate_file,
    client_private_key,
    client_private_key_file,
):
    """
    Creates a data_connector.
    """
    fields = _connector_fields(
        host,
        port,
        database,
        protocol,
        port_range,
        cidr,
        auth_type,
        username,
        password,
        api_key,
        client_id,
        client_secret,
        token_url,
        scopes,
        ca_certificate,
        client_certificate,
        client_private_key,
        tls,
        authorization_header,
        authorization_prefix,
        ca_certificate_file=ca_certificate_file,
        client_certificate_file=client_certificate_file,
        client_private_key_file=client_private_key_file,
    )
    return config.trainml.run(
        config.trainml.client.cloudbender.data_connectors.create(
            provider_uuid=provider,
            region_uuid=region,
            name=name,
            type=type,
            **fields,
        )
    )


@data_connector.command()
@_common_decorators()
@click.argument("data_connector", type=click.STRING, required=True)
@_connection_decorators(False)
@pass_config
def edit(
    config,
    provider,
    region,
    data_connector,
    host,
    port,
    tls,
    database,
    protocol,
    port_range,
    cidr,
    username,
    password,
    api_key,
    authorization_header,
    authorization_prefix,
    client_id,
    client_secret,
    token_url,
    scopes,
    ca_certificate,
    ca_certificate_file,
    client_certificate,
    client_certificate_file,
    client_private_key,
    client_private_key_file,
):
    """
    Edit a data_connector.

    DATA_CONNECTOR may be specified by name or ID, but ID is preferred.
    Only the options provided are changed. The connector type and auth
    type cannot be edited.
    """
    connectors = config.trainml.run(
        config.trainml.client.cloudbender.data_connectors.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    found = search_by_id_name(data_connector, connectors)
    if None is found:
        raise click.UsageError("Cannot find specified data_connector.")

    fields = _connector_fields(
        host,
        port,
        database,
        protocol,
        port_range,
        cidr,
        None,
        username,
        password,
        api_key,
        client_id,
        client_secret,
        token_url,
        scopes,
        ca_certificate,
        client_certificate,
        client_private_key,
        tls,
        authorization_header,
        authorization_prefix,
        ca_certificate_file=ca_certificate_file,
        client_certificate_file=client_certificate_file,
        client_private_key_file=client_private_key_file,
    )
    return config.trainml.run(found.edit(**fields))


@data_connector.command()
@_common_decorators()
@click.argument("data_connector", type=click.STRING)
@pass_config
def remove(config, provider, region, data_connector):
    """
    Remove a data_connector.

    DATA_CONNECTOR may be specified by name or ID, but ID is preferred.
    """
    data_connectors = config.trainml.run(
        config.trainml.client.cloudbender.data_connectors.list(
            provider_uuid=provider, region_uuid=region
        )
    )

    found = search_by_id_name(data_connector, data_connectors)
    if None is found:
        raise click.UsageError("Cannot find specified data_connector.")

    return config.trainml.run(found.remove())
