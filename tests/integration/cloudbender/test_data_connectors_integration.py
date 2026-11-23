import re
from pytest import mark, fixture

pytestmark = [
    mark.sdk,
    mark.integration,
    mark.cloudbender,
    mark.data_connectors,
]

# Fake values only - nothing here points at a real host or credential.
FAKE_HOST = "192.0.2.30"  # RFC 5737 TEST-NET-1 (documentation range)
FAKE_CIDR = "192.0.2.0/24"
FAKE_TOKEN_URL = "http://192.0.2.30:8080/oauth/token"
FAKE_USERNAME = "fake-user"
FAKE_PASSWORD = "fake-password-000000"
FAKE_API_KEY = "fake-api-key-0000000000000000"
FAKE_CLIENT_ID = "fake-client"
FAKE_CLIENT_SECRET = "fake-client-secret-000000"

# Fake self-signed PEM blocks. They are well-formed enough to pass any
# transport/shape handling but do not correspond to a real key pair.
FAKE_CA_CERT = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkTCB+wIJAeKZ2rLc9f0XyZ6hQp1oWk3sVd2E\n"
    "-----END CERTIFICATE-----\n"
)
FAKE_CLIENT_CERT = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBkjCB+wIJAeKZ2rLc9f0XyZ6hQp1oWk3sVd2E\n"
    "-----END CERTIFICATE-----\n"
)
FAKE_PRIVATE_KEY = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MHQCAQEEIBfakePrivateKey000000000000000000000000\n"
    "-----END PRIVATE KEY-----\n"
)


@mark.create
@mark.asyncio
@mark.xdist_group("cloudbender_resources")
class GetDataConnectorTests:
    @fixture(scope="class")
    async def custom_connector(self, trainml, region):
        connector = await trainml.cloudbender.data_connectors.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI-Automated-Custom-Connector",
            type="custom",
            protocol="tcp",
            port_range="8000:8099",
            cidr=FAKE_CIDR,
        )
        yield connector
        await connector.remove()

    @fixture(scope="class")
    async def http_connector(self, trainml, region):
        connector = await trainml.cloudbender.data_connectors.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI-Automated-HTTP-Connector",
            type="http",
            host=FAKE_HOST,
            port=8080,
            auth_type="api_key",
            api_key=FAKE_API_KEY,
        )
        yield connector
        await connector.remove()

    @fixture(scope="class")
    async def pg_connector(self, trainml, region):
        connector = await trainml.cloudbender.data_connectors.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI-Automated-PG-Connector",
            type="pg",
            host=FAKE_HOST,
            port=5432,
            database="testdb",
            auth_type="password",
            username=FAKE_USERNAME,
            password=FAKE_PASSWORD,
        )
        yield connector
        await connector.remove()

    @fixture(scope="class")
    async def redis_connector(self, trainml, region):
        connector = await trainml.cloudbender.data_connectors.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI-Automated-Redis-Connector",
            type="redis",
            host=FAKE_HOST,
            port=6379,
            database="0",
            auth_type="certificate",
            ca_certificate=FAKE_CA_CERT,
            client_certificate=FAKE_CLIENT_CERT,
            client_private_key=FAKE_PRIVATE_KEY,
        )
        yield connector
        await connector.remove()

    @fixture(scope="class")
    async def http_no_tls_connector(self, trainml, region):
        connector = await trainml.cloudbender.data_connectors.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI-Automated-HTTP-NoTLS-Connector",
            type="http",
            host=FAKE_HOST,
            tls=False,
            auth_type="api_key",
            api_key=FAKE_API_KEY,
            authorization_header="X-Api-Key",
            authorization_prefix="Token ",
        )
        yield connector
        await connector.remove()

    async def test_get_data_connectors(self, trainml, region, custom_connector):
        connectors = await trainml.cloudbender.data_connectors.list(
            provider_uuid=region.provider_uuid, region_uuid=region.id
        )
        assert any(c.id == custom_connector.id for c in connectors)

    async def test_get_data_connector(
        self, trainml, provider, region, custom_connector
    ):
        response = await trainml.cloudbender.data_connectors.get(
            provider.id, region.id, custom_connector.id
        )
        assert response.id == custom_connector.id

    async def test_custom_connector_properties(self, region, custom_connector):
        assert isinstance(custom_connector.id, str)
        assert isinstance(custom_connector.provider_uuid, str)
        assert isinstance(custom_connector.region_uuid, str)
        assert isinstance(custom_connector.name, str)
        assert isinstance(custom_connector.type, str)
        assert custom_connector.type == "custom"
        assert custom_connector.provider_uuid == region.provider_uuid
        assert custom_connector.region_uuid == region.id
        data = custom_connector._data_connector
        assert data.get("protocol") == "tcp"
        assert data.get("port_range") == "8000:8099"
        assert data.get("cidr") == FAKE_CIDR

    async def test_http_connector_properties(self, region, http_connector):
        assert http_connector.type == "http"
        assert http_connector.provider_uuid == region.provider_uuid
        assert http_connector.region_uuid == region.id
        data = http_connector._data_connector
        assert data.get("host") == FAKE_HOST
        assert data.get("port") == 8080
        assert data.get("auth_type") == "api_key"
        assert "api_key" not in data

    async def test_http_no_tls_connector_properties(
        self, region, http_no_tls_connector
    ):
        connector = http_no_tls_connector
        assert connector.type == "http"
        assert connector.tls is False
        data = connector._data_connector
        assert data.get("host") == FAKE_HOST
        assert data.get("port") == 80
        assert data.get("tls") is False
        assert data.get("auth_type") == "api_key"
        assert data.get("authorization_header") == "X-Api-Key"
        assert data.get("authorization_prefix") == "Token "
        assert "api_key" not in data

    async def test_edit_data_connector_tls(
        self, trainml, region, http_no_tls_connector
    ):
        # Re-enable TLS and set the port explicitly, then restore.
        updated = await http_no_tls_connector.edit(tls=True, port=8080)
        assert updated.tls is True
        assert updated.port == 8080
        refreshed = await trainml.cloudbender.data_connectors.get(
            region.provider_uuid, region.id, http_no_tls_connector.id
        )
        assert refreshed.tls is True
        assert refreshed.port == 8080
        await refreshed.edit(tls=False, port=80)

    async def test_pg_connector_properties(self, region, pg_connector):
        assert pg_connector.type == "pg"
        assert pg_connector.provider_uuid == region.provider_uuid
        assert pg_connector.region_uuid == region.id
        data = pg_connector._data_connector
        assert data.get("host") == FAKE_HOST
        assert data.get("port") == 5432
        assert data.get("database") == "testdb"
        assert data.get("auth_type") == "password"
        assert data.get("username") == FAKE_USERNAME
        assert "password" not in data

    async def test_redis_connector_properties(self, region, redis_connector):
        assert redis_connector.type == "redis"
        assert redis_connector.provider_uuid == region.provider_uuid
        assert redis_connector.region_uuid == region.id
        data = redis_connector._data_connector
        assert data.get("host") == FAKE_HOST
        assert data.get("port") == 6379
        assert data.get("database") == "0"
        assert data.get("auth_type") == "certificate"
        assert data.get("ca_certificate") == FAKE_CA_CERT
        assert data.get("client_certificate") == FAKE_CLIENT_CERT
        assert "client_private_key" not in data

    async def test_edit_data_connector(self, trainml, region, http_connector):
        updated = await http_connector.edit(port=9090)
        assert updated.port == 9090
        # restore original port so teardown/cleanup stays consistent
        await http_connector.edit(port=8080)
        refreshed = await trainml.cloudbender.data_connectors.get(
            region.provider_uuid, region.id, http_connector.id
        )
        assert refreshed.port == 8080

    async def test_data_connector_str(self, custom_connector):
        string = str(custom_connector)
        regex = r"^{.*\"connector_id\": \"" + custom_connector.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_data_connector_repr(self, custom_connector):
        string = repr(custom_connector)
        regex = (
            r"^DataConnector\( trainml , \*\*{.*'connector_id': '"
            + custom_connector.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
