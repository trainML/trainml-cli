import asyncio
import io
from unittest.mock import AsyncMock, Mock
from pytest import mark, fixture

pytestmark = [mark.cli, mark.unit, mark.cloudbender, mark.data_connectors]

from trainml.cli import Config, TrainMLRunner
from trainml.cli.cloudbender.data_connector import (
    data_connector as data_connector_cmd,
)
from trainml.cloudbender import Cloudbender
from trainml.cloudbender.data_connectors import (
    DataConnectors,
)
from trainml.trainml import TrainML


@fixture
def mock_config():
    """A Config whose runner returns coroutines and whose SDK client
    points at a Mock DataConnectors collection.

    The returned config is pre-wired and must be passed as `obj=` to
    CliRunner.invoke so make_pass_decorator(ensure=True) reuses it
    instead of constructing a fresh Config.
    """
    client = Mock(spec=DataConnectors)
    cloudbender = Mock(spec=Cloudbender)
    cloudbender.data_connectors = client
    sdk = Mock(spec=TrainML)
    sdk.cloudbender = cloudbender
    runner = Mock(spec=TrainMLRunner)
    runner.client = sdk

    def run(*tasks):
        task = tasks[0] if len(tasks) == 1 else asyncio.gather(*tasks)
        return asyncio.run(task)

    runner.run = run
    config = Config()
    config.trainml = runner
    config.client = client
    yield config


class DataConnectorCliTests:
    def test_list_data_connectors(
        self, runner, mock_config, mock_connection_data_connectors
    ):
        mock_config.client.list = AsyncMock(
            return_value=mock_connection_data_connectors
        )
        mock_config.stdout = io.StringIO()
        result = runner.invoke(
            data_connector_cmd,
            ["list", "-p", "prov-id-1", "-r", "reg-id-1"],
            obj=mock_config,
            catch_exceptions=True,
        )
        assert result.exit_code == 0
        mock_config.client.list.assert_called_once_with(
            provider_uuid="prov-id-1", region_uuid="reg-id-1"
        )
        output = mock_config.stdout.getvalue()
        assert "On-Prem HTTP Connector" in output
        assert "api_key" in output

    def test_create_data_connector(self, runner, mock_config, tmp_path):
        pem_file = tmp_path / "key.pem"
        pem_file.write_text("-----BEGIN PRIVATE KEY-----\nfake\n")

        result = runner.invoke(
            data_connector_cmd,
            [
                "create",
                "-p",
                "prov-id-1",
                "-r",
                "reg-id-1",
                "--type",
                "pg",
                "--host",
                "10.0.0.20",
                "--port",
                "5432",
                "--database",
                "ml_data",
                "--auth-type",
                "password",
                "--username",
                "user",
                "--password",
                "secret",
                "--client-private-key-file",
                str(pem_file),
                "my-connector",
            ],
            obj=mock_config,
            catch_exceptions=True,
        )
        assert result.exit_code == 0
        mock_config.client.create.assert_called_once_with(
            provider_uuid="prov-id-1",
            region_uuid="reg-id-1",
            name="my-connector",
            type="pg",
            host="10.0.0.20",
            port=5432,
            tls=None,
            database="ml_data",
            protocol=None,
            port_range=None,
            cidr=None,
            auth_type="password",
            username="user",
            password="secret",
            api_key=None,
            authorization_header=None,
            authorization_prefix=None,
            client_id=None,
            client_secret=None,
            token_url=None,
            scopes=None,
            ca_certificate=None,
            client_certificate=None,
            client_private_key="-----BEGIN PRIVATE KEY-----\nfake\n",
        )

    def test_create_data_connector_tls_and_authorization(self, runner, mock_config):
        result = runner.invoke(
            data_connector_cmd,
            [
                "create",
                "-p",
                "prov-id-1",
                "-r",
                "reg-id-1",
                "--type",
                "http",
                "--host",
                "10.0.0.10",
                "--no-tls",
                "--auth-type",
                "api_key",
                "--api-key",
                "secret-key",
                "--authorization-header",
                "X-Api-Key",
                "--authorization-prefix",
                "Token ",
                "my-connector",
            ],
            obj=mock_config,
            catch_exceptions=True,
        )
        assert result.exit_code == 0
        mock_config.client.create.assert_called_once_with(
            provider_uuid="prov-id-1",
            region_uuid="reg-id-1",
            name="my-connector",
            type="http",
            host="10.0.0.10",
            port=None,
            tls=False,
            database=None,
            protocol=None,
            port_range=None,
            cidr=None,
            auth_type="api_key",
            username=None,
            password=None,
            api_key="secret-key",
            authorization_header="X-Api-Key",
            authorization_prefix="Token ",
            client_id=None,
            client_secret=None,
            token_url=None,
            scopes=None,
            ca_certificate=None,
            client_certificate=None,
            client_private_key=None,
        )

    def test_create_data_connector_rejects_unknown_type(self, runner, mock_config):
        result = runner.invoke(
            data_connector_cmd,
            [
                "create",
                "-p",
                "prov-id-1",
                "-r",
                "reg-id-1",
                "--type",
                "bad-type",
                "my-connector",
            ],
            obj=mock_config,
            catch_exceptions=True,
        )
        assert result.exit_code != 0
        assert "Invalid value for '--type'" in result.output

    def test_create_data_connector_rejects_both_cert_and_file(
        self, runner, mock_config, tmp_path
    ):
        pem_file = tmp_path / "cert.pem"
        pem_file.write_text("CERT")
        result = runner.invoke(
            data_connector_cmd,
            [
                "create",
                "-p",
                "prov-id-1",
                "-r",
                "reg-id-1",
                "--type",
                "http",
                "--host",
                "10.0.0.10",
                "--auth-type",
                "certificate",
                "--ca-certificate",
                "CERT",
                "--ca-certificate-file",
                str(pem_file),
                "my-connector",
            ],
            obj=mock_config,
            catch_exceptions=True,
        )
        assert result.exit_code != 0
        assert "may not both be specified" in result.output
        mock_config.client.create.assert_not_called()

    def test_edit_data_connector(
        self, runner, mock_config, mock_connection_data_connectors
    ):
        found = mock_connection_data_connectors[0]
        found.edit = AsyncMock(return_value=found)
        mock_config.client.list = AsyncMock(
            return_value=mock_connection_data_connectors
        )
        result = runner.invoke(
            data_connector_cmd,
            [
                "edit",
                "-p",
                "prov-id-1",
                "-r",
                "reg-id-1",
                "--host",
                "10.1.1.1",
                "con-conn-1",
            ],
            obj=mock_config,
            catch_exceptions=True,
        )
        assert result.exit_code == 0
        found.edit.assert_called_once_with(
            host="10.1.1.1",
            port=None,
            tls=None,
            database=None,
            protocol=None,
            port_range=None,
            cidr=None,
            auth_type=None,
            username=None,
            password=None,
            api_key=None,
            authorization_header=None,
            authorization_prefix=None,
            client_id=None,
            client_secret=None,
            token_url=None,
            scopes=None,
            ca_certificate=None,
            client_certificate=None,
            client_private_key=None,
        )

    def test_edit_data_connector_not_found(
        self, runner, mock_config, mock_connection_data_connectors
    ):
        mock_config.client.list = AsyncMock(
            return_value=mock_connection_data_connectors
        )
        result = runner.invoke(
            data_connector_cmd,
            [
                "edit",
                "-p",
                "prov-id-1",
                "-r",
                "reg-id-1",
                "does-not-exist",
            ],
            obj=mock_config,
            catch_exceptions=True,
        )
        assert result.exit_code != 0
        assert "Cannot find specified data_connector." in result.output
