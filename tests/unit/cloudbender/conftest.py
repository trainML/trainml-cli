import asyncio
from trainml.projects import Projects
from pytest import fixture, mark
from unittest.mock import Mock, AsyncMock, patch, create_autospec


from trainml.cloudbender import Cloudbender
from trainml.cloudbender.providers import Provider, Providers
from trainml.cloudbender.regions import Region, Regions
from trainml.cloudbender.nodes import Node, Nodes
from trainml.cloudbender.datastores import Datastore, Datastores
from trainml.cloudbender.reservations import Reservation, Reservations
from trainml.cloudbender.device_configs import DeviceConfig, DeviceConfigs

pytestmark = mark.unit


@fixture(scope="session")
def mock_providers():
    trainml = Mock()
    yield [
        Provider(
            trainml,
            **{
                "customer_uuid": "cust-id-1",
                "provider_uuid": "prov-id-1",
                "type": "physical",
                "credits": 10.0,
                "payment_mode": "stripe",
            },
        ),
        Provider(
            trainml,
            **{
                "customer_uuid": "cust-id-1",
                "provider_uuid": "prov-id-2",
                "type": "gcp",
                "credits": 0.0,
                "payment_mode": "credits",
                "credentials": {
                    "project": "proj-1",
                    "id": "gcp@serviceaccount.com",
                },
            },
        ),
    ]


@fixture(scope="session")
def mock_regions():
    trainml = Mock()
    yield [
        Region(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "provider_type": "physical",
                "name": "Physical Region 1",
                "status": "healthy",
            },
        ),
        Region(
            trainml,
            **{
                "provider_uuid": "prov-id-2",
                "region_uuid": "reg-id-2",
                "provider_type": "gcp",
                "name": "Cloud Region 1",
                "status": "healthy",
            },
        ),
    ]


@fixture(scope="session")
def mock_nodes():
    trainml = Mock()
    yield [
        Node(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "rig_uuid": "rig-id-1",
                "type": "permanent",
                "service": "compute",
                "friendly_name": "hq-a100-01",
                "hostname": "hq-a100-01",
                "status": "active",
                "online": True,
                "maintenance_mode": False,
            },
        ),
        Node(
            trainml,
            **{
                "provider_uuid": "prov-id-2",
                "region_uuid": "reg-id-2",
                "rig_uuid": "rig-id-2",
                "type": "ephemeral",
                "service": "compute",
                "friendly_name": "gcp-a100-01",
                "hostname": "gcp-a100-01",
                "status": "active",
                "online": True,
                "maintenance_mode": False,
            },
        ),
        Node(
            trainml,
            **{
                "provider_uuid": "prov-id-2",
                "region_uuid": "reg-id-2",
                "rig_uuid": "rig-id-3",
                "type": "permanent",
                "service": "storage",
                "friendly_name": "gcp-storage-01",
                "hostname": "gcp-storage-01",
                "status": "active",
                "online": True,
                "maintenance_mode": False,
            },
        ),
    ]


@fixture(scope="session")
def mock_datastores():
    trainml = Mock()
    yield [
        Datastore(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "store_id": "store-id-1",
                "type": "nfs",
                "name": "On-prem NFS",
                "uri": "192.168.0.50",
                "root": "/exports",
            },
        ),
        Datastore(
            trainml,
            **{
                "provider_uuid": "prov-id-2",
                "region_uuid": "reg-id-2",
                "store_id": "store-id-2",
                "type": "smb",
                "name": "GCP Samba",
                "uri": "192.168.1.50",
                "root": "/DATA",
            },
        ),
    ]


@fixture(scope="session")
def mock_reservations():
    trainml = Mock()
    yield [
        Reservation(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "reservation_id": "res-id-1",
                "type": "port",
                "name": "On-Prem Service A",
                "resource": "8001",
                "hostname": "service-a.local",
            },
        ),
        Reservation(
            trainml,
            **{
                "provider_uuid": "prov-id-2",
                "region_uuid": "reg-id-2",
                "reservation_id": "res-id-2",
                "type": "port",
                "name": "Cloud Service B",
                "resource": "8001",
                "hostname": "service-b.local",
            },
        ),
    ]


@fixture(scope="session")
def mock_device_configs():
    trainml = Mock()
    yield [
        DeviceConfig(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "config_id": "conf-id-1",
                "name": "IoT 1",
            },
        ),
        DeviceConfig(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "config_id": "conf-id-2",
                "name": "IoT 2",
            },
        ),
    ]


@fixture(scope="function")
def mock_trainml(
    mock_trainml,
    mock_providers,
    mock_regions,
    mock_nodes,
    mock_datastores,
    mock_reservations,
    mock_device_configs,
):
    mock_trainml.cloudbender = create_autospec(Cloudbender)

    mock_trainml.cloudbender.providers = create_autospec(Providers)
    mock_trainml.cloudbender.providers.list = AsyncMock(
        return_value=mock_providers
    )
    mock_trainml.cloudbender.regions = create_autospec(Regions)
    mock_trainml.cloudbender.regions.list = AsyncMock(
        return_value=mock_regions
    )
    mock_trainml.cloudbender.nodes = create_autospec(Nodes)
    mock_trainml.cloudbender.nodes.list = AsyncMock(return_value=mock_nodes)
    mock_trainml.cloudbender.datastores = create_autospec(Datastores)
    mock_trainml.cloudbender.datastores.list = AsyncMock(
        return_value=mock_datastores
    )
    mock_trainml.cloudbender.reservations = create_autospec(Reservations)
    mock_trainml.cloudbender.reservations.list = AsyncMock(
        return_value=mock_reservations
    )
    mock_trainml.cloudbender.device_configs = create_autospec(DeviceConfigs)
    mock_trainml.cloudbender.device_configs.list = AsyncMock(
        return_value=mock_device_configs
    )

    yield mock_trainml
