import asyncio
from trainml.projects import Projects
from pytest import fixture, mark
from unittest.mock import Mock, AsyncMock, patch, create_autospec

from trainml.trainml import TrainML
from trainml.auth import Auth
from trainml.datasets import Dataset, Datasets
from trainml.checkpoints import Checkpoint, Checkpoints
from trainml.volumes import Volume, Volumes
from trainml.models import Model, Models
from trainml.gpu_types import GpuType, GpuTypes
from trainml.environments import Environment, Environments
from trainml.jobs import Job, Jobs
from trainml.connections import Connections
from trainml.projects import (
    Projects,
    Project,
    ProjectDatastore,
    ProjectService,
)
from trainml.cloudbender import Cloudbender
from trainml.cloudbender.providers import Provider, Providers
from trainml.cloudbender.regions import Region, Regions
from trainml.cloudbender.nodes import Node, Nodes
from trainml.cloudbender.devices import Device, Devices
from trainml.cloudbender.datastores import Datastore, Datastores
from trainml.cloudbender.services import Service, Services
from trainml.cloudbender.device_configs import DeviceConfig, DeviceConfigs


pytestmark = mark.unit


@fixture(scope="session")
def mock_my_datasets():
    trainml = Mock()
    yield [
        Dataset(
            trainml,
            dataset_uuid="1",
            project_uuid="proj-id-1",
            name="first one",
            status="ready",
            size=100000000,
            createdAt="2020-12-31T23:59:59.000Z",
        ),
        Dataset(
            trainml,
            dataset_uuid="2",
            project_uuid="proj-id-1",
            name="second one",
            status="ready",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Dataset(
            trainml,
            dataset_uuid="3",
            project_uuid="proj-id-1",
            name="first one",
            status="ready",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Dataset(
            trainml,
            dataset_uuid="4",
            project_uuid="proj-id-1",
            name="other one",
            status="ready",
            size=100000000,
            createdAt="2020-12-31T23:59:59.000Z",
        ),
        Dataset(
            trainml,
            dataset_uuid="5",
            project_uuid="proj-id-1",
            name="not ready",
            status="new",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Dataset(
            trainml,
            dataset_uuid="6",
            project_uuid="proj-id-1",
            name="failed",
            status="failed",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
    ]


@fixture(scope="session")
def mock_public_datasets():
    trainml = Mock()
    yield [
        Dataset(
            trainml,
            dataset_uuid="11",
            name="first one",
            status="ready",
        ),
        Dataset(
            trainml,
            dataset_uuid="12",
            name="second one",
            status="ready",
        ),
        Dataset(
            trainml,
            dataset_uuid="15",
            name="not ready",
            status="new",
        ),
        Dataset(
            trainml,
            dataset_uuid="16",
            name="failed",
            status="failed",
        ),
    ]


@fixture(scope="session")
def mock_my_checkpoints():
    trainml = Mock()
    yield [
        Checkpoint(
            trainml,
            checkpoint_uuid="1",
            project_uuid="proj-id-1",
            name="first one",
            status="ready",
            size=100000000,
            createdAt="2020-12-31T23:59:59.000Z",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="2",
            project_uuid="proj-id-1",
            name="second one",
            status="ready",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="3",
            project_uuid="proj-id-1",
            name="first one",
            status="ready",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="4",
            project_uuid="proj-id-1",
            name="other one",
            status="ready",
            size=100000000,
            createdAt="2020-12-31T23:59:59.000Z",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="5",
            project_uuid="proj-id-1",
            name="not ready",
            status="new",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="6",
            project_uuid="proj-id-1",
            name="failed",
            status="failed",
            size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
    ]


@fixture(scope="session")
def mock_public_checkpoints():
    trainml = Mock()
    yield [
        Checkpoint(
            trainml,
            checkpoint_uuid="11",
            name="first one",
            status="ready",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="12",
            name="second one",
            status="ready",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="15",
            name="not ready",
            status="new",
        ),
        Checkpoint(
            trainml,
            checkpoint_uuid="16",
            name="failed",
            status="failed",
        ),
    ]


@fixture(scope="session")
def mock_models():
    trainml = Mock()
    yield [
        Model(
            trainml,
            model_uuid="1",
            project_uuid="proj-id-1",
            name="first one",
            status="ready",
            size=10000000,
            createdAt="2020-12-31T23:59:59.000Z",
        ),
        Model(
            trainml,
            model_uuid="2",
            project_uuid="proj-id-1",
            name="second one",
            status="ready",
            size=10000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Model(
            trainml,
            model_uuid="5",
            project_uuid="proj-id-1",
            name="not ready",
            status="new",
            size=10000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Model(
            trainml,
            model_uuid="6",
            project_uuid="proj-id-1",
            name="failed",
            status="failed",
            size=10000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
    ]


@fixture(scope="session")
def mock_my_volumes():
    trainml = Mock()
    yield [
        Volume(
            trainml,
            id="1",
            project_uuid="proj-id-1",
            name="first one",
            status="ready",
            capacity="10G",
            used_size=100000000,
            billed_size=100000000,
            createdAt="2020-12-31T23:59:59.000Z",
        ),
        Volume(
            trainml,
            id="2",
            project_uuid="proj-id-1",
            name="second one",
            status="ready",
            capacity="10G",
            used_size=100000000,
            billed_size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Volume(
            trainml,
            id="3",
            project_uuid="proj-id-1",
            name="first one",
            status="ready",
            capacity="10G",
            used_size=100000000,
            billed_size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Volume(
            trainml,
            id="4",
            project_uuid="proj-id-1",
            name="other one",
            status="ready",
            capacity="10G",
            used_size=100000000,
            billed_size=100000000,
            createdAt="2020-12-31T23:59:59.000Z",
        ),
        Volume(
            trainml,
            id="5",
            project_uuid="proj-id-1",
            name="not ready",
            status="new",
            capacity="10G",
            used_size=100000000,
            billed_size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
        Volume(
            trainml,
            id="6",
            project_uuid="proj-id-1",
            name="failed",
            status="failed",
            capacity="10G",
            used_size=100000000,
            billed_size=100000000,
            createdAt="2021-01-01T00:00:01.000Z",
        ),
    ]


@fixture(scope="session")
def mock_gpu_types():
    trainml = Mock()
    yield [
        GpuType(
            trainml,
            **{
                "name": "A100",
                "price": {"min": 2.78, "max": 4.9},
                "id": "a100-id",
                "abbrv": "a100",
            },
        ),
        GpuType(
            trainml,
            **{
                "price": {"min": 1, "max": 3.94},
                "name": "V100",
                "id": "v100-id",
                "abbrv": "v100",
            },
        ),
        GpuType(
            trainml,
            **{
                "price": {"min": 0.35, "max": 0.35},
                "name": "RTX 2080 Ti",
                "id": "2080ti-id",
                "abbrv": "rtx2080ti",
            },
        ),
        GpuType(
            trainml,
            **{
                "price": {"min": 0.1, "max": 0.1},
                "name": "GTX 1060",
                "id": "1060-id",
                "abbrv": "gtx1060",
            },
        ),
    ]


@fixture(scope="function")
def mock_environments():
    trainml = Mock()
    yield [
        Environment(
            trainml,
            **{
                "id": "DEEPLEARNING_PY39",
                "framework": "Deep Learning",
                "py_version": "3.9",
                "cuda_version": "11.7",
                "name": "Deep Learning - Python 3.9",
            },
        ),
        Environment(
            trainml,
            **{
                "id": "DEEPLEARNING_PY310",
                "framework": "Deep Learning",
                "py_version": "3.10",
                "cuda_version": "12.1",
                "name": "Deep Learning - Python 3.10",
            },
        ),
        Environment(
            trainml,
            **{
                "id": "PYTORCH_PY38_17",
                "framework": "PyTorch",
                "py_version": "3.8",
                "version": "1.7",
                "cuda_version": "11.1",
                "name": "PyTorch 1.7 - Python 3.8",
            },
        ),
        Environment(
            trainml,
            **{
                "id": "PYTORCH_PY37_17",
                "framework": "PyTorch",
                "py_version": "3.7",
                "version": "1.7",
                "cuda_version": "10.2",
                "name": "PyTorch 1.7 - Python 3.7",
            },
        ),
        Environment(
            trainml,
            **{
                "id": "PYTORCH_PY37_16",
                "framework": "PyTorch",
                "py_version": "3.7",
                "version": "1.6",
                "cuda_version": "10.2",
                "name": "PyTorch 1.6 - Python 3.7",
            },
        ),
        Environment(
            trainml,
            **{
                "id": "PYTORCH_PY37_15",
                "framework": "PyTorch",
                "py_version": "3.7",
                "version": "1.5",
                "cuda_version": "10.1",
                "name": "PyTorch 1.5 - Python 3.7",
            },
        ),
        Environment(
            trainml,
            **{
                "id": "TENSORFLOW_PY38_24",
                "framework": "Tensorflow",
                "py_version": "3.8",
                "version": "2.4",
                "cuda_version": "11.1",
                "name": "Tensorflow 2.4 - Python 3.8",
            },
        ),
        Environment(
            trainml,
            **{
                "id": "TENSORFLOW_PY37_114",
                "framework": "Tensorflow",
                "py_version": "3.7",
                "version": "1.14",
                "cuda_version": "10.1",
                "name": "Tensorflow 1.14 - Python 3.7",
            },
        ),
    ]


@fixture(scope="session")
def mock_jobs():
    trainml = Mock()
    yield [
        Job(
            trainml,
            **{
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-1",
                "name": "test notebook",
                "start": "2021-02-11T15:46:22.455Z",
                "type": "notebook",
                "status": "new",
                "credits_per_hour": 0.1,
                "credits": 0.1007,
                "workers": [
                    {
                        "rig_uuid": "rig-id-1",
                        "job_worker_uuid": "worker-id-1",
                        "command": "jupyter lab",
                        "status": "new",
                    }
                ],
                "worker_status": "new",
                "resources": {
                    "gpu_count": 1,
                    "gpu_type_id": "1060-id",
                    "disk_size": 10,
                },
                "model": {
                    "size": 7176192,
                    "source_type": "git",
                    "source_uri": "git@github.com:trainML/test-private.git",
                    "status": "new",
                },
                "data": {
                    "datasets": [
                        {
                            "dataset_uuid": "data-id-1",
                            "name": "first one",
                            "type": "public",
                            "size": 184549376,
                        },
                        {
                            "dataset_uuid": "data-id-2",
                            "name": "second one",
                            "type": "public",
                            "size": 5068061409,
                        },
                    ],
                    "status": "ready",
                },
                "environment": {
                    "type": "DEEPLEARNING_PY310",
                    "image_size": 44966989795,
                    "env": [
                        {"value": "env1val", "key": "env1"},
                        {"value": "env2val", "key": "env2"},
                    ],
                    "worker_key_types": ["aws", "gcp"],
                    "status": "new",
                },
                "vpn": {
                    "status": "new",
                    "cidr": "10.106.171.0/24",
                    "client": {
                        "port": "36017",
                        "id": "cus-id-1",
                        "address": "10.106.171.253",
                    },
                    "net_prefix_type_id": 1,
                },
                "nb_token": "token",
            },
        ),
        Job(
            trainml,
            **{
                "project_uuid": "proj-id-1",
                "job_uuid": "job-id-2",
                "name": "test training",
                "start": "2021-02-11T15:48:39.476Z",
                "stop": "2021-02-11T15:50:16.554Z",
                "type": "training",
                "status": "finished",
                "credits_per_hour": 0.3,
                "credits": 0.0054,
                "workers": [
                    {
                        "rig_uuid": "rig-id-1",
                        "job_worker_uuid": "worker-id-11",
                        "command": "PYTHONPATH=$PYTHONPATH:$ML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$ML_DATA_PATH --model_dir=$ML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=1 --batch_size=1024",
                        "status": "stopped",
                    },
                    {
                        "rig_uuid": "rig-id-2",
                        "job_worker_uuid": "worker-id-12",
                        "command": "PYTHONPATH=$PYTHONPATH:$ML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$ML_DATA_PATH --model_dir=$ML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=1 --batch_size=1024",
                        "status": "stopped",
                    },
                    {
                        "rig_uuid": "rig-id-2",
                        "job_worker_uuid": "worker-id-13",
                        "command": "PYTHONPATH=$PYTHONPATH:$ML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$ML_DATA_PATH --model_dir=$ML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=1 --batch_size=1024",
                        "status": "stopped",
                    },
                ],
                "worker_status": "stopped",
                "resources": {
                    "gpu_count": 1,
                    "gpu_type_id": "1060-id",
                    "disk_size": 10,
                },
                "model": {
                    "size": 7086080,
                    "source_type": "git",
                    "source_uri": "git@github.com:trainML/test-private.git",
                    "status": "ready",
                },
                "data": {
                    "datasets": [
                        {
                            "dataset_uuid": "data-id-1",
                            "name": "first one",
                            "type": "public",
                            "size": 184549376,
                        }
                    ],
                    "status": "ready",
                    "output_type": "aws",
                    "output_uri": "s3://trainml-examples/output/resnet_cifar10",
                },
                "environment": {
                    "type": "DEEPLEARNING_PY37",
                    "image_size": 39656398629,
                    "env": [],
                    "worker_key_types": [],
                    "status": "ready",
                },
                "vpn": {
                    "status": "stopped",
                    "cidr": "10.222.241.0/24",
                    "client": {
                        "port": "36978",
                        "id": "cus-id-1",
                        "address": "10.222.241.253",
                    },
                    "net_prefix_type_id": 1,
                },
            },
        ),
    ]


@fixture(scope="session")
def mock_projects():
    trainml = Mock()
    yield [
        Project(
            trainml,
            **{
                "id": "proj-id-1",
                "name": "Personal",
                "owner": True,
                "owner_name": "Me",
                "created_name": "Me",
                "job_all": True,
                "dataset_all": True,
                "model_all": True,
                "createdAt": "2020-12-31T23:59:59.000Z",
            },
        ),
        Project(
            trainml,
            **{
                "id": "project-id-2",
                "name": "My Other Project",
                "owner": True,
                "owner_name": "Me",
                "created_name": "Me",
                "job_all": True,
                "dataset_all": True,
                "model_all": True,
                "createdAt": "2020-12-31T23:59:59.000Z",
            },
        ),
        Project(
            trainml,
            **{
                "id": "project-id-3",
                "name": "Someone Elses Project",
                "owner": False,
                "owner_name": "Someone Else",
                "created_name": "Someone Else",
                "job_all": False,
                "dataset_all": False,
                "model_all": False,
                "createdAt": "2020-12-31T23:59:59.000Z",
            },
        ),
    ]


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
def mock_devices():
    trainml = Mock()
    yield [
        Device(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "device_id": "dev-id-1",
                "type": "device",
                "service": "compute",
                "friendly_name": "hq-orin-01",
                "hostname": "hq-orin-01",
                "status": "active",
                "online": True,
                "maintenance_mode": False,
                "job_status": "running",
                "job_last_deployed": "2023-06-02T21:22:40.084Z",
                "job_config_id": "job-id-1",
                "job_config_revision": "1685740490096",
                "device_config_id": "conf-id-1",
            },
        ),
        Device(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "device_id": "dev-id-2",
                "type": "device",
                "service": "compute",
                "friendly_name": "hq-orin-02",
                "hostname": "hq-orin-02",
                "status": "active",
                "online": True,
                "maintenance_mode": False,
                "job_status": "running",
                "job_last_deployed": "2023-06-02T21:22:40.084Z",
                "job_config_id": "job-id-2",
                "job_config_revision": "1685740490096",
                "device_config_id": "conf-id-1",
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
def mock_project_datastores():
    trainml = Mock()
    yield [
        ProjectDatastore(
            trainml,
            **{
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-1",
                "id": "store-id-1",
                "type": "nfs",
                "name": "On-prem NFS",
            },
        ),
        ProjectDatastore(
            trainml,
            **{
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-2",
                "id": "store-id-2",
                "type": "smb",
                "name": "GCP Samba",
            },
        ),
    ]


@fixture(scope="session")
def mock_services():
    trainml = Mock()
    yield [
        Service(
            trainml,
            **{
                "provider_uuid": "prov-id-1",
                "region_uuid": "reg-id-1",
                "service_id": "res-id-1",
                "type": "port",
                "name": "On-Prem Service A",
                "resource": "8001",
                "hostname": "service-a.local",
            },
        ),
        Service(
            trainml,
            **{
                "provider_uuid": "prov-id-2",
                "region_uuid": "reg-id-2",
                "service_id": "res-id-2",
                "type": "port",
                "name": "Cloud Service B",
                "resource": "8001",
                "hostname": "service-b.local",
            },
        ),
    ]


@fixture(scope="session")
def mock_project_services():
    trainml = Mock()
    yield [
        ProjectService(
            trainml,
            **{
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-1",
                "id": "res-id-1",
                "type": "port",
                "name": "On-Prem Service A",
                "resource": "8001",
                "hostname": "service-a.local",
            },
        ),
        ProjectService(
            trainml,
            **{
                "project_uuid": "proj-id-1",
                "region_uuid": "reg-id-2",
                "id": "res-id-2",
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
    mock_my_datasets,
    mock_public_datasets,
    mock_my_checkpoints,
    mock_public_checkpoints,
    mock_my_volumes,
    mock_models,
    mock_gpu_types,
    mock_environments,
    mock_jobs,
    mock_projects,
    mock_providers,
    mock_regions,
    mock_nodes,
    mock_devices,
    mock_datastores,
    mock_services,
    mock_device_configs,
):
    trainml = create_autospec(TrainML)
    trainml.active_project = "proj-id-1"
    trainml.project = "proj-id-1"
    trainml.datasets = create_autospec(Datasets)
    trainml.checkpoints = create_autospec(Checkpoints)
    trainml.volumes = create_autospec(Volumes)
    trainml.models = create_autospec(Models)
    trainml.gpu_types = create_autospec(GpuTypes)
    trainml.environments = create_autospec(Environments)
    trainml.jobs = create_autospec(Jobs)
    trainml.connections = create_autospec(Connections)
    trainml.projects = create_autospec(Projects)
    trainml.datasets.list = AsyncMock(return_value=mock_my_datasets)
    trainml.datasets.list_public = AsyncMock(return_value=mock_public_datasets)
    trainml.checkpoints.list = AsyncMock(return_value=mock_my_checkpoints)
    trainml.checkpoints.list_public = AsyncMock(return_value=mock_public_checkpoints)
    trainml.models.list = AsyncMock(return_value=mock_models)
    trainml.volumes.list = AsyncMock(return_value=mock_my_volumes)
    trainml.gpu_types.list = AsyncMock(return_value=mock_gpu_types)
    trainml.environments.list = AsyncMock(return_value=mock_environments)
    trainml.jobs.list = AsyncMock(return_value=mock_jobs)
    trainml.projects.list = AsyncMock(return_value=mock_projects)

    trainml.cloudbender = create_autospec(Cloudbender)

    trainml.cloudbender.providers = create_autospec(Providers)
    trainml.cloudbender.providers.list = AsyncMock(return_value=mock_providers)
    trainml.cloudbender.regions = create_autospec(Regions)
    trainml.cloudbender.regions.list = AsyncMock(return_value=mock_regions)
    trainml.cloudbender.nodes = create_autospec(Nodes)
    trainml.cloudbender.nodes.list = AsyncMock(return_value=mock_nodes)
    trainml.cloudbender.devices = create_autospec(Nodes)
    trainml.cloudbender.devices.list = AsyncMock(return_value=mock_devices)
    trainml.cloudbender.datastores = create_autospec(Datastores)
    trainml.cloudbender.datastores.list = AsyncMock(return_value=mock_datastores)
    trainml.cloudbender.services = create_autospec(Services)
    trainml.cloudbender.services.list = AsyncMock(return_value=mock_services)
    trainml.cloudbender.device_configs = create_autospec(DeviceConfigs)
    trainml.cloudbender.device_configs.list = AsyncMock(
        return_value=mock_device_configs
    )
    yield trainml
