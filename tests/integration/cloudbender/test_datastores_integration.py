import re
from pytest import mark, fixture

pytestmark = [mark.sdk, mark.integration, mark.cloudbender, mark.datastores]

# Fake values only - nothing here points at a real host or credential.
FAKE_NFS_URI = "192.0.2.10"  # RFC 5737 TEST-NET-1 (documentation range)
FAKE_NFS_ROOT = "/exports/test"
FAKE_SMB_URI = "192.0.2.20"
FAKE_SMB_ROOT = "/DATA/test"
FAKE_SMB_USERNAME = "fake-smb-user"
FAKE_SMB_PASSWORD = "fake-smb-password-000000"


@mark.create
@mark.asyncio
@mark.xdist_group("cloudbender_resources")
class GetDatastoreTests:
    @fixture(scope="class")
    async def nfs_datastore(self, trainml, region):
        datastore = await trainml.cloudbender.datastores.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI-Automated-NFS-Datastore",
            type="nfs",
            uri=FAKE_NFS_URI,
            root=FAKE_NFS_ROOT,
        )
        yield datastore
        await datastore.remove()

    @fixture(scope="class")
    async def smb_datastore(self, trainml, region):
        datastore = await trainml.cloudbender.datastores.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI-Automated-SMB-Datastore",
            type="smb",
            uri=FAKE_SMB_URI,
            root=FAKE_SMB_ROOT,
            username=FAKE_SMB_USERNAME,
            password=FAKE_SMB_PASSWORD,
        )
        yield datastore
        await datastore.remove()

    async def test_get_datastores(self, trainml, region, nfs_datastore):
        datastores = await trainml.cloudbender.datastores.list(
            provider_uuid=region.provider_uuid, region_uuid=region.id
        )
        assert any(ds.id == nfs_datastore.id for ds in datastores)

    async def test_get_datastore(self, trainml, provider, region, nfs_datastore):
        response = await trainml.cloudbender.datastores.get(
            provider.id, region.id, nfs_datastore.id
        )
        assert response.id == nfs_datastore.id

    async def test_nfs_datastore_properties(self, region, nfs_datastore):
        assert isinstance(nfs_datastore.id, str)
        assert isinstance(nfs_datastore.provider_uuid, str)
        assert isinstance(nfs_datastore.region_uuid, str)
        assert isinstance(nfs_datastore.name, str)
        assert isinstance(nfs_datastore.type, str)
        assert nfs_datastore.type == "nfs"
        assert nfs_datastore.provider_uuid == region.provider_uuid
        assert nfs_datastore.region_uuid == region.id
        assert nfs_datastore._datastore.get("uri") == FAKE_NFS_URI
        assert nfs_datastore._datastore.get("root") == FAKE_NFS_ROOT

    async def test_smb_datastore_properties(self, region, smb_datastore):
        assert isinstance(smb_datastore.id, str)
        assert smb_datastore.type == "smb"
        assert smb_datastore.provider_uuid == region.provider_uuid
        assert smb_datastore.region_uuid == region.id
        data = smb_datastore._datastore
        assert data.get("uri") == FAKE_SMB_URI
        assert data.get("root") == FAKE_SMB_ROOT
        # Credentials are encrypted at rest and must never be echoed back.
        assert "username" not in data
        assert "password" not in data

    async def test_datastore_str(self, nfs_datastore):
        string = str(nfs_datastore)
        regex = r"^{.*\"store_id\": \"" + nfs_datastore.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_datastore_repr(self, nfs_datastore):
        string = repr(nfs_datastore)
        regex = (
            r"^Datastore\( trainml , \*\*{.*'store_id': '"
            + nfs_datastore.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)
