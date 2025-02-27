import re
import sys
import asyncio
from pytest import mark, fixture
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes

pytestmark = [mark.sdk, mark.integration, mark.cloudbender, mark.regions]

def get_csr(service_id):
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096
    )
    csr = x509.CertificateSigningRequestBuilder().subject_name(x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "proxiML"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, service_id),
        x509.NameAttribute(NameOID.COMMON_NAME, "test-client"),  # Client identity
    ])).add_extension(
        x509.ExtendedKeyUsage([
            ExtendedKeyUsageOID.CLIENT_AUTH  # Client authentication usage
        ]),
        critical=True
    ).sign(private_key, hashes.SHA256())
    return csr.public_bytes(serialization.Encoding.PEM).decode("utf-8")

@mark.create
@mark.asyncio
@mark.xdist_group("cloudbender_resources")
class GetServiceTests:
    @fixture(scope="class")
    async def service(self, trainml, region):
        service = await trainml.cloudbender.services.create(
            provider_uuid=region.provider_uuid,
            region_uuid=region.id,
            name="CLI Automated Service",
            type="tcp",
            port="8989",
            public=False,
        )
        await service.wait_for("active")
        yield service
        await service.remove()
        await service.wait_for("archived")

    async def test_get_services(self, trainml, region,service):
        services = await trainml.cloudbender.services.list(provider_uuid=region.provider_uuid, region_uuid=region.id)
        assert len(services) > 0

    async def test_get_service(self, trainml, provider, region, service):
        response = await trainml.cloudbender.services.get(provider.id, region.id, service.id)
        assert response.id == service.id

    async def test_service_properties(self, region, service):
        assert isinstance(service.id, str)
        assert isinstance(service.provider_uuid, str)
        assert isinstance(service.region_uuid, str)
        assert isinstance(service.public, bool)
        assert service.port == "8989"
        assert service.provider_uuid == region.provider_uuid
        assert service.region_uuid == region.id

    async def test_service_str(self, service):
        string = str(service)
        regex = r"^{.*\"service_id\": \"" + service.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_service_repr(self, service):
        string = repr(service)
        regex = (
            r"^Service\( trainml , \*\*{.*'service_id': '"
            + service.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    async def test_service_certificate(self, service):
        service = await service.generate_certificate()
        assert isinstance(service._service.get("auth_cert"), str)
        csr = get_csr(service.id)
        certificate = await service.sign_client_certificate(csr)
        assert isinstance(certificate, str)
