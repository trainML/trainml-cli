import re
import json
import logging
from unittest.mock import AsyncMock, patch
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.cloudbender.regions as specimen
from trainml.exceptions import (
    ApiError,
    SpecificationError,
    TrainMLException,
)

pytestmark = [mark.sdk, mark.unit, mark.cloudbender, mark.regions]


@fixture
def regions(mock_trainml):
    yield specimen.Regions(mock_trainml)


@fixture
def region(mock_trainml):
    yield specimen.Region(
        mock_trainml,
        provider_uuid="1",
        region_uuid="a",
        provider_type="physical",
        name="Physical Region 1",
        createdAt="2020-12-31T23:59:59.000Z",
        status="healthy",
    )


class RegionsTests:
    @mark.asyncio
    async def test_get_region(
        self,
        regions,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await regions.get("1234", "5687")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/5687", "GET", {}
        )

    @mark.asyncio
    async def test_list_regions(
        self,
        regions,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await regions.list("1234")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region", "GET", {}
        )

    @mark.asyncio
    async def test_remove_region(
        self,
        regions,
        mock_trainml,
    ):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await regions.remove("1234", "4567")
        mock_trainml._query.assert_called_once_with(
            "/provider/1234/region/4567", "DELETE", {}
        )

    @mark.asyncio
    async def test_create_region(self, regions, mock_trainml):
        requested_config = dict(
            provider_uuid="provider-id-1",
            name="phys-region",
            public=False,
            storage=dict(mode="local"),
        )
        expected_payload = dict(
            name="phys-region", public=False, storage=dict(mode="local")
        )
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "provider_type": "physical",
            "name": "phys-region",
            "status": "new",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }

        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await regions.create(**requested_config)
        mock_trainml._query.assert_called_once_with(
            "/provider/provider-id-1/region", "POST", None, expected_payload
        )
        assert response.id == "region-id-1"


class regionTests:
    def test_region_properties(self, region):
        assert isinstance(region.id, str)
        assert isinstance(region.provider_uuid, str)
        assert isinstance(region.type, str)
        assert isinstance(region.name, str)
        assert isinstance(region.status, str)

    def test_region_str(self, region):
        string = str(region)
        regex = r"^{.*\"region_uuid\": \"" + region.id + r"\".*}$"
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_region_repr(self, region):
        string = repr(region)
        regex = (
            r"^Region\( trainml , \*\*{.*'region_uuid': '"
            + region.id
            + r"'.*}\)$"
        )
        assert isinstance(string, str)
        assert re.match(regex, string)

    def test_region_bool(self, region, mock_trainml):
        empty_region = specimen.Region(mock_trainml)
        assert bool(region)
        assert not bool(empty_region)

    @mark.asyncio
    async def test_region_remove(self, region, mock_trainml):
        api_response = dict()
        mock_trainml._query = AsyncMock(return_value=api_response)
        await region.remove()
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a", "DELETE"
        )

    @mark.asyncio
    async def test_region_refresh(self, region, mock_trainml):
        api_response = {
            "provider_uuid": "provider-id-1",
            "region_uuid": "region-id-1",
            "provider_type": "physical",
            "name": "phys-region",
            "status": "new",
            "createdAt": "2020-12-31T23:59:59.000Z",
        }
        mock_trainml._query = AsyncMock(return_value=api_response)
        response = await region.refresh()
        mock_trainml._query.assert_called_once_with(
            f"/provider/1/region/a", "GET"
        )
        assert region.id == "region-id-1"
        assert response.id == "region-id-1"

    @mark.asyncio
    async def test_region_stage_dataset(self, region, mock_trainml):
        api_response = dict()
        expected_payload = dict(
            project_uuid="proj-id-1",
            dataset_uuid="dataset-id-1",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        await region.add_dataset("proj-id-1", "dataset-id-1")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/dataset", "POST", None, expected_payload
        )

    @mark.asyncio
    async def test_region_stage_model(self, region, mock_trainml):
        api_response = dict()
        expected_payload = dict(
            project_uuid="proj-id-1",
            model_uuid="model-id-1",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        await region.add_model("proj-id-1", "model-id-1")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/model", "POST", None, expected_payload
        )

    @mark.asyncio
    async def test_region_stage_checkpoint(self, region, mock_trainml):
        api_response = dict()
        expected_payload = dict(
            project_uuid="proj-id-1",
            checkpoint_uuid="checkpoint-id-1",
        )
        mock_trainml._query = AsyncMock(return_value=api_response)
        await region.add_checkpoint("proj-id-1", "checkpoint-id-1")
        mock_trainml._query.assert_called_once_with(
            "/provider/1/region/a/checkpoint", "POST", None, expected_payload
        )
