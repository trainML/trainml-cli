import re
import logging
import json
import os
from unittest.mock import AsyncMock, patch, mock_open
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.trainml as specimen

pytestmark = [mark.sdk, mark.unit]


@patch.dict(
    os.environ,
    {
        "TRAINML_USER": "user-id",
        "TRAINML_KEY": "key",
        "TRAINML_REGION": "region",
        "TRAINML_CLIENT_ID": "client_id",
        "TRAINML_POOL_ID": "pool_id",
        "TRAINML_API_URL": "api.example.com",
        "TRAINML_WS_URL": "api-ws.example.com",
    },
)
def test_trainml_from_envs():
    trainml = specimen.TrainML()
    assert trainml.__dict__.get("api_url") == "api.example.com"
    assert trainml.__dict__.get("ws_url") == "api-ws.example.com"
    assert trainml.auth.__dict__.get("username") == "user-id"
    assert trainml.auth.__dict__.get("password") == "key"
    assert trainml.auth.__dict__.get("region") == "region"
    assert trainml.auth.__dict__.get("client_id") == "client_id"
    assert trainml.auth.__dict__.get("pool_id") == "pool_id"


def test_trainml_env_from_files():
    with patch(
        "trainml.trainml.open",
        mock_open(
            read_data=json.dumps(
                dict(
                    region="region_file",
                    client_id="client_id_file",
                    pool_id="pool_id_file",
                    api_url="api.example.com_file",
                    ws_url="api-ws.example.com_file",
                )
            )
        ),
    ):
        trainml = specimen.TrainML()
    assert trainml.__dict__.get("api_url") == "api.example.com_file"
    assert trainml.__dict__.get("ws_url") == "api-ws.example.com_file"
