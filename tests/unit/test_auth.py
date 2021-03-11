import re
import logging
import json
import os
from unittest.mock import AsyncMock, patch, mock_open, MagicMock
from pytest import mark, fixture, raises
from aiohttp import WSMessage, WSMsgType

import trainml.auth as specimen

pytestmark = [mark.sdk, mark.unit]


@patch.dict(
    os.environ,
    {
        "TRAINML_USER": "user-id",
        "TRAINML_KEY": "key",
        "TRAINML_REGION": "ap-east-1",
        "TRAINML_CLIENT_ID": "client_id",
        "TRAINML_POOL_ID": "pool_id",
    },
)
def test_auth_from_envs():
    auth = specimen.Auth()
    assert auth.__dict__.get("username") == "user-id"
    assert auth.__dict__.get("password") == "key"
    assert auth.__dict__.get("region") == "ap-east-1"
    assert auth.__dict__.get("client_id") == "client_id"
    assert auth.__dict__.get("pool_id") == "pool_id"
