import time
import json
import os
import asyncio
import aiohttp

from .auth import Auth
from .datasets import Datasets
from .jobs import Jobs
from .gpu_types import GpuTypes
from .environments import Environments
from .exceptions import ApiError

CONFIG_DIR = os.path.expanduser(os.environ.get("TRAINML_CONFIG_DIR") or "~/.trainml")


class TrainML(object):
    def __init__(self, **kwargs):
        try:
            with open(f"{CONFIG_DIR}/environment.json", "r") as file:
                env_str = file.read().replace("\n", "")
            env = json.loads(env_str)
        except:
            env = dict()
        self.auth = Auth(
            user=kwargs.get("user"),
            key=kwargs.get("key"),
            region=kwargs.get("region"),
            client_id=kwargs.get("client_id"),
            pool_id=kwargs.get("pool_id"),
        )
        self.datasets = Datasets(self)
        self.jobs = Jobs(self)
        self.gpu_types = GpuTypes(self)
        self.environments = Environments(self)
        self.api_url = (
            kwargs.get("api_url")
            or os.environ.get("TRAINML_API_URL")
            or env.get("api_url")
            or "api.trainml.ai"
        )
        self.ws_url = (
            kwargs.get("ws_url")
            or os.environ.get("TRAINML_WS_URL")
            or env.get("ws_url")
            or "api-ws.trainml.ai"
        )

    async def _query(self, path, method, params=None, data=None, headers=None):
        tokens = self.auth.get_tokens()
        headers = (
            {
                **headers,
                **{
                    "Authorization": tokens.get("id_token"),
                },
            }
            if headers
            else {
                "Authorization": tokens.get("id_token"),
            }
        )
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        url = f"https://{self.api_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, url, data=json.dumps(data), headers=headers
            ) as resp:
                if (resp.status // 100) in [4, 5]:
                    what = await resp.read()
                    content_type = resp.headers.get("content-type", "")
                    resp.close()
                    if content_type == "application/json":
                        raise ApiError(resp.status, json.loads(what.decode("utf8")))
                    else:
                        raise ApiError(resp.status, {"message": what.decode("utf8")})
                results = await resp.json()
                return results

    async def _ws_subscribe(self, entity, id, msg_handler):
        tokens = self.auth.get_tokens()
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                f"wss://{self.ws_url}?Authorization={tokens.get('id_token')}"
            ) as ws:
                asyncio.create_task(
                    ws.send_json(
                        dict(
                            action="getlogs",
                            data=dict(type="init", entity=entity, id=id),
                        )
                    )
                )
                asyncio.create_task(
                    ws.send_json(
                        dict(
                            action="subscribe",
                            data=dict(type="logs", entity=entity, id=id),
                        )
                    )
                )
                async for msg in ws:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    msg_handler(msg)