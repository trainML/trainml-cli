import time
import json
import os
import asyncio
import aiohttp

from .auth import Auth
from .datasets import Datasets
from .jobs import Jobs
from .gpu_types import GpuTypes

CONFIG_DIR = os.path.expanduser(os.environ.get("TRAINML_CONFIG_DIR") or "~/.trainml")


class TrainML(object):
    def __init__(self):
        try:
            with open(f"{CONFIG_DIR}/environment.json", "r") as file:
                env_str = file.read().replace("\n", "")
            env = json.loads(env_str)
        except:
            env = dict()
        self.auth = Auth()
        self.datasets = Datasets(self)
        self.jobs = Jobs(self)
        self.gpu_types = GpuTypes(self)
        self.api_url = (
            os.environ.get("TRAINML_API_URL") or env.get("api_url") or "api.trainml.ai"
        )
        self.ws_url = (
            os.environ.get("TRAINML_WS_URL") or env.get("ws_url") or "api-ws.trainml.ai"
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