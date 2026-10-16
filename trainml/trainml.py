import json
import os
import asyncio
import aiohttp
import logging
import traceback
import random
from importlib.metadata import version

from trainml.utils.auth import Auth
from trainml.datasets import Datasets
from trainml.models import Models
from trainml.checkpoints import Checkpoints
from trainml.volumes import Volumes
from trainml.jobs import Jobs
from trainml.gpu_types import GpuTypes
from trainml.environments import Environments
from trainml.exceptions import ApiError, TrainMLException
from trainml.projects import Projects
from trainml.cloudbender import Cloudbender


async def delayed_close(ws):
    await asyncio.sleep(15)
    if not ws.closed:
        await ws.close()


WS_STATUS_POLL_INTERVAL = 30

ENTITY_COLLECTIONS = {
    "job": "jobs",
    "model": "models",
    "dataset": "datasets",
    "checkpoint": "checkpoints",
    "volume": "volumes",
}

ENTITY_TERMINAL_STATUSES = {
    "job": {"finished", "failed", "canceled"},
    "model": {"ready", "failed"},
    "dataset": {"ready", "failed"},
    "checkpoint": {"ready", "failed"},
    "volume": {"ready", "failed"},
}


class TrainML(object):
    def __init__(self, **kwargs):
        self._version = version("trainml")
        CONFIG_DIR = kwargs.get("config_dir") or os.path.expanduser(
            os.environ.get("TRAINML_CONFIG_DIR") or "~/.trainml"
        )
        try:
            with open(
                f"{CONFIG_DIR}/environment.json", "r", encoding="utf-8"
            ) as file:
                env_str = file.read().replace("\n", "")
            env = json.loads(env_str)
        except OSError:
            env = dict()
        try:
            with open(
                f"{CONFIG_DIR}/config.json", "r", encoding="utf-8"
            ) as file:
                config_str = file.read().replace("\n", "")
            config = json.loads(config_str)
        except OSError:
            config = dict()
        self.domain_suffix = (
            kwargs.get("domain_suffix")
            or os.environ.get("TRAINML_DOMAIN_SUFFIX")
            or env.get("domain_suffix")
            or "proximl.ai"
        )
        self.auth = Auth(
            config_dir=CONFIG_DIR,
            domain_suffix=self.domain_suffix,
            user=kwargs.get("user"),
            key=kwargs.get("key"),
            region=kwargs.get("region"),
            client_id=kwargs.get("client_id"),
            pool_id=kwargs.get("pool_id"),
        )
        self.active_project = (
            kwargs.get("project")
            or os.environ.get("TRAINML_PROJECT")
            or config.get("project")
        )
        self.datasets = Datasets(self)
        self.models = Models(self)
        self.checkpoints = Checkpoints(self)
        self.volumes = Volumes(self)
        self.jobs = Jobs(self)
        self.gpu_types = GpuTypes(self)
        self.environments = Environments(self)
        self.projects = Projects(self)
        self.cloudbender = Cloudbender(self)
        self.api_url = (
            kwargs.get("api_url")
            or os.environ.get("TRAINML_API_URL")
            or env.get("api_url")
            or f"api.{self.domain_suffix}"
        )
        self.ws_url = (
            kwargs.get("ws_url")
            or os.environ.get("TRAINML_WS_URL")
            or env.get("ws_url")
            or f"api-ws.{self.domain_suffix}"
        )

    @property
    def project(self) -> str:
        return self.active_project

    async def _query(
        self,
        path,
        method,
        params=None,
        data=None,
        headers=None,
        max_retries=3,
        backoff_factor=0.5,
    ):
        try:
            tokens = self.auth.get_tokens()
        except TrainMLException as e:
            raise e
        except Exception:
            raise TrainMLException(
                f"Error getting authorization tokens.  Verify configured credentials. Error: {traceback.format_exc()}"
            )
        logging.debug(
            f"Call parameters - Path: {path}, Method: {method}, Params: {params}, Body: {data}, Headers: {headers}"
        )
        headers = (
            {
                **headers,
                **{
                    "Authorization": tokens.get("id_token"),
                    "User-Agent": f"trainML-sdk/{self._version}",
                },
            }
            if headers
            else {
                "Authorization": tokens.get("id_token"),
                "User-Agent": f"trainML-sdk/{self._version}",
            }
        )
        if params:
            if not isinstance(params, dict):
                raise TrainMLException(
                    "Query parameters must be a valid dictionary"
                )
            params = {
                k: (str(v).lower() if isinstance(v, bool) else v)
                for k, v in params.items()
            }  ## aiohttp doesn't support boolean
        if (
            method != "POST"
            and self.active_project
            and (not params or "project_uuid" not in params.keys())
        ):
            params = (
                {**params, **{"project_uuid": self.active_project}}
                if params
                else {"project_uuid": self.active_project}
            )

        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        url = f"https://{self.api_url}{path}"

        logging.debug(
            f"Request - Url: {url}, Method: {method}, Params: {params}, Body: {data}, Headers: {headers}"
        )
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        data=json.dumps(data),
                        headers=headers,
                        params=params,
                    ) as resp:
                        if (resp.status // 100) in [4, 5]:
                            if (
                                resp.status == 502
                                and attempt < max_retries - 1
                            ):
                                wait_time = (
                                    (2**attempt)
                                    * backoff_factor
                                    * (random.random() + 0.5)
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                what = await resp.read()
                                content_type = resp.headers.get(
                                    "content-type", ""
                                )
                                resp.close()
                                if content_type == "application/json":
                                    raise ApiError(
                                        resp.status,
                                        json.loads(what.decode("utf8")),
                                    )
                                else:
                                    raise ApiError(
                                        resp.status,
                                        {"message": what.decode("utf8")},
                                    )
                        results = await resp.json()
                        return results
            except aiohttp.ClientResponseError as e:
                if e.status == 502 and attempt < max_retries - 1:
                    wait_time = (
                        (2**attempt) * backoff_factor * (random.random() + 0.5)
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise ApiError(e.status, f"Error {e.message}")

        raise TrainMLException("Unexpected API failure")

    async def _ws_subscribe(self, entity, project_uuid, id, msg_handler):
        headers = {
            "User-Agent": f"trainML-sdk/{self._version}",
            "Content-Type": "application/json",
        }
        try:
            try:
                tokens = self.auth.get_tokens()
            except TrainMLException as e:
                raise e
            except Exception:
                raise TrainMLException(
                    f"Error getting authorization tokens.  Verify configured credentials. Error: {traceback.format_exc()}"
                )
            async with aiohttp.ClientSession() as session:
                done = False
                done_event = asyncio.Event()
                watcher = None
                if entity in ENTITY_COLLECTIONS:
                    watcher = asyncio.create_task(
                        self._wait_for_entity_terminal(
                            entity, project_uuid, id, done_event
                        )
                    )
                try:
                    async with session.ws_connect(
                        f"wss://{self.ws_url}?Authorization={tokens.get('id_token')}",
                        headers=headers,
                        heartbeat=30,
                    ) as ws:
                        asyncio.create_task(
                            ws.send_json(
                                dict(
                                    action="getlogs",
                                    data=dict(
                                        type="init",
                                        entity=entity,
                                        id=id,
                                        project_uuid=project_uuid,
                                    ),
                                )
                            )
                        )
                        asyncio.create_task(
                            ws.send_json(
                                dict(
                                    action="subscribe",
                                    data=dict(
                                        type="logs",
                                        entity=entity,
                                        id=id,
                                        project_uuid=project_uuid,
                                    ),
                                )
                            )
                        )
                        done = await self._ws_consume(ws, msg_handler, done_event)
                    logging.debug(f"Websocket Disconnected.  Done? {done}")

                    connection_tries = 0
                    while not done:
                        tokens = self.auth.get_tokens()
                        try:
                            async with session.ws_connect(
                                f"wss://{self.ws_url}?Authorization={tokens.get('id_token')}",
                                headers=headers,
                                heartbeat=30,
                            ) as ws:
                                asyncio.create_task(
                                    ws.send_json(
                                        dict(
                                            action="subscribe",
                                            data=dict(
                                                type="logs",
                                                entity=entity,
                                                id=id,
                                                project_uuid=project_uuid,
                                            ),
                                        )
                                    )
                                )
                                done = await self._ws_consume(
                                    ws, msg_handler, done_event
                                )
                            connection_tries = 0
                            logging.debug(f"Websocket Disconnected.  Done? {done}")
                        except Exception:
                            connection_tries += 1
                            logging.debug(
                                f"Connection error: {traceback.format_exc()}"
                            )
                            if connection_tries == 5:
                                raise ApiError(
                                    500,
                                    {
                                        "message": f"Connection error: {traceback.format_exc()}"
                                    },
                                )
                finally:
                    if watcher is not None:
                        watcher.cancel()
                        await asyncio.gather(watcher, return_exceptions=True)
        except GeneratorExit:
            # Handle graceful shutdown - GeneratorExit is raised during
            # event loop cleanup. Don't re-raise to avoid "coroutine ignored"
            # warnings.
            logging.debug("Websocket subscription cancelled during shutdown")
            return
        except asyncio.CancelledError:
            # Re-raise CancelledError to properly propagate task cancellation
            logging.debug("Websocket subscription task cancelled")
            raise

    async def _ws_consume(self, ws, msg_handler, done_event):
        while True:
            recv_task = asyncio.create_task(ws.receive())
            signal_task = asyncio.create_task(done_event.wait())
            try:
                await asyncio.wait(
                    {recv_task, signal_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in (recv_task, signal_task):
                    if not task.done():
                        task.cancel()
            await asyncio.gather(recv_task, signal_task, return_exceptions=True)
            if not recv_task.done() or recv_task.cancelled():
                if not ws.closed:
                    await ws.close()
                return True
            msg = recv_task.result()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
                aiohttp.WSMsgType.CLOSE,
            ):
                logging.debug("Websocket Received Closed Message.")
                await ws.close()
                return False
            data = json.loads(msg.data)
            if data.get("type") == "end":
                asyncio.create_task(delayed_close(ws))
                return True
            msg_handler(data)

    async def _entity_status(self, entity, project_uuid, id):
        collection = getattr(self, ENTITY_COLLECTIONS[entity])
        try:
            obj = await collection.get(id, project_uuid=project_uuid)
        except ApiError:
            return None
        return obj.status

    async def _wait_for_entity_terminal(self, entity, project_uuid, id, done_event):
        terminal = ENTITY_TERMINAL_STATUSES[entity]
        while not done_event.is_set():
            status = await self._entity_status(entity, project_uuid, id)
            if status is None or status in terminal:
                logging.debug(
                    f"Entity {entity}/{id} status {status}; stopping websocket subscription"
                )
                done_event.set()
                return
            await asyncio.sleep(WS_STATUS_POLL_INTERVAL)

    def set_active_project(self, project_uuid):
        CONFIG_DIR = os.path.expanduser(
            os.environ.get("TRAINML_CONFIG_DIR") or "~/.trainml"
        )
        with open(f"{CONFIG_DIR}/config.json", "w") as file:
            json.dump(dict(project=project_uuid), file)
