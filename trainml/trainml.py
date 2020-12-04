from requests import Request, Session
import time
import json
import os

from .auth import Auth
from .datasets import Datasets
from .jobs import Jobs

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
        self.session = Session()
        self.api_url = (
            os.environ.get("TRAINML_API_URL") or env.get("api_url") or "api.trainml.ai"
        )
        self.ws_url = (
            os.environ.get("TRAINML_WS_URL") or env.get("ws_url") or "api-ws.trainml.ai"
        )

    def _query(self, path, method, params=None, data=None, headers=None):
        tokens = self.auth.get_tokens()
        headers = (
            {**headers, **{"Authorization": tokens.get("id_token")}}
            if headers
            else {"Authorization": tokens.get("id_token")}
        )
        url = f"https://{self.api_url}{path}"
        req = Request(method, url, data=json.dumps(data), headers=headers)
        prepped = self.session.prepare_request(req)
        resp = self.session.send(prepped)
        return resp.json()