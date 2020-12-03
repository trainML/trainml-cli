from requests import Request, Session
import time
import json
import os

from auth import Auth
from datasets import Datasets
from jobs import Jobs

CONFIG_DIR = os.environ.get("TRAINML_CONFIG_DIR") or "~/.trainml"


class Trainml(object):
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

    def _query(self, path, method, params, data):
        tokens = self.auth.get_tokens()
        headers = {"Authorization": tokens.get("id_token")}
        url = f"https://{self.api_url}{path}"
        req = Request("GET", url, data=data, headers=headers)
        prepped = self.session.prepare_request(req)
        resp = self.session.send(prepped)
        print(resp.status_code)
        return resp.json()