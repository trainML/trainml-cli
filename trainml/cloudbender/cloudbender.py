from .providers import Providers
from .regions import Regions
from .nodes import Nodes
from .devices import Devices
from .datastores import Datastores
from .data_connectors import DataConnectors
from .services import Services
from .device_configs import DeviceConfigs


class Cloudbender(object):
    def __init__(self, trainml):
        self.trainml = trainml
        self.providers = Providers(trainml)
        self.regions = Regions(trainml)
        self.nodes = Nodes(trainml)
        self.devices = Devices(trainml)
        self.datastores = Datastores(trainml)
        self.data_connectors = DataConnectors(trainml)
        self.services = Services(trainml)
        self.device_configs = DeviceConfigs(trainml)
