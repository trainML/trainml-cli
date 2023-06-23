from .providers import Providers
from .regions import Regions
from .nodes import Nodes
from .devices import Devices
from .datastores import Datastores
from .reservations import Reservations
from .device_configs import DeviceConfigs


class Cloudbender(object):
    def __init__(self, trainml):
        self.trainml = trainml
        self.providers = Providers(trainml)
        self.regions = Regions(trainml)
        self.nodes = Nodes(trainml)
        self.devices = Devices(trainml)
        self.datastores = Datastores(trainml)
        self.reservations = Reservations(trainml)
        self.device_configs = DeviceConfigs(trainml)
