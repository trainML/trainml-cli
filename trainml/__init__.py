import warnings
import logging

with warnings.catch_warnings():
    # this will suppress all warnings in this block
    warnings.filterwarnings("ignore", message="int_from_bytes is deprecated")
    from .trainml import TrainML

logging.basicConfig(
    format="%(asctime)s.%(msecs)03dZ  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


__version__ = "0.5.7"
__all__ = "TrainML"
