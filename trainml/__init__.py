import warnings
import logging

with warnings.catch_warnings():
    # this will suppress all warnings in this block
    warnings.filterwarnings("ignore", message="int_from_bytes is deprecated")
    from .trainml import TrainML

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)


__version__ = "0.4.10"
__all__ = "TrainML"
