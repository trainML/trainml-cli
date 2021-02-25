import warnings

with warnings.catch_warnings():
    # this will suppress all warnings in this block
    warnings.filterwarnings("ignore", message="int_from_bytes is deprecated")
    from .trainml import TrainML


__version__ = "0.1.7"
__all__ = "TrainML"
