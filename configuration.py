"""Compatibility entry point; implementation lives in dataset_hub."""
import sys
from dataset_hub import configuration as _implementation
sys.modules[__name__] = _implementation
