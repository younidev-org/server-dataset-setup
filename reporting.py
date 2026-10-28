"""Compatibility entry point; implementation lives in dataset_hub."""
import sys
from dataset_hub import reporting as _implementation
sys.modules[__name__] = _implementation
