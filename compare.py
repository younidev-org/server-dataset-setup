"""Compatibility entry point; implementation lives in dataset_hub."""
import sys
from dataset_hub import compare as _implementation
if __name__ == "__main__":
    sys.exit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
