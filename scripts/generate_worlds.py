#!/usr/bin/env python
"""Generate worlds and data splits: ``python scripts/generate_worlds.py worlds/linear_toy.yaml``."""

import sys

from twist.cli import main

if __name__ == "__main__":
    main(["generate-world", *sys.argv[1:]])
