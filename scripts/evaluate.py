#!/usr/bin/env python
"""Run a paired experiment end to end: ``python scripts/evaluate.py --config experiments/toy.yaml``."""

import sys

from twist.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
