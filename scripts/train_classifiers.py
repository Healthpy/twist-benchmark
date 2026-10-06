#!/usr/bin/env python
"""Train a classifier on a generated world:
``python scripts/train_classifiers.py --world-dir artifacts/worlds/<id> --config classifiers/gru_small.yaml``."""

import sys

from twist.cli import main

if __name__ == "__main__":
    main(["train-classifier", *sys.argv[1:]])
