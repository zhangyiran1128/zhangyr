#!/usr/bin/env python
"""Preprocess local POP909 MIDI files."""

from __future__ import annotations

import argparse

from preprocess_common import add_common_args, process_midi_dataset


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    process_midi_dataset(parser.parse_args(), "pop909")
