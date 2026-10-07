#!/usr/bin/env python3
"""Run predictions directly from a source checkout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


if __name__ == '__main__':
    from risk_propagation.prediction import main

    raise SystemExit(main())
