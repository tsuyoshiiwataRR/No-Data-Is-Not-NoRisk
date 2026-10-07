#!/usr/bin/env python3
"""Generate the interactive company-risk graph from a source checkout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


if __name__ == '__main__':
    from risk_propagation.visualization import main

    raise SystemExit(main())
