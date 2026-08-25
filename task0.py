#!/usr/bin/env python
"""Task 0 的直接运行脚本。

示例：
    python task0.py plan
    python task0.py smoke
    python task0.py fetch
    python task0.py verify
    python task0.py compute
    python task0.py run
"""

from __future__ import annotations

import sys

from jhtdb_regimes.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
