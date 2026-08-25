#!/usr/bin/env python
"""启动 Task 0 的逐帧 Web 验证面板。

既支持 ``python dashboard.py``，也支持 ``streamlit run dashboard.py``。
前一种形式会自动重新进入 Streamlit CLI，避免 bare-mode ScriptRunContext 警告。
"""

from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        from streamlit.web import cli as streamlit_cli

        original_arguments = sys.argv[1:]
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.address",
            "127.0.0.1",
            *original_arguments,
        ]
        raise SystemExit(streamlit_cli.main())
    from jhtdb_regimes.dashboard import main

    main()
