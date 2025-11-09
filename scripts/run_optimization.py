#!/usr/bin/env python
"""
Restored run_optimization.py

Thin wrapper to execute the optimization pipeline. It delegates to
`run_rl_optimizer.py` to keep a single source of truth for the current
end-to-end run, including what-if analysis and visualizations.

Usage:
    python run_optimization.py [path_to_process_json]

If no path is provided, defaults to `examples/software_project.json`.
"""
from __future__ import annotations
import os
import sys
import subprocess
from typing import List

DEFAULT_PROCESS_PATH = os.path.join("examples", "software_project.json")


def _print_header():
    print("=" * 60)
    print("RUN OPTIMIZATION (wrapper)")
    print("Delegating to run_rl_optimizer.py")
    print("=" * 60)


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Ensure process path
    process_path = argv[0] if argv else DEFAULT_PROCESS_PATH
    if not os.path.exists(process_path):
        print(f"Process file not found: {process_path}")
        return 1

    _print_header()

    # Try to import and call a main() from run_rl_optimizer if present
    try:
        import run_rl_optimizer as rlo  # type: ignore
        if hasattr(rlo, "main") and callable(getattr(rlo, "main")):
            # Pass through the same CLI behavior
            return int(rlo.main([process_path]))
    except Exception as e:
        # Fallback to subprocess if import/call fails
        print(f"Note: Falling back to subprocess due to: {e}")

    # Subprocess fallback
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "run_rl_optimizer.py"), process_path]
    print("Executing:", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
    except Exception as e:
        print(f"Error running run_rl_optimizer.py: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
