"""
railmind_fieldcheck — standalone input-readiness check.

A self-contained tool (web app + headless API) that confirms a customer's sensor data is
well-formed and ready for monitoring: acquire → data health → framing → warmup
sufficiency → a generic PCA effective-dimension overview. It depends only on numpy + the
Python standard library and performs no anomaly scoring (a data-readiness checker, not a
monitor).

Self-contained by construction: `tests/test_standalone.py::test_self_contained` asserts
that importing and running it loads only this package (no external project modules).

    python -m railmind_fieldcheck                    # web app at http://127.0.0.1:8090
    from railmind_fieldcheck import run_field_check   # headless / scripted
"""
from .report import run_field_check
from .server import serve

__all__ = ["run_field_check", "serve"]
__version__ = "0.1.0"
