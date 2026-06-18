"""
Tests for the standalone Field Input Check.

Two jobs:
  1. functional — the input checks behave (PASS/WARN/FAIL across the cases an engineer hits);
  2. self-containment — importing + running this package loads ONLY this package, no
     external project module (the whole point of the standalone build). Run in a clean
     subprocess so other tests can't contaminate sys.modules.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request as U

import numpy as np

from railmind_fieldcheck.report import run_field_check
from railmind_fieldcheck.server import serve


def _csv(X, header=True):
    lines = [",".join(f"c{i}" for i in range(X.shape[1]))] if header else []
    lines += [",".join(f"{v:.5f}" for v in r) for r in X]
    return "\n".join(lines)


def _status(rep, name):
    return next(c["status"] for c in rep["checks"] if c["name"] == name)


def test_healthy_csv_passes():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (900, 6)).astype("f")
    rep = run_field_check({"source": {"type": "csv_text", "text": _csv(X)},
                           "framer": {"type": "passthrough"}, "warmup_n": 150})
    assert rep["verdict"] == "PASS", rep["summary"]
    names = {c["name"] for c in rep["checks"]}
    assert "Feature-space overview" in names
    # this is a readiness checker — it does no anomaly scoring (no score/preview output)
    assert "preview" not in rep
    assert "no anomaly scoring" in rep["note"]


def test_too_short_fails_warmup():
    X = np.random.default_rng(1).normal(0, 1, (80, 5)).astype("f")
    rep = run_field_check({"source": {"type": "csv_text", "text": _csv(X)},
                           "framer": {"type": "passthrough"}, "warmup_n": 200})
    assert _status(rep, "Warmup sufficiency") == "FAIL" and rep["verdict"] == "FAIL"


def test_nan_and_stuck_channel_warn():
    X = np.random.default_rng(2).normal(0, 1, (400, 6)).astype("f")
    X[:, 4] = 9.0; X[7, 1] = np.nan
    rep = run_field_check({"source": {"type": "csv_text", "text": _csv(X)},
                           "framer": {"type": "passthrough"}, "warmup_n": 100})
    h = next(c for c in rep["checks"] if c["name"] == "Data health")
    assert h["status"] == "WARN" and h["n_nan"] == 1 and 4 in h["stuck_channels"]


def test_rank1_geometry_flagged():
    # one latent factor driving all channels -> effective dim ~1
    rng = np.random.default_rng(3)
    f = rng.normal(0, 1, (600, 1)).astype("f")
    X = (f @ rng.normal(0, 1, (1, 8)).astype("f")) + 0.01 * rng.normal(0, 1, (600, 8)).astype("f")
    rep = run_field_check({"source": {"type": "csv_text", "text": _csv(X)},
                           "framer": {"type": "passthrough"}, "warmup_n": 200})
    g = next(c for c in rep["checks"] if c["name"] == "Feature-space overview")
    assert g["effective_dim_80"] <= 2 and g["status"] == "WARN"


def test_waveform_fft_frames():
    rng = np.random.default_rng(4)
    wav = (np.sin(0.2 * np.arange(6000)) + 0.3 * rng.normal(0, 1, 6000)).astype("f")
    rep = run_field_check({"source": {"type": "waveform_csv_text", "text": "v\n" + "\n".join(f"{x:.5f}" for x in wav)},
                           "framer": {"type": "window_fft", "n_fft": 256, "hop": 128}, "warmup_n": 15})
    fr = next(c for c in rep["checks"] if c["name"] == "Framer")
    assert fr["status"] == "PASS" and fr["frame_dim"] == 129


def test_unknown_source_fails_gracefully():
    rep = run_field_check({"source": {"type": "nope"}, "framer": {"type": "passthrough"}})
    assert rep["ok"] is False and rep["verdict"] == "FAIL"


def test_http_server():
    httpd = serve("127.0.0.1", 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)
    try:
        base = f"http://127.0.0.1:{port}"
        page = U.urlopen(base + "/").read().decode()
        assert "Field Input Check" in page and "input-readiness check" in page
        X = np.random.default_rng(5).normal(0, 1, (500, 4)).astype("f")
        cfg = {"source": {"type": "csv_text", "text": _csv(X)}, "framer": {"type": "passthrough"}, "warmup_n": 120}
        req = U.Request(base + "/run", data=json.dumps(cfg).encode(), headers={"Content-Type": "application/json"})
        rep = json.loads(U.urlopen(req).read())
        assert rep["verdict"] in ("PASS", "WARN", "FAIL")
    finally:
        httpd.shutdown()


def test_self_contained():
    """Self-containment guarantee: a clean interpreter that imports railmind_fieldcheck and
    runs a full check across EVERY offline source type and EVERY framer must load ONLY this
    package — no external project module (nothing matching `railmind*` other than the package
    itself). Exercises all branches so an import on any path is caught, not just inferred.
    (Live mqtt/opcua are omitted — they need a broker/server — but the csv_path branch below
    imports the same file-I/O adapter module, so `_sources.py` is exercised.)"""
    code = r"""
import sys, json, os, tempfile, numpy as np
import railmind_fieldcheck as fc
rng = np.random.default_rng(0)
X = rng.normal(0,1,(400,5)).astype('f')
def csv(A):
    return "\n".join([",".join(f"c{i}" for i in range(A.shape[1]))]+[",".join(f"{v:.4f}" for v in r) for r in A])
for fr in ({"type":"passthrough"},{"type":"window_stats","window":16,"hop":8},{"type":"window_fft","n_fft":64,"hop":32}):
    fc.run_field_check({"source":{"type":"csv_text","text":csv(X)},"framer":fr,"warmup_n":20})
wav = "v\n"+"\n".join(f"{v:.4f}" for v in rng.normal(0,1,2000))
fc.run_field_check({"source":{"type":"waveform_csv_text","text":wav},"framer":{"type":"window_fft","n_fft":64,"hop":32},"warmup_n":10})
d = tempfile.mkdtemp(); p = os.path.join(d,"s.csv"); open(p,"w").write(csv(X))
fc.run_field_check({"source":{"type":"csv_path","path":p},"framer":{"type":"passthrough"},"warmup_n":20})
fc.run_field_check({"source":{"type":"csv_text","text":csv(X)},"framer":{"type":"derived","ops":[("diff",0,1),("ratio",("diff",0,1),2)]},"warmup_n":20})
external = sorted(m for m in sys.modules if m.startswith("railmind") and not m.startswith("railmind_fieldcheck"))
print(json.dumps({"external_modules": external}))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["external_modules"] == [], f"external project module loaded: {res['external_modules']}"
