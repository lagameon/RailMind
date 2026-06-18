"""
railmind_fieldcheck.report — standalone input-readiness check module.

Self-contained: depends only on numpy + the Python standard library. It validates the
INPUT PLUMBING — acquire → data health → framing → warmup sufficiency → a generic PCA
effective-dimension overview — so an operator can confirm sensor data is well-formed and
ready for monitoring. It performs no anomaly scoring (a data-readiness checker, not a
monitor).

`run_field_check(cfg) -> report dict` is pure / headless; the web server just
serializes the dict. Every check is independently guarded.
"""
from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional
import numpy as np

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MAX_SAMPLES_DEFAULT = 5000
LIVE_BUDGET_S_DEFAULT = 8.0


def _check(name: str, status: str, detail: str, **extra) -> Dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **extra}


# ───────────────────────── acquisition ─────────────────────────

def _parse_csv_text(text: str, *, delimiter: str = ",", has_header: bool = True,
                    columns: Optional[List[int]] = None, max_rows: int = MAX_SAMPLES_DEFAULT):
    rows: List[np.ndarray] = []
    first = True
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        if first and has_header:
            first = False
            continue
        first = False
        try:
            vals = [float(p) for p in line.split(delimiter)]
        except ValueError:
            continue
        v = np.asarray(vals, np.float32)
        if columns is not None:
            v = v[columns]
        rows.append(v)
        if len(rows) >= max_rows:
            break
    return rows


def _acquire(cfg: Dict[str, Any]):
    src = cfg.get("source", {})
    stype = src.get("type", "csv_text")
    cap = int(cfg.get("max_samples", MAX_SAMPLES_DEFAULT))

    if stype == "csv_text":
        rows = _parse_csv_text(src.get("text", ""), delimiter=src.get("delimiter", ","),
                               has_header=src.get("has_header", True),
                               columns=src.get("columns"), max_rows=cap)
        return rows, f"csv_text · {len(rows)} rows"

    if stype == "waveform_csv_text":
        rows = _parse_csv_text(src.get("text", ""), delimiter=src.get("delimiter", ","),
                               has_header=src.get("has_header", True),
                               columns=src.get("columns"), max_rows=cap)
        flat = [np.asarray([float(r.ravel()[0])], np.float32) for r in rows if r.size]
        return flat, f"waveform_csv_text · {len(flat)} samples (1 channel)"

    if stype == "csv_path":
        from ._sources import CsvSource
        s = CsvSource(src["path"], delimiter=src.get("delimiter", ","),
                      has_header=src.get("has_header", True), columns=src.get("columns"),
                      max_rows=cap)
        rows = []
        for x in s.stream():
            rows.append(np.asarray(x, np.float32))
            if len(rows) >= cap:
                break
        return rows, f"csv_path · {src['path']} · {len(rows)} rows"

    if stype in ("mqtt", "opcua"):
        budget = float(cfg.get("live_budget_s", LIVE_BUDGET_S_DEFAULT))
        if stype == "mqtt":
            from ._sources import MqttSource
            s = MqttSource(src["host"], src["topic"], port=int(src.get("port", 1883)),
                           qos=int(src.get("qos", 0)))
        else:
            from ._sources import OpcUaSource
            s = OpcUaSource(src["endpoint"], src["node_ids"], period_s=float(src.get("period_s", 1.0)))
        rows, t0 = [], time.monotonic()
        try:
            for x in s.stream():
                rows.append(np.asarray(x, np.float32))
                if len(rows) >= cap or (time.monotonic() - t0) > budget:
                    break
        finally:
            try:
                s.close()
            except Exception:
                pass
        return rows, f"{stype} · collected {len(rows)} samples in ≤{budget:.0f}s"

    raise ValueError(f"unknown source type {stype!r}")


# ───────────────────────── checks ─────────────────────────

def _health_check(raw: List[np.ndarray]) -> Dict[str, Any]:
    if not raw:
        return _check("Data health", FAIL, "no samples acquired — check source / path / connection")
    lens = {r.size for r in raw}
    if len(lens) != 1:
        return _check("Data health", FAIL,
                      f"inconsistent sample width across rows: {sorted(lens)[:6]} — fix ragged columns",
                      n_samples=len(raw))
    X = np.vstack([r.reshape(1, -1) for r in raw]).astype(np.float64)
    n, d = X.shape
    n_nan = int(np.isnan(X).sum()); n_inf = int(np.isinf(X).sum())
    stds = np.nanstd(X, axis=0)
    stuck = [int(i) for i in np.where(stds < 1e-9)[0]]

    def _ptp(col):
        c = col[~np.isnan(col)]
        return float(np.ptp(c)) if c.size else float("nan")
    rng = [_ptp(X[:, j]) for j in range(d)]
    msgs, status = [], PASS
    if n_nan or n_inf:
        status = WARN; msgs.append(f"{n_nan} NaN / {n_inf} Inf values")
    if stuck:
        status = WARN; msgs.append(f"{len(stuck)} stuck/constant channel(s): {stuck[:8]}")
    if not msgs:
        msgs.append("no NaN/Inf, no stuck channels")
    return _check("Data health", status, f"{n}×{d} samples · " + "; ".join(msgs),
                  n_samples=n, n_features=d, n_nan=n_nan, n_inf=n_inf, stuck_channels=stuck,
                  feature_range=[round(v, 4) for v in rng[:16]])


def _build_framer(cfg: Dict[str, Any]):
    f = cfg.get("framer", {})
    ft = f.get("type", "passthrough")
    from ._framing import (PassthroughFramer, WindowStatsFramer, WindowFFTFramer, DerivedChannelFramer)
    if ft == "passthrough":
        return PassthroughFramer(), "passthrough"
    if ft == "window_stats":
        return WindowStatsFramer(window=int(f.get("window", 64)), hop=int(f.get("hop", 32)),
                                 stats=tuple(f.get("stats", ("mean", "std", "range", "slope")))), \
            f"window_stats(window={f.get('window',64)}, hop={f.get('hop',32)})"
    if ft == "window_fft":
        return WindowFFTFramer(n_fft=int(f.get("n_fft", 256)), hop=int(f.get("hop", 128)),
                               normalize=bool(f.get("normalize", True))), \
            f"window_fft(n_fft={f.get('n_fft',256)}, hop={f.get('hop',128)})"
    if ft == "derived":
        return DerivedChannelFramer(ops=f["ops"], den_floor=float(f.get("den_floor", 1e-3))), \
            f"derived({len(f.get('ops', []))} channels)"
    raise ValueError(f"unknown framer type {ft!r}")


def _frame(raw: List[np.ndarray], cfg: Dict[str, Any]):
    framer, fname = _build_framer(cfg)
    out = [np.asarray(y, np.float32) for r in raw if (y := framer.frame(r)) is not None]
    framed = np.vstack(out) if out else np.empty((0, 0), np.float32)
    return framed, {"framer": fname, "n_frames": len(out)}


def _framer_check(framed: np.ndarray, frinfo: Dict[str, Any], raw: List[np.ndarray]) -> Dict[str, Any]:
    if frinfo["n_frames"] == 0:
        return _check("Framer", FAIL,
                      f"{frinfo['framer']} produced 0 feature frames from {len(raw)} samples — "
                      f"window never filled (increase data, or reduce window/n_fft)", **frinfo)
    return _check("Framer", PASS, f"{frinfo['framer']} → {frinfo['n_frames']} frames × {int(framed.shape[1])}D",
                  frame_dim=int(framed.shape[1]), **frinfo)


def _warmup_check(framed: np.ndarray, warmup_n: int) -> Dict[str, Any]:
    nf = int(framed.shape[0])
    if nf < warmup_n:
        return _check("Warmup sufficiency", FAIL,
                      f"only {nf} frames < warmup_n={warmup_n}: cannot fit a baseline. "
                      f"Provide ≥ {warmup_n} (ideally ≥ {2*warmup_n}) frames of HEALTHY data.",
                      n_frames=nf, warmup_n=warmup_n)
    if nf < 2 * warmup_n:
        return _check("Warmup sufficiency", WARN,
                      f"{nf} frames ≥ warmup_n={warmup_n} but < 2×: a real run wants more headroom.",
                      n_frames=nf, warmup_n=warmup_n)
    return _check("Warmup sufficiency", PASS,
                  f"{nf} frames ≥ 2×warmup_n ({warmup_n})", n_frames=nf, warmup_n=warmup_n)


def _geometry_check(framed: np.ndarray, warmup_n: int) -> Dict[str, Any]:
    """Generic PCA effective-dimension overview (pure numpy SVD). Informational: it
    tells the engineer how many independent axes their feature space really has."""
    if framed.size == 0 or framed.shape[0] < max(4, warmup_n):
        return _check("Feature-space overview", WARN, "skipped — not enough frames for a PCA overview")
    X = framed[:warmup_n].astype(np.float64)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    Zc = Z - Z.mean(0)
    s = np.linalg.svd(Zc, compute_uv=False)
    var = s ** 2
    total = float(var.sum()) or 1.0
    cum = np.cumsum(var) / total
    eff80 = int(np.searchsorted(cum, 0.80) + 1)
    d = int(framed.shape[1])
    status = PASS
    note = f"{d} channels → effective dimension ≈ {eff80} (80% of variance)"
    if eff80 <= 1 and d > 1:
        status = WARN
        note += " — near rank-1 (one dominant degradation factor); covariance/whitening methods may be unstable here"
    return _check("Feature-space overview", status, note,
                  n_features=d, effective_dim_80=eff80,
                  cumulative_variance=[round(float(c), 4) for c in cum[:32]])


# ───────────────────────── orchestration ─────────────────────────

def run_field_check(cfg: Dict[str, Any]) -> Dict[str, Any]:
    warmup_n = int(cfg.get("warmup_n", 200))
    checks: List[Dict[str, Any]] = []

    try:
        raw, src_summary = _acquire(cfg)
    except Exception as e:
        return {"ok": False, "verdict": FAIL,
                "checks": [_check("Acquire input", FAIL, f"{type(e).__name__}: {e}")],
                "summary": {"pass": 0, "warn": 0, "fail": 1}}

    checks.append(_check("Acquire input", PASS if raw else FAIL, src_summary, n_raw=len(raw)))
    checks.append(_guard("Data health", lambda: _health_check(raw)))

    framed, frinfo = np.empty((0, 0), np.float32), {"framer": "?", "n_frames": 0}
    try:
        framed, frinfo = _frame(raw, cfg)
        checks.append(_framer_check(framed, frinfo, raw))
    except Exception as e:
        checks.append(_check("Framer", FAIL, f"{type(e).__name__}: {e}"))

    checks.append(_guard("Warmup sufficiency", lambda: _warmup_check(framed, warmup_n)))
    checks.append(_guard("Feature-space overview", lambda: _geometry_check(framed, warmup_n)))

    summary = {s.lower(): sum(1 for c in checks if c["status"] == s) for s in (PASS, WARN, FAIL)}
    verdict = FAIL if summary["fail"] else (WARN if summary["warn"] else PASS)
    return {"ok": summary["fail"] == 0, "verdict": verdict, "source": src_summary,
            "warmup_n": warmup_n, "checks": checks, "summary": summary,
            "note": "input-readiness check only — no anomaly scoring performed"}


def _guard(name: str, fn) -> Dict[str, Any]:
    try:
        return fn()
    except Exception as e:
        return _check(name, FAIL, f"{type(e).__name__}: {e}")
