# RailMind Field Input Check — Standalone

A self-contained tool to confirm a customer's on-site sensor data is **well-formed and
ready for monitoring**, before a deployment. It runs entirely on the local machine and
depends only on **Python + numpy**, so it can be handed to a customer or field partner and
run anywhere.

```bash
python -m railmind_fieldcheck            # web app at http://127.0.0.1:8090
```

## What it does

A readiness report (PASS / WARN / FAIL) over the input plumbing:

1. **Acquire input** — CSV upload, CSV path, raw waveform + FFT, or live MQTT / OPC-UA.
2. **Data health** — shape, NaN/Inf, stuck/constant channels, ragged columns.
3. **Framer** — turns raw samples into feature frames; flags "0 frames" (window never filled).
4. **Warmup sufficiency** — enough healthy frames to learn a baseline (and headroom).
5. **Feature-space overview** — a generic PCA effective-dimension readout (how many
   independent axes the data really has; flags near rank-1).

## Scope

This is a **data-readiness checker, not a monitor**: it validates that data is well-formed
and ready, and performs **no anomaly scoring**. Use it to de-risk the on-site data plumbing
before a monitoring deployment.

## Self-contained (verified)

`tests/test_standalone.py::test_self_contained` launches a clean interpreter, imports this
package, runs a full check across every offline source type and every framer, and asserts
that **only this package is loaded** (`sys.modules` contains no module outside
`railmind_fieldcheck`). It is a hard test, not a claim. The I/O modules (`_sources.py`,
`_framing.py`) are pure numpy/stdlib, so the folder ships and runs on its own.

## Layout

```
railmind_fieldcheck/
  report.py     # run_field_check(cfg) — acquire/health/frame/warmup/geometry (pure)
  _sources.py   # I/O adapters (CSV / waveform / MQTT / OPC-UA), pure numpy/stdlib
  _framing.py   # framers (passthrough / window_stats / window_fft / derived)
  server.py     # stdlib HTTP server (GET / · GET /health · POST /run)
  page.py       # self-contained dark UI + PCA scree plot
  __main__.py   # python -m railmind_fieldcheck
  tests/        # functional + self-containment tests
```

Run the tests: `pytest railmind_fieldcheck/tests/ -v`.
