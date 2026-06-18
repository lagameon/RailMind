# RailMind — Field Input Check

A small, **self-contained** tool to confirm that on-site sensor data is **well-formed and
ready for monitoring** — *before* a RailMind deployment. It runs entirely on the local
machine (a Raspberry Pi, a laptop, an edge gateway) and depends only on **Python 3 + numpy**.
No connection to any RailMind service is required, and it performs **no anomaly scoring** —
it only checks that the data plumbing is clean.

---

## Quickstart (Raspberry Pi / Linux / macOS)

```bash
git clone https://github.com/lagameon/RailMind.git
cd RailMind/tools/fieldcheck
pip install numpy
python3 -m railmind_fieldcheck        # opens the web UI at http://127.0.0.1:8090
```

Pick your data source in the browser and click **Run check**. `Ctrl-C` to stop.

Headless / scripted use:

```python
from railmind_fieldcheck import run_field_check
report = run_field_check({
    "source": {"type": "csv_path", "path": "my_data.csv"},
    "framer": {"type": "passthrough"},
    "warmup_n": 200,
})
print(report["verdict"])   # PASS / WARN / FAIL
```

---

## What it checks

A readiness report (**PASS / WARN / FAIL**) over the input plumbing:

1. **Acquire input** — CSV upload, CSV path, raw waveform + FFT, or live **MQTT / OPC-UA**.
2. **Data health** — shape, NaN/Inf, stuck/constant channels, ragged columns.
3. **Framer** — turns raw samples into feature frames; flags "0 frames" (window never filled).
4. **Warmup sufficiency** — enough healthy frames to learn a baseline (and headroom).
5. **Feature-space overview** — a generic PCA effective-dimension readout (how many
   independent axes the data really has; flags near rank-1).

The goal: end up with a **CSV file (or equivalent)** of real machine data that **passes the
check** — that file is the starting point for a monitoring deployment.

---

## Kurzanleitung (Deutsch)

Ein kleines, **eigenständiges** Tool, das prüft, ob die Messdaten Ihrer Anlage **sauber und
monitoring-fähig** sind — *bevor* RailMind aufgespielt wird. Es läuft komplett lokal
(Raspberry Pi, Laptop, Edge-Gateway), braucht nur **Python 3 + numpy** und keinen Zugang zu
einem RailMind-Dienst. Es bewertet nichts — es stellt nur sicher, dass die Daten in Ordnung
sind.

```bash
git clone https://github.com/lagameon/RailMind.git
cd RailMind/tools/fieldcheck
pip install numpy
python3 -m railmind_fieldcheck        # Weboberfläche unter http://127.0.0.1:8090
```

Datenquelle auswählen (CSV-Datei, Roh-Waveform oder live über MQTT / OPC-UA) und auf
**Run check** klicken. Ziel: eine **CSV-Datei** (oder ein vergleichbares Datenformat) mit
echten Anlagendaten erzeugen, die den Check **besteht**.

---

## Tests

```bash
pytest railmind_fieldcheck/tests/ -v
```

`tests/test_standalone.py::test_self_contained` launches a clean interpreter and asserts that
importing and running the package loads **only this package** — it ships and runs on its own.
