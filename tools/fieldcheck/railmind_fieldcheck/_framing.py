# Framers for the standalone Field Input Check.
# Pure numpy + Python standard library; self-contained.
"""
FRAMERS — turn raw source samples into FEATURE VECTORS.

A Framer.frame(raw) returns a feature vector (1-D array) OR None ("need more
samples", e.g. a window not yet full). The caller skips None.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Sequence, Deque, Any
import numpy as np


class Framer(ABC):
    @abstractmethod
    def frame(self, raw: np.ndarray) -> Optional[np.ndarray]:
        ...

    def reset(self) -> None:
        pass


class PassthroughFramer(Framer):
    """The source already yields feature vectors. Identity."""

    def frame(self, raw: np.ndarray) -> Optional[np.ndarray]:
        return np.asarray(raw, dtype=np.float32).ravel()


class WindowStatsFramer(Framer):
    """Sliding/hop window over the raw stream -> per-feature summary statistics."""

    _STATS = ("mean", "std", "min", "max", "range", "slope")

    def __init__(self, window: int = 64, hop: int = 32,
                 stats: Sequence[str] = ("mean", "std", "range", "slope")):
        if window < 2 or not (1 <= hop <= window):
            raise ValueError("require window>=2 and 1<=hop<=window")
        bad = set(stats) - set(self._STATS)
        if bad:
            raise ValueError(f"unknown stats {bad}; allowed {self._STATS}")
        self.window = int(window)
        self.hop = int(hop)
        self.stats = tuple(stats)
        self._buf: Deque[np.ndarray] = deque(maxlen=self.window)
        self._since = 0

    def reset(self) -> None:
        self._buf.clear()
        self._since = 0

    def frame(self, raw: np.ndarray) -> Optional[np.ndarray]:
        self._buf.append(np.asarray(raw, dtype=np.float32).ravel())
        self._since += 1
        if len(self._buf) < self.window or self._since < self.hop:
            return None
        self._since = 0
        W = np.asarray(self._buf)
        parts = []
        for s in self.stats:
            if s == "mean":   parts.append(W.mean(0))
            elif s == "std":  parts.append(W.std(0))
            elif s == "min":  parts.append(W.min(0))
            elif s == "max":  parts.append(W.max(0))
            elif s == "range": parts.append(W.max(0) - W.min(0))
            elif s == "slope":
                t = np.arange(len(W), dtype=np.float32); t -= t.mean()
                denom = float((t * t).sum())
                parts.append(((W - W.mean(0)) * t[:, None]).sum(0) / denom if denom > 1e-9
                             else np.zeros(W.shape[1], np.float32))
        return np.concatenate(parts).astype(np.float32)


class WindowFFTFramer(Framer):
    """Buffer a raw 1-channel waveform and emit an FFT magnitude spectrum every
    `hop` samples. Output dim = n_fft//2+1."""

    def __init__(self, n_fft: int = 256, hop: int = 128, normalize: bool = True):
        if n_fft < 4 or not (1 <= hop <= n_fft):
            raise ValueError("require n_fft>=4 and 1<=hop<=n_fft")
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.normalize = bool(normalize)
        self._buf: Deque[float] = deque(maxlen=self.n_fft)
        self._since = 0

    def reset(self) -> None:
        self._buf.clear()
        self._since = 0

    def frame(self, raw: np.ndarray) -> Optional[np.ndarray]:
        x = np.asarray(raw, dtype=np.float32).ravel()
        self._buf.append(float(x[0]))
        self._since += 1
        if len(self._buf) < self.n_fft or self._since < self.hop:
            return None
        self._since = 0
        mag = np.abs(np.fft.rfft(np.asarray(self._buf, dtype=np.float32)))
        if self.normalize:
            nrm = np.linalg.norm(mag)
            if nrm > 1e-10:
                mag = mag / nrm
        return mag.astype(np.float32)


class DerivedChannelFramer(Framer):
    """Build physics-informed DERIVED channels from a raw tag vector — stateless.

    `ops` specs: int (raw index) or nested tuple:
      ("passthrough", i) | ("diff", a, b) | ("sum", a, b) | ("ratio", num, den)
    Filter example — raw [P_up, P_down, Q] -> [ΔP, R=ΔP/Q]:
      DerivedChannelFramer([("diff", 0, 1), ("ratio", ("diff", 0, 1), 2)])
    """

    def __init__(self, ops: Sequence[Any], den_floor: float = 1e-3):
        if not ops:
            raise ValueError("DerivedChannelFramer needs >=1 op")
        self.ops = list(ops)
        self.den_floor = float(den_floor)

    def _eval(self, spec, raw: np.ndarray) -> float:
        if isinstance(spec, (int, np.integer)):
            return float(raw[int(spec)])
        op = spec[0]
        if op == "passthrough":
            return float(raw[int(spec[1])])
        if op == "diff":
            return self._eval(spec[1], raw) - self._eval(spec[2], raw)
        if op == "sum":
            return self._eval(spec[1], raw) + self._eval(spec[2], raw)
        if op == "ratio":
            num = self._eval(spec[1], raw)
            den = self._eval(spec[2], raw)
            den = den if abs(den) >= self.den_floor else (self.den_floor if den >= 0 else -self.den_floor)
            return num / den
        raise ValueError(f"unknown derived op {op!r}")

    def frame(self, raw: np.ndarray) -> Optional[np.ndarray]:
        raw = np.asarray(raw, dtype=np.float64).ravel()
        return np.array([self._eval(s, raw) for s in self.ops], dtype=np.float32)
