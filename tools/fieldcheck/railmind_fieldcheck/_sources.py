# I/O adapters for the standalone Field Input Check.
# Pure numpy + Python standard library; self-contained so this package ships and runs
# on its own.
"""
Data SOURCES — the I/O adapters that pull a raw sample stream off the wire.

A Source yields raw samples (a 1-D feature vector OR a raw scalar/waveform window
that a Framer reduces to a feature vector). Zero hard dependencies: IterableSource
and CsvSource cover offline / file-drop / historian-export with NO extra packages.
The industrial protocol adapters (MqttSource, OpcUaSource) LAZILY import their client
lib so this module always imports cleanly even when those libs are absent.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from typing import Iterator, Iterable, Optional, Sequence, Callable
import numpy as np


class Source(ABC):
    """Abstract data source. `stream()` yields raw samples as 1-D float arrays."""

    @abstractmethod
    def stream(self) -> Iterator[np.ndarray]:
        ...

    def close(self) -> None:
        pass

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.stream()


class IterableSource(Source):
    """Wrap ANY Python iterable / generator of samples (custom on-site driver)."""

    def __init__(self, it: Iterable):
        self._it = it

    def stream(self) -> Iterator[np.ndarray]:
        for s in self._it:
            yield np.asarray(s, dtype=np.float32).ravel()


class CsvSource(Source):
    """Replay a CSV file as a sample stream (historian export / file-drop / PoC)."""

    def __init__(self, path: str, delimiter: str = ",", has_header: bool = True,
                 columns: Optional[Sequence[int]] = None, max_rows: Optional[int] = None):
        self.path = path
        self.delimiter = delimiter
        self.has_header = has_header
        self.columns = list(columns) if columns is not None else None
        self.max_rows = max_rows

    def stream(self) -> Iterator[np.ndarray]:
        with open(self.path, newline="") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            if self.has_header:
                next(reader, None)
            n = 0
            for row in reader:
                if not row:
                    continue
                vals = [row[i] for i in self.columns] if self.columns is not None else row
                try:
                    x = np.asarray(vals, dtype=np.float32)
                except ValueError:
                    continue
                yield x
                n += 1
                if self.max_rows is not None and n >= self.max_rows:
                    return


class CallbackSource(Source):
    """Live source driven by a blocking `read()` callback (None = end)."""

    def __init__(self, read: Callable[[], Optional[np.ndarray]]):
        self._read = read

    def stream(self) -> Iterator[np.ndarray]:
        while True:
            s = self._read()
            if s is None:
                return
            yield np.asarray(s, dtype=np.float32).ravel()


class MqttSource(Source):
    """Subscribe to an MQTT topic; yield each payload as a float vector.
    Requires `paho-mqtt` (lazy import)."""

    def __init__(self, host: str, topic: str, port: int = 1883, qos: int = 0,
                 payload_parser: Optional[Callable[[bytes], np.ndarray]] = None,
                 queue_max: int = 10000):
        self.host, self.port, self.topic, self.qos = host, port, topic, qos
        self.queue_max = queue_max
        self._parser = payload_parser or (
            lambda b: np.asarray(b.decode().split(","), dtype=np.float32))
        self._client = None

    def stream(self) -> Iterator[np.ndarray]:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as e:
            raise ImportError("MqttSource requires `pip install paho-mqtt`") from e
        import queue
        q: "queue.Queue" = queue.Queue(maxsize=self.queue_max)

        def on_message(_c, _u, msg):
            try:
                q.put_nowait(self._parser(msg.payload))
            except Exception:
                pass

        self._client = mqtt.Client()
        self._client.on_message = on_message
        self._client.connect(self.host, self.port)
        self._client.subscribe(self.topic, qos=self.qos)
        self._client.loop_start()
        while True:
            yield q.get()

    def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()


class OpcUaSource(Source):
    """Poll an OPC-UA server node at a fixed interval. Requires `asyncua` (lazy import)."""

    def __init__(self, endpoint: str, node_ids: Sequence[str], period_s: float = 1.0):
        self.endpoint = endpoint
        self.node_ids = list(node_ids)
        self.period_s = period_s
        self._client = None

    def stream(self) -> Iterator[np.ndarray]:
        try:
            from asyncua.sync import Client
        except ImportError as e:
            raise ImportError("OpcUaSource requires `pip install asyncua`") from e
        import time
        self._client = Client(self.endpoint)
        self._client.connect()
        nodes = [self._client.get_node(nid) for nid in self.node_ids]
        try:
            while True:
                yield np.asarray([n.read_value() for n in nodes], dtype=np.float32)
                time.sleep(self.period_s)
        finally:
            self.close()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
