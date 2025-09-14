from __future__ import annotations
import sys
import time
import csv
import json
import logging
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple, Any
from collections import defaultdict, deque

import numpy as np
import requests
import serial

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
except Exception:
    InfluxDBClient = None
    Point = None
    WritePrecision = None

SERIAL_PORT = "COM3"
BAUDRATE = 115200
SERIAL_TIMEOUT = 2.0

MODEL_TFLITE = "model_int8_window16.tflite"
METADATA_JSON = "model_metadata.json"
CSV_FILE = "raw_data.csv"

WINDOW_SIZE = 16
MAX_BUFFERS = 256

INFLUX_ENABLED = False
INFLUX_URL = "http://127.0.0.1:8086"
INFLUX_TOKEN = ""
INFLUX_ORG = ""
INFLUX_BUCKET = "mycosense"

WEBHOOK_URL: Optional[str] = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("serial_infer")


@dataclass
class ModelMeta:
    mean: float
    std: float
    mse_threshold: float
    reg_a: float = 0.0
    reg_b: float = 0.0


class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        details = self.interpreter.get_input_details()
        self.input_detail = details[0]
        out_details = self.interpreter.get_output_details()
        self.output_detail = out_details[0]
        self.in_scale, self.in_zero = self.input_detail.get("quantization", (0.0, 0))
        self.out_scale, self.out_zero = self.output_detail.get("quantization", (0.0, 0))
        logger.info("Model loaded. input_dtype=%s quant=(%s,%s) output_dtype=%s quant=(%s,%s)",
                    self.input_detail['dtype'], self.in_scale, self.in_zero,
                    self.output_detail['dtype'], self.out_scale, self.out_zero)

    def infer_window(self, window: np.ndarray) -> Tuple[float, np.ndarray]:
        assert window.ndim == 1
        in_shape = list(self.input_detail["shape"])
        in_shape[0] = 1
        try:
            input_arr = window.reshape(in_shape).astype(np.float32)
        except Exception:
            input_arr = window.reshape(1, -1).astype(np.float32)

        if np.issubdtype(self.input_detail["dtype"], np.integer):
            if self.in_scale == 0:
                raise RuntimeError("Input quantization scale is 0; cannot quantize")
            q = np.round(input_arr / self.in_scale + self.in_zero).astype(self.input_detail["dtype"])
            self.interpreter.set_tensor(self.input_detail["index"], q)
        else:
            self.interpreter.set_tensor(self.input_detail["index"], input_arr)

        self.interpreter.invoke()
        out_raw = self.interpreter.get_tensor(self.output_detail["index"])

        if np.issubdtype(self.output_detail["dtype"], np.integer):
            out_deq = (out_raw.astype(np.float32) - self.out_zero) * self.out_scale
        else:
            out_deq = out_raw.astype(np.float32)

        in_float = input_arr.astype(np.float32)
        try:
            mse = float(np.mean((in_float - out_deq) ** 2))
        except Exception:
            minlen = min(in_float.size, out_deq.size)
            mse = float(np.mean((in_float.flat[:minlen] - out_deq.flat[:minlen]) ** 2))
        return mse, out_deq


class SerialReader:
    def __init__(self, port: str, baudrate: int, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.open()

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            logger.info("opened serial port %s @ %d", self.port, self.baudrate)
        except Exception as e:
            logger.exception("Failed to open serial port %s: %s", self.port, e)
            self.ser = None

    def readline(self) -> Optional[str]:
        if self.ser is None:
            self.open()
            if self.ser is None:
                time.sleep(1.0)
                return None
        try:
            raw = self.ser.readline()
            if not raw:
                return None
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            logger.warning("serial read error: %s attempting reconnect", e)
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            time.sleep(1.0)
            return None

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


def load_metadata(path: str) -> ModelMeta:
    with open(path, "r") as fh:
        d = json.load(fh)
    return ModelMeta(
        mean=float(d.get("mean", 0.0)),
        std=float(d.get("std", 1.0)),
        mse_threshold=float(d.get("mse_threshold", d.get("threshold", 0.0))),
        reg_a=float(d.get("reg_a", d.get("a", 0.0))),
        reg_b=float(d.get("reg_b", d.get("b", 0.0))),
    )


def safe_parse_line(line: str) -> Optional[Tuple[int, int, float]]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        ts = int(parts[0])
        node = int(parts[1])
        voltage = float(parts[2])
        return ts, node, voltage
    except Exception:
        return None


def append_csv_row(path: str, row: tuple):
    header = ["timestamp_ms", "node", "voltage", "mse", "anomaly"]
    write_header = not (path and __import__("os").path.exists(path))
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
        fh.flush()


def influx_write_point(client: Any, node: int, ts_ns: int, voltage: float, mse: float, anomaly: bool):
    if client is None:
        return
    p = Point("mycosense") \
        .tag("node", str(node)) \
        .field("voltage", float(voltage)) \
        .field("mse", float(mse)) \
        .field("anomaly", int(anomaly)) \
        .time(ts_ns, WritePrecision.NS)
    client.write_api().write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)


def main():
    logger.info("Loading metadata from %s", METADATA_JSON)
    meta = load_metadata(METADATA_JSON)
    model = TFLiteModel(MODEL_TFLITE)
    sr = SerialReader(SERIAL_PORT, BAUDRATE, SERIAL_TIMEOUT)
    windows: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    total_processed = 0
    influx_client = None
    if INFLUX_ENABLED:
        if InfluxDBClient is None:
            logger.error("influxDB client not installed, disable INFLUX_ENABLED or install influxdb-client.")
            sys.exit(1)
        influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        logger.info("Connected to InfluxDB at %s (bucket=%s)", INFLUX_URL, INFLUX_BUCKET)
    reqs = requests.Session()
    logger.info("Entering main loop, listening for serial data...")
    try:
        while True:
            raw = sr.readline()
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            parsed = safe_parse_line(line)
            if parsed is None:
                logger.debug("Skipping unparsable line: %s", line)
                continue
            ts_ms, node, voltage = parsed
            windows[node].append(float(voltage))
            total_processed += 1
            if len(windows[node]) >= WINDOW_SIZE:
                win = np.array(list(windows[node]), dtype=np.float32)
                win_norm = (win - meta.mean) / meta.std
                try:
                    mse, _ = model.infer_window(win_norm)
                except Exception as e:
                    logger.exception("Inference failed: %s", e)
                    continue
                anomaly = mse > meta.mse_threshold
                csv_row = (ts_ms, node, voltage, mse, int(anomaly))
                append_csv_row(CSV_FILE, csv_row)
                logger.info("node=%d voltage=%.6f mse=%.6f anomaly=%s", node, voltage, mse, anomaly)
                if INFLUX_ENABLED and influx_client is not None:
                    try:
                        ts_ns = int(ts_ms) * 1_000_000
                        influx_write_point(influx_client, node, ts_ns, voltage, mse, anomaly)
                    except Exception as e:
                        logger.warning("failed to write Influx point: %s", e)
                if WEBHOOK_URL:
                    payload = {"timestamp_ms": ts_ms, "node": node, "voltage": voltage, "mse": mse, "anomaly": bool(anomaly)}
                    try:
                        resp = reqs.post(WEBHOOK_URL, json=payload, timeout=2.0)
                        logger.debug("Webhook POST %s status=%s", WEBHOOK_URL, resp.status_code)
                    except Exception as e:
                        logger.warning("Webhook error: %s", e)
            if len(windows) > MAX_BUFFERS:
                keys = list(windows.keys())
                for k in keys[: len(keys) - MAX_BUFFERS]:
                    windows.pop(k, None)
                    logger.debug("Evicted window buffer for node %s", k)
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down.")
    finally:
        sr.close()
        if influx_client:
            try:
                influx_client.close()
            except Exception:
                pass
        logger.info("Shutdown complete. Processed %d lines.", total_processed)


if __name__ == "__main__":
    main()
