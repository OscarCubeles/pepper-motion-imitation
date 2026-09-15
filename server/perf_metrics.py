import time


def make_send_metrics():
    return {
        "count": 0,
        "latency_sum_sec": 0.0,
        "latency_samples": 0,
    }


def record_send_metrics(send_metrics: dict, send_metrics_lock, frame_ready_ts):
    now = time.perf_counter()
    with send_metrics_lock:
        send_metrics["count"] += 1
        if frame_ready_ts is not None:
            send_metrics["latency_sum_sec"] += max(0.0, now - float(frame_ready_ts))
            send_metrics["latency_samples"] += 1


def consume_send_metrics(send_metrics: dict, send_metrics_lock):
    with send_metrics_lock:
        count = int(send_metrics["count"])
        latency_sum_sec = float(send_metrics["latency_sum_sec"])
        latency_samples = int(send_metrics["latency_samples"])
        send_metrics["count"] = 0
        send_metrics["latency_sum_sec"] = 0.0
        send_metrics["latency_samples"] = 0

    avg_latency_ms = None
    if latency_samples > 0:
        avg_latency_ms = (latency_sum_sec * 1000.0) / latency_samples
    return count, avg_latency_ms
