import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path

import psutil


CURRENT_FILE = Path(__file__).resolve()
DL_POSE_DIR = CURRENT_FILE.parents[1]
SERVER_DIR = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[3]

from server.baseline.evaluation import evaluate_ik_methods as ik_eval  # noqa: E402


METHODS = ("current_solution_raw", "current_solution_constrained", "ikpy", "original_solution")
PERFORMANCE_METRICS = (
    "latency_ms",
    "latency_jitter_ms",
    "cpu_percent",
    "gpu_percent",
    "memory_percent",
    "memory_mb",
)


class GpuSampler:
    def __init__(self):
        self.available = False
        self.error = None
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def sample_percent(self):
        if not self.available:
            return None
        try:
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(utilization.gpu)
        except Exception:
            return None

    def close(self):
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


def _load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def _json_safe(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _mean(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _stdev(values):
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return 0.0 if clean else None
    return float(statistics.pstdev(clean))


def _process_cpu_percent(process, start_cpu, end_cpu, elapsed_sec, cpu_count):
    if elapsed_sec <= 0:
        return None
    cpu_delta = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
    divisor = max(1, int(cpu_count or 1))
    return max(0.0, 100.0 * cpu_delta / (elapsed_sec * divisor))


def _process_memory_sample(process, total_memory_bytes):
    try:
        rss_bytes = float(process.memory_info().rss)
    except Exception:
        return None, None

    memory_mb = rss_bytes / (1024.0 * 1024.0)
    memory_percent = None
    if total_memory_bytes:
        memory_percent = 100.0 * rss_bytes / float(total_memory_bytes)
    return memory_percent, memory_mb


def _evaluate_method_frame(method_name, frame_data, method_state, chains, joint_indices):
    for side in ik_eval.ARM_SIDES:
        ik_eval._method_result(
            method_name,
            frame_data,
            side,
            chains=chains,
            joint_indices=joint_indices,
            method_state=method_state,
        )


def _frame_performance(
    method_name,
    frame_data,
    method_state,
    chains,
    joint_indices,
    process,
    gpu_sampler,
    cpu_count,
    total_memory_bytes,
):
    start_cpu = process.cpu_times()
    gpu_before = gpu_sampler.sample_percent()
    memory_percent_before, memory_mb_before = _process_memory_sample(process, total_memory_bytes)
    start_time = time.perf_counter()
    error = None

    try:
        _evaluate_method_frame(method_name, frame_data, method_state, chains, joint_indices)
    except Exception as exc:
        error = str(exc)

    end_time = time.perf_counter()
    gpu_after = gpu_sampler.sample_percent()
    memory_percent_after, memory_mb_after = _process_memory_sample(process, total_memory_bytes)
    end_cpu = process.cpu_times()

    elapsed_sec = end_time - start_time
    gpu_values = [value for value in (gpu_before, gpu_after) if value is not None]
    memory_percent_values = [
        value for value in (memory_percent_before, memory_percent_after) if value is not None
    ]
    memory_mb_values = [value for value in (memory_mb_before, memory_mb_after) if value is not None]

    return {
        "latency_ms": elapsed_sec * 1000.0,
        "cpu_percent": _process_cpu_percent(process, start_cpu, end_cpu, elapsed_sec, cpu_count),
        "gpu_percent": _mean(gpu_values),
        "memory_percent": _mean(memory_percent_values),
        "memory_mb": _mean(memory_mb_values),
        "error": error,
    }


def _summarize_frames(frames):
    latency_values = [frame.get("latency_ms") for frame in frames if frame.get("error") is None]
    cpu_values = [frame.get("cpu_percent") for frame in frames if frame.get("error") is None]
    gpu_values = [frame.get("gpu_percent") for frame in frames if frame.get("error") is None]
    memory_percent_values = [frame.get("memory_percent") for frame in frames if frame.get("error") is None]
    memory_mb_values = [frame.get("memory_mb") for frame in frames if frame.get("error") is None]
    errors = [frame for frame in frames if frame.get("error")]

    return {
        "frame_count": len(frames),
        "error_count": len(errors),
        "latency_ms": _mean(latency_values),
        "latency_jitter_ms": _stdev(latency_values),
        "cpu_percent": _mean(cpu_values),
        "gpu_percent": _mean(gpu_values),
        "memory_percent": _mean(memory_percent_values),
        "memory_mb": _mean(memory_mb_values),
    }


def evaluate_file(
    path,
    methods,
    chains=None,
    joint_indices=None,
    process=None,
    gpu_sampler=None,
    repeats=1,
    cpu_count=None,
    total_memory_bytes=None,
):
    annotations = _load_json(path)
    video_result = {
        "video_id": annotations.get("video_id", path.parent.name),
        "category": annotations.get("category", path.parent.parent.name),
        "annotation_file": str(path),
        "methods": {},
    }

    frames = annotations.get("frames", [])
    if any(method in methods for method in ("current_solution_raw", "current_solution_constrained")):
        ik_eval._populate_runtime_orientation_labels(
            frames,
            path.parent,
            annotations.get("fps", 8),
        )
    for method_name in methods:
        method_state = {}
        if method_name == "current_solution_constrained":
            method_state = {
                "pose_handler": ik_eval.PoseHandler(),
                "classifier": ik_eval.AngleClassifier(),
            }

        method_frames = []
        for repeat_index in range(repeats):
            for frame_data in frames:
                frame_metrics = _frame_performance(
                    method_name,
                    frame_data,
                    method_state,
                    chains,
                    joint_indices,
                    process,
                    gpu_sampler,
                    cpu_count,
                    total_memory_bytes,
                )
                method_frames.append(
                    {
                        "repeat_index": repeat_index,
                        "frame_id": frame_data.get("frame_id"),
                        "timestamp": frame_data.get("timestamp"),
                        "image_file": frame_data.get("image_file"),
                        **frame_metrics,
                    }
                )

        video_result["methods"][method_name] = {
            "frames": method_frames,
            "summary": _summarize_frames(method_frames),
        }

    return video_result


def _summarize_dataset(videos, methods):
    summary = {}
    for method_name in methods:
        frames = []
        for video in videos:
            frames.extend(video.get("methods", {}).get(method_name, {}).get("frames", []))
        summary[method_name] = _summarize_frames(frames)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute offline system performance metrics for annotated videos."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "dataset" / "pepper_singularity_motions",
        help="annotations_filled.json, video folder, category folder, or dataset folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "04-evaluation" / "results" / "performance_metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ik_eval.METHODS,
        default=list(METHODS),
        help="Methods to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of offline replay passes per video.",
    )
    parser.add_argument(
        "--max-video-index",
        type=int,
        default=None,
        help=(
            "Benchmark only video_XXX folders whose numeric index is at or below "
            "this value, independently in every category."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = ik_eval._annotation_files(args.input)
    if not files:
        raise RuntimeError(f"No annotations_filled.json files found under {args.input}")
    files = ik_eval._filter_annotation_files_by_max_video_index(
        files,
        args.max_video_index,
    )
    if not files:
        raise RuntimeError("No annotation files remain after applying the video-index limit.")
    if args.max_video_index is not None:
        print(
            f"[info] benchmarking video indices through video_{args.max_video_index:03d}: "
            f"{len(files)} annotation files"
        )
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    chains = None
    joint_indices = {
        "rwri_smpl": 4,
        "lwri_smpl": 7,
        "thor_smpl": 8,
    }
    if any(method in args.methods for method in ("ikpy", "original_solution")):
        with contextlib.redirect_stdout(io.StringIO()):
            chains = ik_eval.ikpyu.load_pepper_chains(str(DL_POSE_DIR))

    process = psutil.Process()
    cpu_count = psutil.cpu_count(logical=True) or 1
    total_memory_bytes = psutil.virtual_memory().total
    gpu_sampler = GpuSampler()

    results = {
        "dataset": str(args.input),
        "output_file": str(args.output),
        "benchmark_type": "offline_replay_from_annotations",
        "metrics": {
            "latency_ms": "Average per-frame time from offline frame acquisition start to method output generation.",
            "latency_jitter_ms": "Population standard deviation of per-frame latency_ms values.",
            "cpu_percent": "Average process CPU utilization during method execution as a percentage of total logical CPU capacity.",
            "gpu_percent": "Average GPU utilization sampled via NVML during method execution; null if unavailable.",
            "memory_percent": "Average process RSS memory during method execution as a percentage of total system RAM.",
            "memory_mb": "Average process RSS memory during method execution in MiB.",
        },
        "notes": [
            "This script replays annotations_filled.json frames offline; it does not measure camera capture, Metrabs inference, WebSocket transfer, or real Pepper setAngles execution.",
            "Latency is measured with time.perf_counter around the method computation for both arms of each frame.",
            "CPU usage is computed from process CPU time deltas over each measured frame and divided by the number of logical CPU cores.",
            "GPU usage is sampled through pynvml/NVML when available; IK-only offline code may show 0 or null even if the live pose-estimation pipeline uses GPU.",
            "Memory usage is sampled from psutil Process.memory_info().rss before and after each frame computation.",
        ],
        "environment": {
            "python": sys.version,
            "cpu_logical_count": cpu_count,
            "cpu_physical_count": psutil.cpu_count(logical=False),
            "system_memory_total_bytes": total_memory_bytes,
            "gpu_sampler_available": gpu_sampler.available,
            "gpu_sampler_error": gpu_sampler.error,
        },
        "videos": [],
    }

    try:
        for path in files:
            print(f"[info] benchmarking {path}")
            results["videos"].append(
                evaluate_file(
                    path,
                    args.methods,
                    chains=chains,
                    joint_indices=joint_indices,
                    process=process,
                    gpu_sampler=gpu_sampler,
                    repeats=args.repeats,
                    cpu_count=cpu_count,
                    total_memory_bytes=total_memory_bytes,
                )
            )
    finally:
        gpu_sampler.close()

    results["summary"] = _summarize_dataset(results["videos"], args.methods)
    _save_json(args.output, _json_safe(results))
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
