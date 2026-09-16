#!/usr/bin/env python3
"""Build reproducible, browser-ready analysis data for the IK dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SEED = 20260729
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_SAMPLES = 100_000

METHODS = (
    "current_solution_constrained",
    "current_solution_raw",
    "ikpy",
    "original_solution",
)
BASELINES = (
    "current_solution_raw",
    "ikpy",
    "original_solution",
)
QUALITY_METRICS = ("EEAr", "SOAx", "HJL", "WOM", "HJAr", "TSE", "SYN")
LOWER_IS_BETTER = {"EEAr", "SOAx", "HJL", "HJAr", "TSE", "SYN"}
PERFORMANCE_METRICS = (
    "latency_ms",
    "latency_jitter_ms",
    "cpu_percent",
    "gpu_percent",
    "memory_percent",
    "memory_mb",
)

METHOD_METADATA = {
    "current_solution_constrained": {
        "label": "Proposed constrained",
        "shortLabel": "Constrained",
        "role": "proposed",
        "color": "#006D77",
        "dash": "",
    },
    "current_solution_raw": {
        "label": "Proposed unconstrained",
        "shortLabel": "Unconstrained",
        "role": "ablation",
        "color": "#83A9A7",
        "dash": "6 4",
    },
    "ikpy": {
        "label": "IKPy",
        "shortLabel": "IKPy",
        "role": "baseline",
        "color": "#D97706",
        "dash": "2 3",
    },
    "original_solution": {
        "label": "Baseline",
        "shortLabel": "Baseline",
        "role": "baseline",
        "color": "#5B5F97",
        "dash": "10 4",
    },
}

METRIC_METADATA = {
    "EEAr": {
        "label": "Reference elbow elevation error",
        "unit": "normalized error",
        "direction": "lower",
        "definition": "Absolute difference between normalized human and reconstructed robot elbow elevation.",
    },
    "SOAx": {
        "label": "Soechting arm configuration error",
        "unit": "normalized distance",
        "direction": "lower",
        "definition": "Distance between normalized human and robot shoulder-to-elbow direction vectors.",
    },
    "HJL": {
        "label": "Human joint-limit violation ratio",
        "unit": "proportion",
        "direction": "lower",
        "definition": "Ratio of complete two-arm frames classified as non-human-doable.",
    },
    "WOM": {
        "label": "Wrist orientation match",
        "unit": "proportion",
        "direction": "higher",
        "definition": "Proportion of valid frames whose discrete wrist-orientation label matches the annotation.",
    },
    "HJAr": {
        "label": "Human joint-angle error",
        "unit": "normalized MSE",
        "direction": "lower",
        "definition": "Mean squared normalized difference between annotated and reproduced arm joint angles.",
    },
    "TSE": {
        "label": "Trajectory smoothness error",
        "unit": "squared radian difference",
        "direction": "lower",
        "definition": "Squared second-order joint-trajectory finite-difference error.",
    },
    "SYN": {
        "label": "Synergy reconstruction error",
        "unit": "descriptor distance",
        "direction": "lower",
        "definition": "Reconstruction error after projecting the robot posture into the learned human synergy space.",
    },
}

CATEGORY_LABELS = {
    "basketball_movements": "Basketball",
    "celebration_movements": "Celebration",
    "pepper_singularity_motions": "Lateral Arm Raising",
    "police_traffic": "Police traffic",
    "upper_body_jogging": "Jogging",
}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean_or_none(values: Iterable[float | None]):
    clean = [float(value) for value in values if finite(value)]
    return statistics.fmean(clean) if clean else None


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def paired_bootstrap_ci(
    differences: Sequence[float],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    if not differences:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(differences)
    estimates = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def sign_flip_p_value(
    differences: Sequence[float],
    samples: int = PERMUTATION_SAMPLES,
    seed: int = SEED,
) -> float:
    if not differences:
        return math.nan
    observed = abs(statistics.fmean(differences))
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(samples):
        statistic = abs(
            statistics.fmean(
                value if rng.getrandbits(1) else -value for value in differences
            )
        )
        exceedances += statistic >= observed - 1e-15
    return (exceedances + 1) / (samples + 1)


def rank_biserial(differences: Sequence[float], tolerance: float = 1e-12) -> float:
    nonzero = [value for value in differences if abs(value) > tolerance]
    if not nonzero:
        return 0.0
    ordered = sorted(enumerate(nonzero), key=lambda pair: abs(pair[1]))
    ranks = [0.0] * len(nonzero)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            abs(ordered[end][1]), abs(ordered[cursor][1]), rel_tol=0, abs_tol=tolerance
        ):
            end += 1
        average_rank = ((cursor + 1) + end) / 2
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if finite(value)]
    valid.sort(key=lambda pair: pair[1])
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted


def direction_adjusted_difference(metric: str, constrained: float, baseline: float) -> float:
    return baseline - constrained if metric in LOWER_IS_BETTER else constrained - baseline


def aggregate_arm_metric(summary: dict, method: str, metric: str):
    method_summary = summary.get(method, {})
    if metric == "HJL":
        # HJL is a complete two-arm frame indicator copied to both arm summaries.
        for side in ("left", "right"):
            value = method_summary.get(side, {}).get(metric)
            if finite(value):
                return float(value)
        return None
    return mean_or_none(
        method_summary.get(side, {}).get(metric) for side in ("left", "right")
    )


def metric_coverage(video: dict, method: str, metric: str) -> dict:
    valid = 0
    total = 0
    structural = 0
    frames = video.get("methods", {}).get(method, {}).get("frames", [])
    for frame in frames:
        for arm in frame.get("arms", {}).values():
            total += 1
            value = arm.get("metrics", {}).get(metric)
            if finite(value):
                valid += 1
            elif metric == "TSE" and frame.get("frame_id") in (0, 1):
                structural += 1
    return {
        "valid": valid,
        "total": total,
        "missing": total - valid,
        "structuralMissing": structural,
    }


def classify_video_singularities(video: dict) -> dict:
    """Summarize raw-solution singularity diagnostics at complete-frame level."""
    frames = video.get("methods", {}).get("current_solution_raw", {}).get("frames", [])
    expected_diagnostics = 0
    available_diagnostics = 0
    counts = {
        "singularFrameCount": 0,
        "dualFrameCount": 0,
        "elbowRollFrameCount": 0,
        "shoulderRollFrameCount": 0,
        "leftArmDualFrameCount": 0,
        "leftArmElbowRollFrameCount": 0,
        "leftArmShoulderRollFrameCount": 0,
        "leftArmNoSingularityFrameCount": 0,
        "rightArmDualFrameCount": 0,
        "rightArmElbowRollFrameCount": 0,
        "rightArmShoulderRollFrameCount": 0,
        "rightArmNoSingularityFrameCount": 0,
    }

    for frame in frames:
        frame_any = False
        frame_dual = False
        frame_elbow = False
        frame_shoulder = False
        arms = frame.get("arms", {})
        for side in ("left", "right"):
            expected_diagnostics += 1
            arm = arms.get(side, {})
            diagnostics = arm.get("metrics", {}).get("HJL_diagnostics")
            singularity = (
                diagnostics.get("singularity")
                if isinstance(diagnostics, dict)
                else None
            )
            if not isinstance(singularity, dict) or not isinstance(
                singularity.get("has_singularity"), bool
            ):
                continue

            available_diagnostics += 1
            if not singularity["has_singularity"]:
                counts[f"{side}ArmNoSingularityFrameCount"] += 1
                continue

            frame_any = True
            singularity_type = str(singularity.get("singularity_type") or "").lower()
            frame_dual = frame_dual or singularity_type.startswith("dual")
            frame_elbow = frame_elbow or singularity_type.startswith("elbow_roll")
            frame_shoulder = frame_shoulder or singularity_type.startswith("shoulder_roll")
            if singularity_type.startswith("dual"):
                counts[f"{side}ArmDualFrameCount"] += 1
            elif singularity_type.startswith("elbow_roll"):
                counts[f"{side}ArmElbowRollFrameCount"] += 1
            elif singularity_type.startswith("shoulder_roll"):
                counts[f"{side}ArmShoulderRollFrameCount"] += 1

        counts["singularFrameCount"] += int(frame_any)
        counts["dualFrameCount"] += int(frame_dual)
        counts["elbowRollFrameCount"] += int(frame_elbow)
        counts["shoulderRollFrameCount"] += int(frame_shoulder)

    classification_available = (
        expected_diagnostics > 0 and available_diagnostics == expected_diagnostics
    )
    return {
        "videoId": video.get("video_id"),
        "category": video.get("category"),
        "categoryLabel": CATEGORY_LABELS.get(video.get("category"), video.get("category")),
        "classificationAvailable": classification_available,
        "hasAny": counts["singularFrameCount"] > 0,
        "hasDual": counts["dualFrameCount"] > 0,
        "hasElbowRoll": counts["elbowRollFrameCount"] > 0,
        "hasShoulderRoll": counts["shoulderRollFrameCount"] > 0,
        "totalFrameCount": len(frames),
        "diagnosticArmCount": available_diagnostics,
        "expectedDiagnosticArmCount": expected_diagnostics,
        **counts,
    }


def build_quality(ik_results: dict):
    quality_rows = []
    trajectory_rows = []
    video_records = []
    source_videos = ik_results.get("videos", [])
    for video in source_videos:
        video_id = video.get("video_id")
        category = video.get("category")
        record = {
            "videoId": video_id,
            "category": category,
            "categoryLabel": CATEGORY_LABELS.get(category, category),
            "methods": {},
        }
        for method in METHODS:
            metrics = {
                metric: aggregate_arm_metric(video.get("summary", {}), method, metric)
                for metric in QUALITY_METRICS
            }
            coverage = {
                metric: metric_coverage(video, method, metric)
                for metric in ("WOM", "TSE")
            }
            record["methods"][method] = {"metrics": metrics, "coverage": coverage}
            for metric, value in metrics.items():
                quality_rows.append(
                    {
                        "videoId": video_id,
                        "category": category,
                        "categoryLabel": CATEGORY_LABELS.get(category, category),
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
            for frame in video.get("methods", {}).get(method, {}).get("frames", []):
                for side, arm in frame.get("arms", {}).items():
                    row = {
                        "videoId": video_id,
                        "category": category,
                        "method": method,
                        "frameId": frame.get("frame_id"),
                        "timestamp": frame.get("timestamp"),
                        "side": side,
                    }
                    for metric in QUALITY_METRICS:
                        row[metric] = arm.get("metrics", {}).get(metric)
                    trajectory_rows.append(row)
        video_records.append(record)
    video_singularities = [classify_video_singularities(video) for video in source_videos]
    return video_records, quality_rows, trajectory_rows, video_singularities


def build_comparisons(video_records: list[dict]):
    comparisons = []
    comparison_index = 0
    for baseline in BASELINES:
        for metric in QUALITY_METRICS:
            differences = []
            for video in video_records:
                constrained = video["methods"]["current_solution_constrained"]["metrics"][metric]
                baseline_value = video["methods"][baseline]["metrics"][metric]
                if finite(constrained) and finite(baseline_value):
                    differences.append(
                        direction_adjusted_difference(
                            metric, float(constrained), float(baseline_value)
                        )
                    )
            tolerance = 1e-12
            wins = sum(value > tolerance for value in differences)
            losses = sum(value < -tolerance for value in differences)
            ties = len(differences) - wins - losses
            ci_low, ci_high = paired_bootstrap_ci(
                differences, seed=SEED + comparison_index * 17
            )
            comparison = {
                "baseline": baseline,
                "metric": metric,
                "n": len(differences),
                "meanDifference": statistics.fmean(differences) if differences else None,
                "medianDifference": statistics.median(differences) if differences else None,
                "ciLow": ci_low,
                "ciHigh": ci_high,
                "rankBiserial": rank_biserial(differences),
                "pValue": sign_flip_p_value(
                    differences, seed=SEED + comparison_index * 37
                ),
                "pAdjusted": None,
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
            comparisons.append(comparison)
            comparison_index += 1
    adjusted = holm_adjust([row["pValue"] for row in comparisons])
    for row, adjusted_value in zip(comparisons, adjusted):
        row["pAdjusted"] = adjusted_value
    return comparisons


def find_identical_series(video_records: list[dict]):
    identical = []
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1 :]:
            metrics = []
            for metric in QUALITY_METRICS:
                pairs = [
                    (
                        video["methods"][left]["metrics"][metric],
                        video["methods"][right]["metrics"][metric],
                    )
                    for video in video_records
                ]
                if all(
                    finite(a)
                    and finite(b)
                    and math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
                    for a, b in pairs
                ):
                    metrics.append(metric)
            if metrics:
                identical.append({"left": left, "right": right, "metrics": metrics})
    return identical


def build_performance(performance_results: dict):
    repeat_rows = []
    latency_samples = []
    frame_rows = []
    grouped_video = defaultdict(list)
    for video in performance_results.get("videos", []):
        video_id = video.get("video_id")
        category = video.get("category")
        for method in METHODS:
            frames = video.get("methods", {}).get(method, {}).get("frames", [])
            repeats = defaultdict(list)
            for frame in frames:
                repeats[frame.get("repeat_index", 0)].append(frame)
                frame_rows.append(
                    {
                        "videoId": video_id,
                        "category": category,
                        "method": method,
                        "repeatIndex": frame.get("repeat_index", 0),
                        "frameId": frame.get("frame_id"),
                        "timestamp": frame.get("timestamp"),
                        "latency_ms": frame.get("latency_ms"),
                        "cpu_percent": frame.get("cpu_percent"),
                        "gpu_percent": frame.get("gpu_percent"),
                        "memory_percent": frame.get("memory_percent"),
                        "memory_mb": frame.get("memory_mb"),
                    }
                )
                latency = frame.get("latency_ms")
                if finite(latency):
                    latency_samples.append(
                        {
                            "videoId": video_id,
                            "category": category,
                            "method": method,
                            "repeatIndex": frame.get("repeat_index", 0),
                            "frameId": frame.get("frame_id"),
                            "value": latency,
                        }
                    )
            for repeat_index, repeat_frames in sorted(repeats.items()):
                latency_values = [
                    frame.get("latency_ms")
                    for frame in repeat_frames
                    if finite(frame.get("latency_ms"))
                ]
                row = {
                    "videoId": video_id,
                    "category": category,
                    "categoryLabel": CATEGORY_LABELS.get(category, category),
                    "method": method,
                    "repeatIndex": repeat_index,
                    "frameCount": len(repeat_frames),
                    "latency_ms": mean_or_none(latency_values),
                    "latency_jitter_ms": statistics.pstdev(latency_values)
                    if len(latency_values) > 1
                    else 0.0,
                }
                for metric in PERFORMANCE_METRICS[2:]:
                    row[metric] = mean_or_none(frame.get(metric) for frame in repeat_frames)
                repeat_rows.append(row)
                grouped_video[(video_id, category, method)].append(row)
    video_rows = []
    for (video_id, category, method), repeats in grouped_video.items():
        row = {
            "videoId": video_id,
            "category": category,
            "categoryLabel": CATEGORY_LABELS.get(category, category),
            "method": method,
            "repeatCount": len(repeats),
        }
        for metric in PERFORMANCE_METRICS:
            row[metric] = mean_or_none(repeat_row.get(metric) for repeat_row in repeats)
        video_rows.append(row)
    return repeat_rows, video_rows, latency_samples, frame_rows


def write_csv(path: Path, rows: list[dict], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_analysis(ik_path: Path, performance_path: Path, output_dir: Path):
    ik_results = json.loads(ik_path.read_text(encoding="utf-8"))
    performance_results = json.loads(performance_path.read_text(encoding="utf-8"))
    video_records, quality_rows, trajectory_rows, video_singularities = build_quality(ik_results)
    comparisons = build_comparisons(video_records)
    repeat_rows, performance_rows, latency_samples, performance_frame_rows = build_performance(
        performance_results
    )
    categories = sorted(
        {video["category"] for video in video_records},
        key=lambda value: list(CATEGORY_LABELS).index(value)
        if value in CATEGORY_LABELS
        else value,
    )
    analysis = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "bootstrapSamples": BOOTSTRAP_SAMPLES,
            "permutationSamples": PERMUTATION_SAMPLES,
            "videoCount": len(video_records),
            "frameCount": sum(
                len(
                    video.get("methods", {})
                    .get("current_solution_constrained", {})
                    .get("frames", [])
                )
                for video in ik_results.get("videos", [])
            ),
            "categoryCount": len(categories),
            "primaryUnit": "video",
            "sources": [
                "04-evaluation/results/ik_method_metrics.json",
                "04-evaluation/results/performance_metrics.json",
            ],
            "caveats": [
                "Inferential statistics use 25 paired videos; frames are not treated as independent experimental units.",
                "Category panels are descriptive because each category contains five videos.",
                "Performance results benchmark offline IK computation only and exclude capture, pose inference, networking, and robot actuation.",
                "Current-solution WOM uses the current MediaPipe label, the previous valid per-hand label when detection is missing, or a zero-WristYaw forward-kinematics fallback before the first valid detection.",
                "HJAr references stored annotation angles; results are not independent when those angles were generated by the proposed solution.",
                "Video singularity filters use current_solution_raw HJL diagnostics and retain complete-video metric summaries rather than selecting individual frames.",
            ],
        },
        "methods": [
            {"id": method, **METHOD_METADATA[method]} for method in METHODS
        ],
        "metrics": [
            {"id": metric, **METRIC_METADATA[metric]} for metric in QUALITY_METRICS
        ],
        "performanceMetrics": [
            {
                "id": metric,
                "label": {
                    "latency_ms": "Latency",
                    "latency_jitter_ms": "Latency jitter",
                    "cpu_percent": "CPU use",
                    "gpu_percent": "GPU use",
                    "memory_percent": "Memory use",
                    "memory_mb": "Memory",
                }[metric],
                "unit": {
                    "latency_ms": "ms",
                    "latency_jitter_ms": "ms",
                    "cpu_percent": "%",
                    "gpu_percent": "%",
                    "memory_percent": "%",
                    "memory_mb": "MiB",
                }[metric],
                "direction": "lower",
            }
            for metric in PERFORMANCE_METRICS
        ],
        "categories": [
            {"id": category, "label": CATEGORY_LABELS.get(category, category)}
            for category in categories
        ],
        "videos": video_records,
        "videoSingularities": video_singularities,
        "qualityRows": quality_rows,
        "pairedComparisons": comparisons,
        "identicalSeries": find_identical_series(video_records),
        "trajectoryRows": trajectory_rows,
        "performanceRepeats": repeat_rows,
        "performanceVideos": performance_rows,
        "performanceFrames": performance_frame_rows,
        "latencySamples": latency_samples,
        "sourceNotes": {
            "ik": ik_results.get("notes", []),
            "performance": performance_results.get("notes", []),
            "environment": performance_results.get("environment", {}),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "video_quality.csv",
        quality_rows,
        ("videoId", "category", "categoryLabel", "method", "metric", "value"),
    )
    write_csv(
        output_dir / "video_singularities.csv",
        video_singularities,
        (
            "videoId",
            "category",
            "categoryLabel",
            "classificationAvailable",
            "hasAny",
            "hasDual",
            "hasElbowRoll",
            "hasShoulderRoll",
            "totalFrameCount",
            "singularFrameCount",
            "dualFrameCount",
            "elbowRollFrameCount",
            "shoulderRollFrameCount",
            "leftArmDualFrameCount",
            "leftArmElbowRollFrameCount",
            "leftArmShoulderRollFrameCount",
            "leftArmNoSingularityFrameCount",
            "rightArmDualFrameCount",
            "rightArmElbowRollFrameCount",
            "rightArmShoulderRollFrameCount",
            "rightArmNoSingularityFrameCount",
            "diagnosticArmCount",
            "expectedDiagnosticArmCount",
        ),
    )
    write_csv(
        output_dir / "paired_comparisons.csv",
        comparisons,
        (
            "baseline",
            "metric",
            "n",
            "meanDifference",
            "medianDifference",
            "ciLow",
            "ciHigh",
            "rankBiserial",
            "pValue",
            "pAdjusted",
            "wins",
            "ties",
            "losses",
        ),
    )
    write_csv(
        output_dir / "performance_repeats.csv",
        repeat_rows,
        (
            "videoId",
            "category",
            "categoryLabel",
            "method",
            "repeatIndex",
            "frameCount",
            *PERFORMANCE_METRICS,
        ),
    )
    return analysis


def parse_args():
    dashboard_root = Path(__file__).resolve().parents[1]
    evaluation_root = dashboard_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ik",
        type=Path,
        default=evaluation_root / "results" / "ik_method_metrics.json",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=evaluation_root / "results" / "performance_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dashboard_root / "public" / "data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analysis = build_analysis(args.ik.resolve(), args.performance.resolve(), args.output_dir)
    print(
        f"Wrote {args.output_dir / 'analysis.json'} "
        f"({analysis['metadata']['videoCount']} videos, "
        f"{len(analysis['pairedComparisons'])} paired comparisons)."
    )


if __name__ == "__main__":
    main()
