import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_analysis.py"
SPEC = importlib.util.spec_from_file_location("build_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def singularity_arm(has_singularity, singularity_type=None):
        return {
            "metrics": {
                "HJL_diagnostics": {
                    "singularity": {
                        "has_singularity": has_singularity,
                        "singularity_type": singularity_type,
                    }
                }
            }
        }

    def test_metric_direction(self):
        self.assertAlmostEqual(
            analysis.direction_adjusted_difference("EEAr", 0.2, 0.5), 0.3
        )
        self.assertAlmostEqual(
            analysis.direction_adjusted_difference("WOM", 0.8, 0.5), 0.3
        )

    def test_hjl_is_not_averaged_twice(self):
        summary = {
            "method": {
                "left": {"HJL": 0.25},
                "right": {"HJL": 0.25},
            }
        }
        self.assertEqual(analysis.aggregate_arm_metric(summary, "method", "HJL"), 0.25)

    def test_arm_metrics_are_averaged(self):
        summary = {
            "method": {
                "left": {"EEAr": 0.2},
                "right": {"EEAr": 0.6},
            }
        }
        self.assertAlmostEqual(
            analysis.aggregate_arm_metric(summary, "method", "EEAr"), 0.4
        )

    def test_tse_structural_missingness(self):
        video = {
            "methods": {
                "method": {
                    "frames": [
                        {
                            "frame_id": frame_id,
                            "arms": {
                                "left": {"metrics": {"TSE": None if frame_id < 2 else 0.1}},
                                "right": {"metrics": {"TSE": None if frame_id < 2 else 0.2}},
                            },
                        }
                        for frame_id in range(4)
                    ]
                }
            }
        }
        coverage = analysis.metric_coverage(video, "method", "TSE")
        self.assertEqual(coverage, {"valid": 4, "total": 8, "missing": 4, "structuralMissing": 4})

    def test_bootstrap_is_deterministic(self):
        values = [0.1, 0.2, 0.3, -0.1]
        first = analysis.paired_bootstrap_ci(values, samples=500, seed=17)
        second = analysis.paired_bootstrap_ci(values, samples=500, seed=17)
        self.assertEqual(first, second)

    def test_holm_adjustment(self):
        adjusted = analysis.holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_rank_biserial_ties(self):
        self.assertEqual(analysis.rank_biserial([0.0, 0.0]), 0.0)
        self.assertGreater(analysis.rank_biserial([1.0, 2.0, -0.1]), 0)

    def test_repeat_aggregation(self):
        source = {
            "videos": [
                {
                    "video_id": "video_001",
                    "category": "basketball_movements",
                    "methods": {
                        method: {
                            "frames": [
                                {
                                    "repeat_index": 0,
                                    "frame_id": 0,
                                    "latency_ms": value,
                                    "cpu_percent": 10.0,
                                    "gpu_percent": 0.0,
                                    "memory_percent": 2.0,
                                    "memory_mb": 600.0,
                                }
                                for value in (1.0, 3.0)
                            ]
                        }
                        for method in analysis.METHODS
                    },
                }
            ]
        }
        repeats, videos, samples, frames = analysis.build_performance(source)
        self.assertEqual(len(repeats), 4)
        self.assertEqual(len(videos), 4)
        self.assertEqual(len(samples), 8)
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0]["latency_ms"], 1.0)
        self.assertEqual(repeats[0]["latency_ms"], 2.0)
        self.assertEqual(repeats[0]["latency_jitter_ms"], 1.0)

    def test_video_singularity_classification_deduplicates_bilateral_frames(self):
        video = {
            "video_id": "video_001",
            "category": "test_category",
            "methods": {
                "current_solution_raw": {
                    "frames": [
                        {
                            "arms": {
                                "left": self.singularity_arm(True, "dual (singular)"),
                                "right": self.singularity_arm(True, "dual (singular)"),
                            }
                        },
                        {
                            "arms": {
                                "left": self.singularity_arm(True, "elbow_roll (singular)"),
                                "right": self.singularity_arm(True, "shoulder_roll (singular)"),
                            }
                        },
                    ]
                }
            },
        }
        result = analysis.classify_video_singularities(video)
        self.assertTrue(result["classificationAvailable"])
        self.assertTrue(result["hasAny"])
        self.assertTrue(result["hasDual"])
        self.assertTrue(result["hasElbowRoll"])
        self.assertTrue(result["hasShoulderRoll"])
        self.assertEqual(result["singularFrameCount"], 2)
        self.assertEqual(result["dualFrameCount"], 1)
        self.assertEqual(result["elbowRollFrameCount"], 1)
        self.assertEqual(result["shoulderRollFrameCount"], 1)
        self.assertEqual(result["leftArmDualFrameCount"], 1)
        self.assertEqual(result["leftArmElbowRollFrameCount"], 1)
        self.assertEqual(result["leftArmShoulderRollFrameCount"], 0)
        self.assertEqual(result["leftArmNoSingularityFrameCount"], 0)
        self.assertEqual(result["rightArmDualFrameCount"], 1)
        self.assertEqual(result["rightArmElbowRollFrameCount"], 0)
        self.assertEqual(result["rightArmShoulderRollFrameCount"], 1)
        self.assertEqual(result["rightArmNoSingularityFrameCount"], 0)
        for side in ("left", "right"):
            arm_total = sum(
                result[f"{side}Arm{kind}FrameCount"]
                for kind in ("Dual", "ElbowRoll", "ShoulderRoll", "NoSingularity")
            )
            self.assertEqual(arm_total, result["totalFrameCount"])

    def test_video_without_singularity_is_classified_only_when_diagnostics_complete(self):
        complete = {
            "methods": {
                "current_solution_raw": {
                    "frames": [{"arms": {
                        "left": self.singularity_arm(False),
                        "right": self.singularity_arm(False),
                    }}]
                }
            }
        }
        unavailable = {
            "methods": {
                "current_solution_raw": {
                    "frames": [{"arms": {
                        "left": self.singularity_arm(False),
                        "right": {"metrics": {}},
                    }}]
                }
            }
        }
        complete_result = analysis.classify_video_singularities(complete)
        unavailable_result = analysis.classify_video_singularities(unavailable)
        self.assertTrue(complete_result["classificationAvailable"])
        self.assertFalse(complete_result["hasAny"])
        self.assertFalse(unavailable_result["classificationAvailable"])
        self.assertFalse(unavailable_result["hasAny"])

    def test_generated_dataset_has_expected_experimental_units(self):
        data_path = Path(__file__).resolve().parents[1] / "public" / "data" / "analysis.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(data["metadata"]["videoCount"], 25)
        self.assertEqual(data["metadata"]["categoryCount"], 5)
        self.assertEqual(len(data["methods"]), 4)
        self.assertEqual(len(data["metrics"]), 7)
        self.assertEqual(len(data["qualityRows"]), 25 * 4 * 7)
        singularities = data["videoSingularities"]
        self.assertEqual(len(singularities), 25)
        self.assertEqual(sum(row["hasAny"] for row in singularities), 23)
        self.assertEqual(sum(row["hasDual"] for row in singularities), 6)
        self.assertEqual(sum(row["hasElbowRoll"] for row in singularities), 18)
        self.assertEqual(sum(row["hasShoulderRoll"] for row in singularities), 6)
        self.assertEqual(
            sum(row["classificationAvailable"] and not row["hasAny"] for row in singularities),
            2,
        )
        self.assertTrue(all(row["classificationAvailable"] for row in singularities))
        self.assertEqual(len(data["pairedComparisons"]), 3 * 7)
        self.assertTrue(all(row["n"] == 25 for row in data["pairedComparisons"]))
        self.assertEqual(len(data["performanceRepeats"]), 25 * 4 * 3)
        self.assertEqual(len(data["performanceVideos"]), 25 * 4)
        self.assertEqual(len(data["latencySamples"]), 25 * 4 * 3 * 24)
        self.assertEqual(len(data["performanceFrames"]), 25 * 4 * 3 * 24)
        self.assertEqual(
            set(data["performanceFrames"][0]).intersection(
                {"latency_ms", "cpu_percent", "gpu_percent", "memory_percent", "memory_mb"}
            ),
            {"latency_ms", "cpu_percent", "gpu_percent", "memory_percent", "memory_mb"},
        )

    def test_generated_coverage_distinguishes_missing_from_structural(self):
        data_path = Path(__file__).resolve().parents[1] / "public" / "data" / "analysis.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        constrained = data["videos"][0]["methods"]["current_solution_constrained"]
        self.assertEqual(constrained["coverage"]["TSE"]["structuralMissing"], 4)
        self.assertGreaterEqual(constrained["coverage"]["WOM"]["missing"], 0)


if __name__ == "__main__":
    unittest.main()
