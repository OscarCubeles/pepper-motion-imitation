import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


EVALUATION_DIR = Path(__file__).resolve().parents[1]
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

MODULE_PATH = EVALUATION_DIR / "evaluation_results_app.py"
SPEC = importlib.util.spec_from_file_location("evaluation_results_app_under_test", MODULE_PATH)
evaluation_results_app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation_results_app
SPEC.loader.exec_module(evaluation_results_app)


class FrameComparisonTests(unittest.TestCase):
    def test_adjacent_frame_uses_available_frames_and_stops_at_bounds(self):
        frame_ids = [1, 3, 6]

        self.assertEqual(evaluation_results_app._adjacent_frame(frame_ids, 3, -1), 1)
        self.assertEqual(evaluation_results_app._adjacent_frame(frame_ids, 3, 1), 6)
        self.assertEqual(evaluation_results_app._adjacent_frame(frame_ids, 1, -1), 1)
        self.assertEqual(evaluation_results_app._adjacent_frame(frame_ids, 6, 1), 6)

    def test_adjacent_frame_defaults_to_first_available_frame(self):
        self.assertEqual(evaluation_results_app._adjacent_frame([2, 4], 99, 1), 2)
        self.assertIsNone(evaluation_results_app._adjacent_frame([], None, 1))

    def setUp(self):
        self.rows = pd.DataFrame([
            {
                "method": "ground_truth",
                "side": "left",
                "WOM": 1,
                "wom_orientation_label": "FRONT",
                "wom_orientation_detected_label": None,
                "wom_orientation_source": None,
                "angle_shoulder_pitch": 0.5,
                "angle_elbow_roll": -0.25,
            },
            {
                "method": "current_solution_constrained",
                "side": "left",
                "WOM": 0,
                "wom_orientation_label": "RIGHT",
                "wom_orientation_detected_label": None,
                "wom_orientation_source": "previous_mediapipe",
                "angle_shoulder_pitch": 0.6,
                "angle_elbow_roll": -0.5,
            },
        ])

    def test_wom_comparison_explains_previous_label_mismatch(self):
        comparison = evaluation_results_app._wom_comparison_rows(self.rows)
        constrained = comparison[comparison["Method"] == "Constrained"].iloc[0]

        self.assertEqual(constrained["Ground truth"], "FRONT")
        self.assertEqual(constrained["Effective orientation"], "RIGHT")
        self.assertEqual(constrained["Status"], "Mismatch")
        self.assertIn("retained previous orientation", constrained["Explanation"])

    def test_angle_comparison_reports_method_minus_ground_truth(self):
        comparison = evaluation_results_app._angle_comparison_rows(self.rows, "Radians")
        constrained = comparison[comparison["Method"] == "Constrained"].iloc[0]

        self.assertAlmostEqual(constrained["shoulder_pitch delta (rad)"], 0.1)
        self.assertAlmostEqual(constrained["elbow_roll delta (rad)"], -0.25)


class MatrixAggregationTests(unittest.TestCase):
    def test_long_aggregation_groups_video_category_method_and_metric(self):
        summary = pd.DataFrame([
            {"video_id": "video_001", "category": "Basketball", "method": "ikpy", "side": "left", "EEAr": 1.0, "WOM": 0.25},
            {"video_id": "video_001", "category": "Basketball", "method": "ikpy", "side": "right", "EEAr": 3.0, "WOM": 0.75},
            {"video_id": "video_002", "category": "Basketball", "method": "ikpy", "side": "left", "EEAr": 2.0, "WOM": 0.5},
        ])

        aggregated = evaluation_results_app._aggregate_ik_metric_long(
            summary,
            ["Basketball"],
            ["ikpy"],
            ["EEAr", "WOM"],
        )

        ee_ar = aggregated[aggregated["metric"] == "EEAr"].iloc[0]
        wom = aggregated[aggregated["metric"] == "WOM"].iloc[0]
        self.assertAlmostEqual(ee_ar["value"], 2.0)
        self.assertEqual(ee_ar["observation_count"], 3)
        self.assertEqual(ee_ar["normalized_value"], 0.5)
        self.assertEqual(wom["normalized_value"], 0.5)

    def test_normalization_direction_is_explicit(self):
        summary = pd.DataFrame([
            {"video_id": "video_001", "category": "A", "method": "ikpy", "side": "left", "EEAr": 1.0, "WOM": 0.2},
            {"video_id": "video_002", "category": "A", "method": "original_solution", "side": "left", "EEAr": 3.0, "WOM": 0.8},
        ])
        aggregated = evaluation_results_app._aggregate_ik_metric_long(
            summary,
            ["A"],
            ["ikpy", "original_solution"],
            ["EEAr", "WOM"],
        )
        values = aggregated.set_index(["method", "metric"])["normalized_value"]
        self.assertEqual(values["ikpy", "EEAr"], 1.0)
        self.assertEqual(values["original_solution", "EEAr"], 0.0)
        self.assertEqual(values["ikpy", "WOM"], 0.0)
        self.assertEqual(values["original_solution", "WOM"], 1.0)


if __name__ == "__main__":
    unittest.main()
