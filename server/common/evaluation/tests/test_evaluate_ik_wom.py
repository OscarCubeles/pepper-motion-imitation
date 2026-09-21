import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


EVALUATION_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = EVALUATION_DIR / "evaluate_ik_methods.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ik_methods_under_test", MODULE_PATH)
evaluate_ik_methods = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate_ik_methods
SPEC.loader.exec_module(evaluate_ik_methods)


class CurrentSolutionWomFallbackTests(unittest.TestCase):
    def setUp(self):
        self.angles = {
            "shoulder_pitch": 0.1,
            "shoulder_roll": 0.2,
            "elbow_yaw": 0.3,
            "elbow_roll": 0.4,
            "wrist_yaw": 1.2,
        }

    def test_valid_mediapipe_label_is_used_and_retained(self):
        previous = {}

        label, source = evaluate_ik_methods._resolve_current_wom_orientation(
            "left",
            self.angles,
            "left",
            chains=None,
            previous_labels=previous,
        )

        self.assertEqual(label, "LEFT")
        self.assertEqual(source, "mediapipe")
        self.assertEqual(previous, {"left": "LEFT"})

    def test_missing_label_reuses_previous_label_for_same_hand(self):
        previous = {"left": "BACK"}

        with mock.patch.object(
            evaluate_ik_methods,
            "_robot_hand_orientation_label",
        ) as fk_label:
            label, source = evaluate_ik_methods._resolve_current_wom_orientation(
                None,
                self.angles,
                "left",
                chains=object(),
                previous_labels=previous,
            )

        self.assertEqual(label, "BACK")
        self.assertEqual(source, "previous_mediapipe")
        fk_label.assert_not_called()

    def test_first_missing_label_uses_fk_with_zero_wrist_yaw(self):
        previous = {}

        with mock.patch.object(
            evaluate_ik_methods,
            "_robot_hand_orientation_label",
            return_value="FRONT",
        ) as fk_label:
            label, source = evaluate_ik_methods._resolve_current_wom_orientation(
                "UNKNOWN",
                self.angles,
                "right",
                chains="chains",
                previous_labels=previous,
            )

        self.assertEqual(label, "FRONT")
        self.assertEqual(source, "fk_zero_wrist_yaw")
        called_angles, called_side, called_chains = fk_label.call_args.args
        self.assertEqual(called_side, "right")
        self.assertEqual(called_chains, "chains")
        self.assertEqual(called_angles["wrist_yaw"], 0.0)
        self.assertEqual(self.angles["wrist_yaw"], 1.2)
        self.assertEqual(previous, {})

    def test_previous_labels_are_independent_per_hand_and_video(self):
        first_video = {"left": "UP"}
        second_video = {}

        with mock.patch.object(
            evaluate_ik_methods,
            "_robot_hand_orientation_label",
            return_value="DOWN",
        ):
            right_label, right_source = evaluate_ik_methods._resolve_current_wom_orientation(
                None,
                self.angles,
                "right",
                chains="chains",
                previous_labels=first_video,
            )
            new_video_label, new_video_source = evaluate_ik_methods._resolve_current_wom_orientation(
                None,
                self.angles,
                "left",
                chains="chains",
                previous_labels=second_video,
            )

        self.assertEqual((right_label, right_source), ("DOWN", "fk_zero_wrist_yaw"))
        self.assertEqual((new_video_label, new_video_source), ("DOWN", "fk_zero_wrist_yaw"))


class VideoIndexFilterTests(unittest.TestCase):
    def test_keeps_indices_at_or_below_limit_across_categories(self):
        files = [
            Path("dataset") / "basketball_movements" / "video_001" / "annotations_filled.json",
            Path("dataset") / "basketball_movements" / "video_006" / "annotations_filled.json",
            Path("dataset") / "police_traffic" / "video_005" / "annotations_filled.json",
            Path("dataset") / "police_traffic" / "video_010" / "annotations_filled.json",
        ]

        filtered = evaluate_ik_methods._filter_annotation_files_by_max_video_index(files, 5)

        self.assertEqual(filtered, [files[0], files[2]])

    def test_none_limit_preserves_all_files(self):
        files = [Path("dataset/category/video_010/annotations_filled.json")]
        self.assertEqual(
            evaluate_ik_methods._filter_annotation_files_by_max_video_index(files, None),
            files,
        )

    def test_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            evaluate_ik_methods._filter_annotation_files_by_max_video_index([], 0)

    def test_rejects_annotations_outside_numbered_video_folder(self):
        files = [Path("dataset/category/custom/annotations_filled.json")]
        with self.assertRaisesRegex(ValueError, "not inside a video_XXX folder"):
            evaluate_ik_methods._filter_annotation_files_by_max_video_index(files, 5)


if __name__ == "__main__":
    unittest.main()
