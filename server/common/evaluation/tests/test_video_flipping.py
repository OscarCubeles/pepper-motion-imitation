import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "video_flipping.py"
SPEC = importlib.util.spec_from_file_location("video_flipping", MODULE_PATH)
video_flipping = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = video_flipping
SPEC.loader.exec_module(video_flipping)


def _arm(shoulder_x, orientation, pitch, roll, yaw, elbow_roll):
    return {
        "shoulder": [shoulder_x, 2.0, 3.0],
        "elbow": [shoulder_x + 1.0, 4.0, 5.0],
        "wrist": [shoulder_x + 2.0, 6.0, 7.0],
        "joint_angles": {
            "shoulder_pitch": pitch,
            "shoulder_roll": roll,
            "elbow_yaw": yaw,
            "elbow_roll": elbow_roll,
        },
        "wrist_orientation": orientation,
    }


class AnnotationMirroringTests(unittest.TestCase):
    def test_mirrors_coordinates_swaps_arms_and_transforms_angles(self):
        frame = {
            "frame_id": 0,
            "timestamp": 0.0,
            "image_file": "frame_000.png",
            "torso": [10.0, 20.0, 30.0],
            "left_arm": _arm(100.0, "LEFT", 0.1, 0.2, -0.3, -0.4),
            "right_arm": _arm(-200.0, "FRONT", 1.1, -1.2, 1.3, 1.4),
            "metrabs_visualization": "metrabs_visualizations/frame_000_metrabs.jpg",
        }
        original = copy.deepcopy(frame)

        mirrored = video_flipping.mirror_frame_annotation(frame)

        self.assertEqual(frame, original)
        self.assertEqual(mirrored["torso"], [-10.0, 20.0, 30.0])
        self.assertEqual(mirrored["left_arm"]["shoulder"], [200.0, 2.0, 3.0])
        self.assertEqual(mirrored["right_arm"]["shoulder"], [-100.0, 2.0, 3.0])
        self.assertEqual(
            mirrored["left_arm"]["joint_angles"],
            {
                "shoulder_pitch": 1.1,
                "shoulder_roll": 1.2,
                "elbow_yaw": -1.3,
                "elbow_roll": -1.4,
            },
        )
        self.assertEqual(mirrored["left_arm"]["wrist_orientation"], "FRONT")
        self.assertEqual(mirrored["right_arm"]["wrist_orientation"], "RIGHT")
        self.assertNotIn("metrabs_visualization", mirrored)

    def test_preserves_null_values(self):
        self.assertIsNone(video_flipping.mirror_point(None))
        arm = _arm(1.0, None, None, None, None, None)
        mirrored = video_flipping.mirror_arm(arm)
        self.assertTrue(all(value is None for value in mirrored["joint_angles"].values()))

    def test_applies_camera_space_x_offset_after_mirroring(self):
        point = [100.0, 20.0, 1500.0]
        self.assertEqual(
            video_flipping.mirror_point(point, x_offset_mm=60.0),
            [-40.0, 20.0, 1500.0],
        )

        frame = {
            "torso": [10.0, 20.0, 1500.0],
            "left_arm": _arm(100.0, "LEFT", 0.1, 0.2, 0.3, 0.4),
            "right_arm": _arm(-200.0, "RIGHT", 0.5, -0.6, -0.7, 0.8),
        }
        mirrored = video_flipping.mirror_frame_annotation(frame, x_offset_mm=60.0)
        self.assertEqual(mirrored["torso"][0], 50.0)
        self.assertEqual(mirrored["left_arm"]["shoulder"][0], 260.0)
        self.assertEqual(mirrored["right_arm"]["shoulder"][0], -40.0)


class EndToEndTests(unittest.TestCase):
    def test_creates_next_video_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            category = root / "police_traffic"
            source = category / "video_006"
            frames = source / "frames"
            frames.mkdir(parents=True)
            (category / "video_009").mkdir()

            frame_names = []
            source_images = []
            for index in range(2):
                image = np.zeros((48, 64, 3), dtype=np.uint8)
                image[:, :20] = (10 + index, 40, 200)
                image[:, 20:] = (220, 30 + index, 5)
                frame_name = f"frame_{index:03d}.png"
                cv2.imwrite(str(frames / frame_name), image)
                frame_names.append(frame_name)
                source_images.append(image)

            writer = cv2.VideoWriter(
                str(source / "video.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                8.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for image in source_images:
                writer.write(image)
            writer.release()

            annotation = {
                "video_id": "video_006",
                "category": "police_traffic",
                "fps": 8,
                "duration_seconds": 0.25,
                "frames": [
                    {
                        "frame_id": index,
                        "timestamp": index / 8.0,
                        "image_file": frame_name,
                        "torso": [1.0, 2.0, 3.0],
                        "left_arm": _arm(10.0, "UP", 0.1, 0.2, 0.3, -0.4),
                        "right_arm": _arm(-10.0, "DOWN", 0.5, -0.6, -0.7, 0.8),
                    }
                    for index, frame_name in enumerate(frame_names)
                ],
            }
            for name in video_flipping.REQUIRED_ANNOTATIONS:
                (source / name).write_text(json.dumps(annotation), encoding="utf-8")

            source_before = {
                path.relative_to(source): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
            }

            destination = video_flipping.augment_video(source)

            self.assertEqual(destination.name, "video_010")
            self.assertTrue((destination / "video.mp4").is_file())
            self.assertFalse((destination / "annotations_filled.before_angles.json").exists())
            for index, frame_name in enumerate(frame_names):
                actual = cv2.imread(str(destination / "frames" / frame_name))
                expected = cv2.flip(source_images[index], 1)
                np.testing.assert_array_equal(actual, expected)

            output_annotations = json.loads(
                (destination / "annotations_filled.json").read_text(encoding="utf-8")
            )
            self.assertEqual(output_annotations["video_id"], "video_010")
            self.assertEqual(output_annotations["frames"][0]["left_arm"]["shoulder"][0], 10.0)

            source_after = {
                path.relative_to(source): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
            }
            self.assertEqual(source_after, source_before)

            capture = cv2.VideoCapture(str(destination / "video.mp4"))
            decoded_count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded_count += 1
            capture.release()
            self.assertEqual(decoded_count, 2)


if __name__ == "__main__":
    unittest.main()
