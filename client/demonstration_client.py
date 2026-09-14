# -*- coding: utf-8 -*-

import os
import sys
import time
import wave
import audioop
from threading import Event, Thread

try:
    from cStringIO import StringIO as BytesIO
except ImportError:  # pragma: no cover
    from io import BytesIO

CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
KINEMATICS_DIR = os.path.join(CLIENT_DIR, "kinematics_new")
EXERCISES_DIR = os.path.join(CLIENT_DIR, "exercises")

# Keep flat imports consistent with the existing client scripts.
for path in (CLIENT_DIR, KINEMATICS_DIR, EXERCISES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import pose_stream_runtime
import arm_raise
import compare_motion_dtw
from pepper_config import (
    PEPPER_IP,
    PEPPER_PORT,
    FEEDBACK_SERVER_URL,
    FEEDBACK_SERVER_TIMEOUT_SEC,
)
from feedback_client import (
    FeedbackAudioClient,
    FeedbackClientError,
    clamp_round_score,
    score_to_feedback_category,
)
from torso_imitation_logger import (
    start_arm_logging,
    stop_arm_logging,
    get_arm_logging_paths,
    update_arm_keypoints,
)

POSE_SERVER_HOST = "localhost"
POSE_SERVER_PORT = 8080
DEFAULT_EXERCISE_NAME = "arms_rise"
PROJECT_ROOT_DIR = os.path.dirname(CLIENT_DIR)
PRERECORDED_AUDIO_DIR = os.path.join(PROJECT_ROOT_DIR, "ai_feedback_server", "audio")
GREETING_AUDIO_FILE = u"Hallo.wav"
FOLLOW_AUDIO_FILE = u"Folgen.wav"
END_AUDIO_FILE = u"Spaß.wav"
AUDIO_STREAM_TARGET_SAMPLE_RATE = 48000
AUDIO_STREAM_TARGET_VOLUME = 50
AUDIO_STREAM_MAX_CHUNK_BYTES = 16384
AUDIO_STREAM_DRAIN_GUARD_SEC = 0.25
POSE_STREAM_FIRST_FRAME_TIMEOUT_SEC = 12.0
SHOW_DEMONSTRATION_LOG_PLOTS = False


class ArmMotionLoggerAdapter(object):
    """Owns arm-motion logging for demonstration mode."""

    def __init__(self):
        self.keypoints_csv_path = None
        self.angles_csv_path = None

    def start(self):
        start_arm_logging(show_plots=SHOW_DEMONSTRATION_LOG_PLOTS)
        self.keypoints_csv_path, self.angles_csv_path = get_arm_logging_paths()
        print("Arm keypoint CSV logger started: %s" % str(self.keypoints_csv_path))
        print("Arm angle CSV logger started: %s" % str(self.angles_csv_path))
        return self.keypoints_csv_path, self.angles_csv_path

    def stop(self):
        stop_arm_logging()
        return self.keypoints_csv_path, self.angles_csv_path

    def on_joint_frame(self, frame):
        # Reuse the existing arm logger module without coupling to the runtime.
        update_arm_keypoints(
            frame.get("torso"),
            frame.get("neck"),
            frame.get("left_shoulder"),
            frame.get("left_elbow"),
            frame.get("left_wrist"),
            frame.get("right_shoulder"),
            frame.get("right_elbow"),
            frame.get("right_wrist"),
        )


class PoseCaptureRunner(object):
    """Runs the pose-stream connection in a background thread for logging-only capture."""

    def __init__(
        self,
        server_host,
        port,
        duration_sec,
        joint_observer,
        mirroring_imitation=False,
        run_duration_from_first_payload=False,
        first_payload_timeout_sec=None,
    ):
        self.server_host = str(server_host) if server_host else "localhost"
        self.port = int(port)
        self.duration_sec = float(duration_sec)
        self.joint_observer = joint_observer
        self.mirroring_imitation = bool(mirroring_imitation)
        self.run_duration_from_first_payload = bool(run_duration_from_first_payload)
        self.first_payload_timeout_sec = (
            None if first_payload_timeout_sec is None else float(first_payload_timeout_sec)
        )
        self.result = None
        self.error = None
        self._thread = None
        self._first_frame_event = Event()

    def start(self):
        self._thread = Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def join(self):
        if self._thread is not None:
            self._thread.join()
        if self.error is not None:
            raise self.error
        return self.result or {}

    def wait_for_first_frame(self, timeout_sec):
        timeout = None if timeout_sec is None else float(timeout_sec)
        return self._first_frame_event.wait(timeout=timeout)

    def _on_joint_frame(self, frame):
        self._first_frame_event.set()
        if self.joint_observer is not None:
            self.joint_observer(frame)

    def _run(self):
        try:
            self.result = pose_stream_runtime.run_pose_stream_client(
                server_host=self.server_host,
                port=self.port,
                mirroring_imitation=self.mirroring_imitation,
                run_duration_sec=self.duration_sec,
                run_duration_from_first_payload=self.run_duration_from_first_payload,
                first_payload_timeout_sec=self.first_payload_timeout_sec,
                joint_observer=self._on_joint_frame,
                enable_robot_imitation=False,
                manage_pepper_security=False,
                send_to_stand_on_exit=False,
                mode_label="Demonstration capture",
            )
        except Exception as exc:
            self.error = exc


class DemonstrationClient(object):
    def __init__(
        self,
        server_host="localhost",
        port=8080,
        arm_raise_duration_sec=5.0,
        pose_stream_startup_delay_sec=0.4,
        pose_stream_tail_sec=0.5,
        pose_stream_first_frame_timeout_sec=POSE_STREAM_FIRST_FRAME_TIMEOUT_SEC,
        mirroring_imitation=False,
        feedback_server_url=FEEDBACK_SERVER_URL,
        exercise_name=DEFAULT_EXERCISE_NAME,
    ):
        self.server_host = str(server_host) if server_host else "localhost"
        self.port = int(port)
        self.arm_raise_duration_sec = float(arm_raise_duration_sec)
        self.pose_stream_startup_delay_sec = float(pose_stream_startup_delay_sec)
        self.pose_stream_tail_sec = float(pose_stream_tail_sec)
        self.pose_stream_first_frame_timeout_sec = (
            None if pose_stream_first_frame_timeout_sec is None else float(pose_stream_first_frame_timeout_sec)
        )
        self.mirroring_imitation = bool(mirroring_imitation)
        self.exercise_name = str(exercise_name) if exercise_name else DEFAULT_EXERCISE_NAME
        self.feedback_client = FeedbackAudioClient(
            base_url=(feedback_server_url or FEEDBACK_SERVER_URL),
            timeout_sec=FEEDBACK_SERVER_TIMEOUT_SEC,
            max_retries=1,
        )
        self.arm_logger = ArmMotionLoggerAdapter()
        self._audio_output_flushed_once = False

    def run(self):
        motion_proxy, posture_proxy, audio_device = self._create_pepper_proxies()
        try:
            template_angles_path, motion_keypoints_path, motion_angles_path, session_info = (
                self._run_live_demonstration_and_capture(motion_proxy, posture_proxy, audio_device)
            )
        except RuntimeError as exc:
            print("")
            print("Demonstration aborted: %s" % str(exc))
            return {
                "error": "capture_setup_failed",
                "message": str(exc),
            }
        if not self._has_usable_motion_angles(motion_angles_path):
            reason = str(session_info.get("stop_reason"))
            self._handle_missing_motion_capture(reason)
            return {
                "error": "no_motion_data",
                "stop_reason": reason,
                "template_angles_path": template_angles_path,
                "motion_angles_path": motion_angles_path,
                "motion_keypoints_path": motion_keypoints_path,
            }

        try:
            score_info = self._score_arm_raise(template_angles_path, motion_angles_path)
        except ValueError as exc:
            self._handle_scoring_error(exc, session_info)
            return {
                "error": "dtw_failed",
                "stop_reason": str(session_info.get("stop_reason")),
                "message": str(exc),
            }

        print("")
        print("Demonstration summary")
        print("---------------------")
        print("Template angles: %s" % template_angles_path)
        print("Human keypoints: %s" % motion_keypoints_path)
        print("Human angles:    %s" % motion_angles_path)
        print("Stop reason:     %s" % str(session_info.get("stop_reason")))
        print("DTW total cost:  %.6f" % score_info["total_cost"])
        print("DTW avg cost:    %.6f" % score_info["avg_cost"])
        print("DTW path length: %d" % score_info["path_len"])
        print("Score (0-100):   %.2f" % score_info["score"])

        self._send_feedback_audio_and_play(
            score=score_info["score"],
            audio_device=audio_device,
        )
        time.sleep(1.0)
        self._play_local_audio_file(audio_device, END_AUDIO_FILE, "end cue")
        return score_info

    def _run_live_demonstration_and_capture(self, motion_proxy, posture_proxy, audio_device):
        print("Starting live demonstration + capture (%.1f s template)..." % self.arm_raise_duration_sec)

        motion_proxy.wakeUp()
        posture_proxy.goToPosture("StandInit", 0.5)
        time.sleep(0.8)

        self._play_local_audio_file(audio_device, GREETING_AUDIO_FILE, "greeting")
        time.sleep(1.0)
        self._play_local_audio_file(audio_device, FOLLOW_AUDIO_FILE, "follow cue")
        time.sleep(0.2)

        capture_duration = self.arm_raise_duration_sec + max(0.0, self.pose_stream_tail_sec)
        capture_runner = PoseCaptureRunner(
            server_host=self.server_host,
            port=self.port,
            duration_sec=capture_duration,
            joint_observer=self.arm_logger.on_joint_frame,
            mirroring_imitation=self.mirroring_imitation,
            run_duration_from_first_payload=True,
            first_payload_timeout_sec=self.pose_stream_first_frame_timeout_sec,
        )
        capture_runner.start()

        # Give the websocket capture thread a moment to connect before motion starts.
        if self.pose_stream_startup_delay_sec > 0.0:
            time.sleep(self.pose_stream_startup_delay_sec)
        if self.pose_stream_first_frame_timeout_sec is not None and self.pose_stream_first_frame_timeout_sec > 0.0:
            print("Waiting for first pose payload (timeout %.1f s)..." % self.pose_stream_first_frame_timeout_sec)
            if not capture_runner.wait_for_first_frame(self.pose_stream_first_frame_timeout_sec):
                session_info = capture_runner.join()
                raise RuntimeError(
                    "No pose payload received before timeout (stop reason: %s). "
                    "Warm up the pose server and ensure a person is visible to the camera." % (
                        str(session_info.get("stop_reason")),
                    )
                )

        template_logging_prev = arm_raise.ENABLE_TEMPLATE_LOGGER
        template_show_plot_prev = arm_raise.TEMPLATE_LOGGER_SHOW_PLOT
        arm_raise.ENABLE_TEMPLATE_LOGGER = True
        arm_raise.TEMPLATE_LOGGER_SHOW_PLOT = SHOW_DEMONSTRATION_LOG_PLOTS
        session_info = None
        try:
            # Human logger starts exactly for the template execution window.
            self.arm_logger.start()
            arm_raise.run_template(
                motion_proxy,
                self.exercise_name,
                arm_raise.move_arms_rise,
                duration_sec=self.arm_raise_duration_sec,
                settle_sec=0.0,
            )
        finally:
            self.arm_logger.stop()
            arm_raise.ENABLE_TEMPLATE_LOGGER = template_logging_prev
            arm_raise.TEMPLATE_LOGGER_SHOW_PLOT = template_show_plot_prev

        session_info = capture_runner.join()

        template_angles_path = arm_raise.get_logging_path()
        motion_keypoints_path = self.arm_logger.keypoints_csv_path
        motion_angles_path = self.arm_logger.angles_csv_path

        if not template_angles_path:
            raise RuntimeError("Arm raise template logger did not produce a CSV path.")
        if not motion_keypoints_path or not motion_angles_path:
            raise RuntimeError("Arm imitation logger did not produce CSV files.")

        print("Arm raise template recorded: %s" % str(template_angles_path))
        posture_proxy.goToPosture("StandInit", 0.5)
        time.sleep(0.4)
        return template_angles_path, motion_keypoints_path, motion_angles_path, (session_info or {})

    def _score_arm_raise(self, template_angles_path, motion_angles_path):
        # Initial demonstration mode compares shoulder angles only because the
        # Pepper template and human arm logger use different elbow columns.
        return compare_motion_dtw.compare_angle_csvs(
            template_angles_path=template_angles_path,
            motion_angles_path=motion_angles_path,
            template_angle_cols=[
                "LShoulderPitch",
                "LShoulderRoll",
                "RShoulderPitch",
                "RShoulderRoll",
            ],
            motion_angle_cols=[
                "LShoulderPitch_rad",
                "LShoulderRoll_rad",
                "RShoulderPitch_rad",
                "RShoulderRoll_rad",
            ],
        )

    def _send_feedback_audio_and_play(self, score, audio_device):
        score_int = clamp_round_score(score)
        category = score_to_feedback_category(score_int)
        print(
            "Feedback request payload: score=%d category=%s exercise_name=%s" %
            (score_int, category, self.exercise_name)
        )
        try:
            wav_bytes = self.feedback_client.generate_feedback_audio(
                score=score_int,
                category=category,
                exercise_name=self.exercise_name,
            )
            self._play_wav_bytes(audio_device, wav_bytes)
            print("Pepper feedback audio played from remote server.")
        except FeedbackClientError as exc:
            print("Feedback server request failed: %s" % str(exc))
        except Exception as exc:
            print("Feedback audio playback failed: %s" % str(exc))

    def _play_local_audio_file(self, audio_device, file_name, label):
        wav_path = os.path.join(PRERECORDED_AUDIO_DIR, file_name)
        if not os.path.exists(wav_path):
            raise RuntimeError("Missing prerecorded audio file: %s" % wav_path)
        with open(wav_path, "rb") as handle:
            wav_bytes = handle.read()
        if not wav_bytes.startswith("RIFF"):
            raise RuntimeError("Prerecorded file is not a WAV/RIFF payload: %s" % wav_path)
        self._play_wav_bytes(audio_device, wav_bytes)
        print("Played prerecorded audio (%s): %s" % (label, wav_path))

    def _play_wav_bytes(self, audio_device, wav_bytes):
        if not wav_bytes:
            raise RuntimeError("Feedback server returned empty audio payload.")
        pcm_bytes, sample_rate = self._wav_to_stereo_pcm_16(wav_bytes, AUDIO_STREAM_TARGET_SAMPLE_RATE)
        try:
            self._set_output_volume(audio_device, AUDIO_STREAM_TARGET_VOLUME)
        except Exception:
            pass
        if not self._audio_output_flushed_once:
            try:
                audio_device.flushAudioOutputs()
            except Exception:
                pass
            self._audio_output_flushed_once = True
        mode, sent_frames = self._stream_pcm_to_output(audio_device, pcm_bytes, sample_rate)
        duration = float(sent_frames) / float(sample_rate)
        if AUDIO_STREAM_DRAIN_GUARD_SEC > 0.0:
            time.sleep(AUDIO_STREAM_DRAIN_GUARD_SEC)
        print(
            "Feedback audio streamed via ALAudioDevice (mode=%s, duration=%.2fs, guard=%.2fs)." % (
                mode,
                duration,
                AUDIO_STREAM_DRAIN_GUARD_SEC,
            )
        )

    def _wav_to_stereo_pcm_16(self, wav_bytes, target_rate):
        # Convert to stereo 16-bit PCM at target sample rate for sendRemoteBufferToOutput.
        source = BytesIO(wav_bytes)
        try:
            reader = wave.open(source, "rb")
        except Exception:
            raise RuntimeError("Feedback payload is not a readable WAV stream.")

        try:
            nchannels = reader.getnchannels()
            sampwidth = reader.getsampwidth()
            framerate = reader.getframerate()
            comptype = reader.getcomptype()
            if comptype != "NONE":
                raise RuntimeError("Compressed WAV feedback is not supported.")
            frames = reader.readframes(reader.getnframes())
        finally:
            try:
                reader.close()
            except Exception:
                pass

        # Convert sample width to 16-bit PCM.
        if sampwidth != 2:
            frames = audioop.lin2lin(frames, sampwidth, 2)
            sampwidth = 2

        # Convert to stereo interleaved frames.
        if nchannels == 1:
            frames = audioop.tostereo(frames, sampwidth, 1.0, 1.0)
            nchannels = 2
        elif nchannels != 2:
            mono = audioop.tomono(frames, sampwidth, 0.5, 0.5)
            frames = audioop.tostereo(mono, sampwidth, 1.0, 1.0)
            nchannels = 2

        # Resample for stable robot playback speed.
        if framerate != int(target_rate):
            frames, _state = audioop.ratecv(frames, sampwidth, nchannels, framerate, int(target_rate), None)
            framerate = int(target_rate)

        if nchannels != 2 or sampwidth != 2:
            raise RuntimeError("Failed to normalize feedback audio for robot output.")
        return frames, framerate

    def _stream_pcm_to_output(self, audio_device, pcm_stereo_16, sample_rate):
        frame_bytes = 4  # 2 channels * 16-bit
        chunk_size = (AUDIO_STREAM_MAX_CHUNK_BYTES // frame_bytes) * frame_bytes
        if chunk_size <= 0:
            raise RuntimeError("Invalid audio stream chunk size.")

        send_mode = None
        sent_frames = 0
        start_wall = time.time()

        for chunk in self._chunk_bytes(pcm_stereo_16, chunk_size):
            if len(chunk) % frame_bytes != 0:
                pad = "\x00" * (frame_bytes - (len(chunk) % frame_bytes))
                chunk = chunk + pad
            nb_frames = len(chunk) // frame_bytes

            if send_mode is None:
                try:
                    self._send_remote_chunk(audio_device, nb_frames, chunk, "bytes")
                    send_mode = "bytes"
                except Exception:
                    self._send_remote_chunk(audio_device, nb_frames, chunk, "list")
                    send_mode = "list"
            else:
                self._send_remote_chunk(audio_device, nb_frames, chunk, send_mode)

            sent_frames += nb_frames

            expected_elapsed = float(sent_frames) / float(sample_rate)
            elapsed = time.time() - start_wall
            if expected_elapsed > elapsed:
                time.sleep(expected_elapsed - elapsed)

        return send_mode, sent_frames

    def _send_remote_chunk(self, audio_device, nb_frames, chunk, mode):
        if mode == "bytes":
            return audio_device.sendRemoteBufferToOutput(nb_frames, chunk)
        if mode == "list":
            return audio_device.sendRemoteBufferToOutput(nb_frames, self._to_alvalue_list(chunk))
        raise RuntimeError("Unsupported audio send mode: %s" % str(mode))

    def _chunk_bytes(self, data, chunk_size):
        index = 0
        total = len(data)
        while index < total:
            yield data[index:index + chunk_size]
            index += chunk_size

    def _to_alvalue_list(self, chunk):
        return [ord(b) for b in chunk]

    def _set_output_volume(self, audio_device, target_volume):
        target = int(target_volume)
        if target < 0:
            target = 0
        if target > 100:
            target = 100
        try:
            audio_device.setOutputVolume(target)
        except Exception:
            pass

    def _has_usable_motion_angles(self, motion_angles_path):
        try:
            header, data = compare_motion_dtw.load_csv(motion_angles_path)
            rows = compare_motion_dtw.extract_columns(
                header,
                data,
                [
                    "LShoulderPitch_rad",
                    "LShoulderRoll_rad",
                    "RShoulderPitch_rad",
                    "RShoulderRoll_rad",
                ],
            )
            return len(rows) > 0
        except Exception:
            return False

    def _handle_missing_motion_capture(self, stop_reason):
        print("")
        print("No usable human motion data was captured; skipping DTW.")
        print("Pose server connection: ws://%s:%s/PepperCommands" % (self.server_host, self.port))
        print("Capture stop reason: %s" % str(stop_reason))

    def _handle_scoring_error(self, exc, session_info):
        print("")
        print("DTW scoring failed: %s" % str(exc))
        print("Capture stop reason: %s" % str(session_info.get("stop_reason")))

    def _create_pepper_proxies(self):
        try:
            motion_proxy = arm_raise.ALProxy("ALMotion", PEPPER_IP, PEPPER_PORT)
            posture_proxy = arm_raise.ALProxy("ALRobotPosture", PEPPER_IP, PEPPER_PORT)
            audio_device = arm_raise.ALProxy("ALAudioDevice", PEPPER_IP, PEPPER_PORT)
            return motion_proxy, posture_proxy, audio_device
        except Exception as exc:
            raise RuntimeError(
                "Failed to connect to Pepper proxies (%s:%s): %s" % (
                    PEPPER_IP,
                    PEPPER_PORT,
                    exc,
                )
            )


def main():
    client = DemonstrationClient(server_host=POSE_SERVER_HOST, port=POSE_SERVER_PORT)
    client.run()


if __name__ == "__main__":
    main()