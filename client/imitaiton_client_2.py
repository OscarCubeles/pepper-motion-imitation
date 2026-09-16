# -*- coding: utf-8 -*-
from __future__ import print_function
import pepper_config as settings
settings.set_client_dir()
import pose_stream_runtime2 as pose_stream_runtime

try:
    from naoqi import ALProxy
except ImportError:
    ALProxy = None


class HandOrientationAnnouncer(object):
    """Reports hand orientation every N frames using robot TTS"""
    
    def __init__(self, announce_interval=6, pepper_ip=None, pepper_port=None):
        self.announce_interval = announce_interval
        self.frame_count = 0
        self.tts_proxy = None
        self.connected = False
        
        if ALProxy is not None:
            try:
                ip = pepper_ip or settings.PEPPER_IP
                port = pepper_port or settings.PEPPER_PORT
                #print("[ANNOUNCER] Connecting to Pepper at {0}:{1}".format(ip, port))
                self.tts_proxy = ALProxy("ALTextToSpeech", ip, port)
                self.connected = True
                #print("[ANNOUNCER] Successfully connected to Pepper TTS")
            except Exception as e:
                print("")
                #print("[ANNOUNCER] ERROR: Could not connect to Pepper TTS: " + str(e))
                #print("[ANNOUNCER] Make sure Pepper is running and reachable")
        else:
            print("")
            #print("[ANNOUNCER] Warning: naoqi module not available")
    
    def __call__(self, joint_data):
        """Called as joint_observer for each frame"""
        self.frame_count += 1
        
        # Every N frames, announce hand orientation
        if self.frame_count % self.announce_interval == 0:
            hand_orientation = joint_data.get("hand_orientation", {})
            if hand_orientation:
                #print("[ANNOUNCER] Frame {0}: hand_orientation={1}".format(self.frame_count, hand_orientation))
                if self.connected and self.tts_proxy is not None:
                    self._announce_orientation(hand_orientation)
                elif not self.connected:
                    #print("[ANNOUNCER] Error: Not connected to Pepper TTS")ç
                    print("")
            else:
                #print("[ANNOUNCER] Frame {0}: No hand orientation data".format(self.frame_count))
                print("")
    
    def _announce_orientation(self, hand_orientation):
        """Say hand orientations using robot TTS"""
        try:
            announcement = self._build_announcement(hand_orientation)
            if announcement:
                #print("[ANNOUNCE] " + announcement)
                # Convert to native string for naoqi compatibility in Python 2.7
                announcement_str = str(announcement)
                #print("[ANNOUNCER] Sending to Pepper: " + announcement_str)
                self.tts_proxy.say(announcement_str)
                #print("[ANNOUNCER] Message sent successfully")
            #else:
                #print("[ANNOUNCER] No announcement text built")
            #    print("")
        except Exception as e:
            
            #print("[ANNOUNCER] Error announcing orientation: " + str(e))
            import traceback
            traceback.print_exc()
    
    def _build_announcement(self, hand_orientation):
        """Build announcement text from orientation labels"""
        parts = []
        
        if "Right" in hand_orientation:
            right = hand_orientation["Right"]
            right_primary = str(right.get("primary", "")).lower()
            if right_primary and right_primary != "unknown":
                parts.append("Right " + right_primary)
        
        if "Left" in hand_orientation:
            left = hand_orientation["Left"]
            left_primary = str(left.get("primary", "")).lower()
            if left_primary and left_primary != "unknown":
                parts.append("Left " + left_primary)
        
        if parts:
            return str(", ".join(parts))
        return None


def run_imitation_client(
    server_host=settings.SERVER_HOST,
    port=settings.PORT,
    mirroring_imitation=settings.MIRRORING_IMITATION,
    request_message=settings.REQUEST_MESSAGE,
    stop_key="q",
    run_duration_sec=None,
    joint_observer=None,
    backend_profile=settings.POSE_BACKEND_PROFILE,
    announce_hand_orientation=True,
    announce_interval=6,
    pepper_ip=None,
    pepper_port=None
):
    # Imitation entry point stays thin; websocket/streaming runtime is shared.
    
    # Add hand orientation announcer if requested
    if announce_hand_orientation:
        announcer = HandOrientationAnnouncer(
            announce_interval=announce_interval,
            pepper_ip=pepper_ip,
            pepper_port=pepper_port
        )
        # Chain with existing observer if provided
        if joint_observer is not None:
            original_observer = joint_observer
            def chained_observer(data):
                announcer(data)
                original_observer(data)
            joint_observer = chained_observer
        else:
            joint_observer = announcer
    
    return pose_stream_runtime.run_pose_stream_client(
        server_host=server_host,
        port=port,
        mirroring_imitation=mirroring_imitation,
        request_message=request_message,
        stop_key=stop_key,
        run_duration_sec=run_duration_sec,
        joint_observer=joint_observer,
        backend_profile=backend_profile,
    )


def main():
    return run_imitation_client(
        server_host=settings.SERVER_HOST, 
        port=settings.PORT, 
        backend_profile=settings.POSE_BACKEND_PROFILE,
        announce_hand_orientation=True,
        announce_interval=6
    )


if __name__ == "__main__":
    main()
