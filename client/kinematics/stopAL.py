from naoqi import ALProxy
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from pepper_config import PEPPER_IP, PEPPER_PORT

IP = PEPPER_IP
PORT = PEPPER_PORT

try:
    awareness = ALProxy('ALBasicAwareness', IP, PORT)
    autonomouslife = ALProxy("ALAutonomousLife", IP, PORT)
    posture = ALProxy('ALRobotPosture', IP, PORT)
except Exception as e:
    awareness = None
    autonomouslife = None
    posture = None
    print(e)

if autonomouslife:
    autonomouslife.setAutonomousAbilityEnabled("AutonomousBlinking", False)
    autonomouslife.setAutonomousAbilityEnabled("BackgroundMovement", False)
    autonomouslife.setAutonomousAbilityEnabled("BasicAwareness", False)
    autonomouslife.setAutonomousAbilityEnabled("ListeningMovement", False)
    autonomouslife.setAutonomousAbilityEnabled("SpeakingMovement", False)
    # autonomouslife.setState('disabled')

if awareness:
    awareness.setEngagementMode('SemiEngaged') # 'FullyEngaged'
    awareness.setTrackingMode('WholeBody') # 'Head' 'BodyRotation'
    awareness.setStimulusDetectionEnabled('Sound', False)
    awareness.setStimulusDetectionEnabled('Movement', False)
    awareness.setStimulusDetectionEnabled('People', False)
    awareness.setStimulusDetectionEnabled('Touch', False)
    awareness.stopAwareness()

if posture:
    posture.goToPosture("Stand", 0.3)
