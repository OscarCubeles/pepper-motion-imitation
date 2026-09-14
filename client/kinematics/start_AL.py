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
except Exception as e:
    awareness = None
    autonomouslife = None
    print(e)

if autonomouslife:
    autonomouslife.setAutonomousAbilityEnabled("AutonomousBlinking", True)
    autonomouslife.setAutonomousAbilityEnabled("BackgroundMovement", True)
    autonomouslife.setAutonomousAbilityEnabled("BasicAwareness", True)
    autonomouslife.setAutonomousAbilityEnabled("ListeningMovement", True)
    autonomouslife.setAutonomousAbilityEnabled("SpeakingMovement", True)

if awareness:
    awareness.setEngagementMode('FullyEngaged') # 'FullyEngaged'
    awareness.setTrackingMode('WholeBody') # 'Head' 'BodyRotation'
    awareness.setStimulusDetectionEnabled('Sound', True)
    awareness.setStimulusDetectionEnabled('Movement', True)
    awareness.setStimulusDetectionEnabled('People', True)
    awareness.setStimulusDetectionEnabled('Touch', True)
    awareness.startAwareness()

    
