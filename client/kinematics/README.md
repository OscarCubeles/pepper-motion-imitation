# Kinematics Processing and Robot Control

## Overview

This folder contains the core kinematics processing system that converts 3D human pose data into Pepper robot motion commands. It implements inverse kinematics, forward kinematics, workspace scaling, and direct robot control via the NAOqi SDK.

## Core Modules

### Robot Communication & Control

- **`client_dl.py`**: Main client script that receives 3D pose data from the server and controls Pepper
- **`pepper_commands.py`**: Pepper robot command wrapper functions
- **`SayHi.py`**: Demo script for Pepper "Say Hi" gesture
- **`start_AL.py`**: Script to start Pepper's autonomous life
- **`stopAL.py`**: Script to stop Pepper's autonomous life

### Kinematics Processing

- **`inverse_kinematics.py`**: Converts 3D poses to joint angles
- **`inverse_kinematics_rangeupdate.py`**: Updated version with joint range constraints
- **`forward_kinematics.py`**: Computes end-effector positions from joint angles
- **`transformation_matrices.py`**: Homogeneous transformation matrices for kinematic chains
- **`hip_knee_pairs.py`**: Hip and knee joint pairing logic

### Scaling & Coordinate Transformation

- **`scaling_spherical.py`**: Spherical coordinate-based scaling to Pepper's workspace
- **`scaling_spherical_mirroring.py`**: Scaling with mirroring support
- **`math_functions.py`**: Mathematical utility functions

### Configuration

- **`../pepper_config.py`**: Central Pepper endpoint configuration

### Testing & Utilities

- **`test_file.py`**: Testing utilities
- **`testing_kinematics.py`**: Kinematics testing scripts

## Key Features

### 1. Inverse Kinematics
Converts 3D keypoint positions (from pose estimation) into joint angles that Pepper can execute. Handles:
- Shoulder and elbow joints for both arms
- Hand open/close commands from wrist-to-fingertip distance
- Hip, knee, and ankle joints for both legs
- Joint angle constraints and limits

### 2. Forward Kinematics
Computes the position of end-effectors given joint angles. Used for:
- Validation of inverse kinematics solutions
- Workspace analysis
- Motion planning

### 3. Workspace Scaling
Scales human pose data to fit Pepper's physical constraints:
- Different body proportions (human vs robot)
- Workspace limitations
- Joint range limits

### 4. Coordinate Transformation
Aligns the camera coordinate system with Pepper's reference frame:
- Origin alignment
- Axis orientation
- Mirror mode for natural imitation

## Configuration

### Robot IP Address

Edit `client/pepper_config.py` and set:

```python
PEPPER_MODE = "real"  # or "sim"
```

## Usage

### Running the Main Client

```bash
conda activate pepper
python client_dl.py
```

This script:
1. Connects to the pose estimation server via WebSocket (default: `ws://127.0.0.1:8765`)
2. Receives 3D pose keypoints
3. Scales and transforms the pose data
4. Computes inverse kinematics to get joint angles
5. Sends commands to Pepper robot

### Testing Individual Modules

```bash
# Test inverse kinematics
python testing_kinematics.py

# Test Pepper connection
python start_AL.py
```

### Demo Gestures

```bash
# Make Pepper say hi
python SayHi.py
```

## Workflow

```
3D Pose Data (from server)
    ↓
Coordinate Transformation
    ↓
Workspace Scaling
    ↓
Inverse Kinematics
    ↓
Joint Angle Commands
    ↓
Pepper Robot (via NAOqi SDK)
```

## Dependencies

- Python 2.7 (32-bit)
- NAOqi SDK 2.5.5
- numpy
- websocket-client
- keyboard

## Notes

- All `.pyc` files are compiled Python bytecode (can be ignored for version control)
- Ensure Pepper's autonomous life is started before running motion commands
- The robot should be in a safe position with adequate clearance before starting
- Monitor the robot during operation for safety

## Related Folders

- **Parent folder (`client/`)**: Contains additional documentation and media
- **Server implementations (`server/`)**: Pose estimation systems
- **Demo scripts (`Pepperdemo/`)**: Ready-to-use batch files for demos

## Contributors

- **Darja**: Original kinematics implementation and Pepper robot integration
