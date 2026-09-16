import numpy as np


class HumanArmClassifier:
    """
    Classifier for human arm reachability based on shoulder pitch and elbow yaw joint constraints.
    
    This classifier works for both LEFT and RIGHT arms and only applies when
    there is a singularity condition detected.
    """
    
    def __init__(self):
        """
        Initialize the classifier.
        
        The arm is specified per classification call in the classify() method.
        """
    
    @staticmethod
    def _radians_to_degrees(radians):
        """Convert radians to degrees."""
        return np.degrees(radians)
    
    @staticmethod
    def _interp(x, x0, x1, y0, y1):
        """Linear interpolation between two points."""
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    
    def _in_interval(self, x, y, x_points, y_min, y_max):
        """Check whether y lies inside the interpolated valid range at x."""
        if x < x_points[0] or x > x_points[-1]:
            return False
        
        for i in range(len(x_points) - 1):
            if x_points[i] <= x <= x_points[i + 1]:
                ymin = self._interp(x, x_points[i], x_points[i+1], y_min[i], y_min[i+1])
                ymax = self._interp(x, x_points[i], x_points[i+1], y_max[i], y_max[i+1])
                
                return ymin <= y <= ymax
        
        return False
    
    def ey_valid_from_sp(self, sp):
        """Get valid elbow yaw (EY) range given shoulder pitch (SP)."""
        sp_points = [-119.5, -60, 0, 60, 119.5]
        ey_min = [10, -50, -119, -119, -119]
        ey_max = [105, 119, 90, 40, -60]
        
        return sp_points, ey_min, ey_max
    
    def sp_valid_from_ey(self, ey):
        """Get valid shoulder pitch (SP) range given elbow yaw (EY)."""
        ey_points = [-119.5, -60, 0, 60, 119.5]
        sp_min = [25, -32.5, -90, -119.5, -119.5]
        sp_max = [119.5, 119.5, 75, 15, -47.5]
        
        return ey_points, sp_min, sp_max
    
    def ey_valid_from_sp_right(self, sp):
        """Get valid elbow yaw (EY) range given shoulder pitch (SP) for RIGHT arm."""
        sp_points = [-119.5, -60, 0, 60, 119.5]
        ey_min = [-119.5, -119.5, -75, -5, 35]
        ey_max = [-5, 50, 100, 119.5, 119.5]
        
        return sp_points, ey_min, ey_max
    
    def sp_valid_from_ey_right(self, ey):
        """Get valid shoulder pitch (SP) range given elbow yaw (EY) for RIGHT arm."""
        ey_points = [-119.5, -60, 0, 60, 119.5]
        sp_min = [-119.5, -119.5, -85, -30, 30]
        sp_max = [-30, 5, 60, 119.5, 119.5]
        
        return ey_points, sp_min, sp_max
    
    def classify(self, sp, ey, arm: str = "left", singularity: dict = None) -> str | None:
        """
        Checks if pepper pose is doable for a human
        
        This method only produces results when a singularity condition is detected.
        
        Args:
            sp: Shoulder pitch angle (degrees)
            ey: Elbow yaw angle (degrees)
            arm: "left" or "right" - which arm's model to use
            singularity: Singularity detection object with structure:
                {
                    "has_singularity": bool,
                    "singularity_type": None or "elbow_roll" or "shoulder_roll" or "dual",
                    "confidence": None or "singular" or "confident_singular",
                    "warnings": [list of warning strings],
                    "details": {...}
                }
            
        Returns:
            "NON_HUMAN_DOABLE": Position is not reachable by a human
            "HUMAN_DOABLE": Position is reachable by a human
            None: Classification not applicable (no singularity)
        """

        # Validate arm parameter
        arm = arm.lower()
        if arm not in ("left", "right"):
            return None
        
        # Only apply when there's a shoulder_roll singularity
        if singularity is None:
            return None
        
        singularity_type = singularity.get("singularity_type")

        # Check for shoulder_roll singularity (can be standalone or part of dual)
        # Handle both formats: "shoulder_roll" or "shoulder_roll (singular)", "dual" or "dual (singular)"
        if singularity_type is None:
            return None
        
        has_shoulder_roll = "shoulder_roll" in str(singularity_type)
        has_dual = "dual" in str(singularity_type)
        
        if not (has_shoulder_roll or has_dual):
            return None
        
        # Validate that we have valid angle values
        if sp is None or ey is None:
            return None
        
        # Convert radians to degrees if needed
        sp = self._radians_to_degrees(sp)
        ey = self._radians_to_degrees(ey)
        
        # Use appropriate model based on arm
        if arm == "left":
            sp_points, ey_min, ey_max = self.ey_valid_from_sp(sp)
            ey_points, sp_min, sp_max = self.sp_valid_from_ey(ey)
        elif arm == "right":
            sp_points, ey_min, ey_max = self.ey_valid_from_sp_right(sp)
            ey_points, sp_min, sp_max = self.sp_valid_from_ey_right(ey)
        
        ey_is_valid_for_sp = self._in_interval(sp, ey, sp_points, ey_min, ey_max)
        sp_is_valid_for_ey = self._in_interval(ey, sp, ey_points, sp_min, sp_max)
        
        if not (ey_is_valid_for_sp and sp_is_valid_for_ey):
            return "NON_HUMAN_DOABLE"
        return "HUMAN_DOABLE"
