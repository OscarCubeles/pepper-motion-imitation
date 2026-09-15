class EMASmoothing:
    """
    Exponential Moving Average smoothing for orientation vectors.
    Helps reduce jitter and occlusion artifacts.
    Discards values with low confidence to prevent corrupting the smoothed estimate.
    """
    def __init__(self, alpha=0.2, confidence_threshold=0.9):
        """
        Args:
            alpha: smoothing factor (0-1). Higher = more responsive, lower = smoother
            confidence_threshold: minimum confidence to apply smoothing (0-1)
        """
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.previous_value = None
    
    def smooth(self, current_value, confidence=1.0):
        """
        Apply EMA smoothing to current value if confidence is high enough.
        
        Args:
            current_value: numpy array of current measurement
            confidence: confidence score (0-1). If below threshold, keeps previous value
        
        Returns:
            smoothed value
        """
        if self.previous_value is None:
            self.previous_value = current_value.copy()
            return current_value
        
        # If confidence is too low, don't update - keep previous smoothed value
        if confidence < self.confidence_threshold:
            return self.previous_value
        
        # EMA formula: smoothed = alpha * current + (1 - alpha) * previous
        smoothed = self.alpha * current_value + (1 - self.alpha) * self.previous_value
        self.previous_value = smoothed.copy()
        return smoothed
    
    def reset(self):
        """Reset the smoothing state."""
        self.previous_value = None
