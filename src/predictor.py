# src/predictor.py
# SafePath AI - Trajectory Prediction Module
# Stores a 12-frame coordinate history per track ID and predicts
# exactly 3 seconds ahead using a moving-average velocity (smoothing).
# Stationary objects (speed below threshold) return no prediction.

from collections import deque
import numpy as np

# ── Tunable constants ────────────────────────────────────────────────────────
SMOOTHING_WINDOW   = 12    # frames used for moving-average velocity (≈0.4 s @ 30 fps)
PREDICT_SECONDS    = 3     # seconds to project the Invisible Corridor ahead
MIN_VELOCITY_PX    = 1.5   # px/frame – objects slower than this are considered stationary


class TrajectoryPredictor:
    """
    Maintains per-object coordinate histories and generates
    smoothed linear ghost-path predictions 3 seconds into the future.

    Velocity estimation:
        The mean frame-to-frame displacement over the last SMOOTHING_WINDOW
        frames is used as the velocity vector.  This acts as a moving-average
        filter that removes jitter without adding significant latency.

    FPS-aware projection:
        predict() requires the current video FPS so that the corridor always
        covers exactly PREDICT_SECONDS regardless of camera frame-rate.

    Stationary filter:
        If the magnitude of the smoothed velocity is below MIN_VELOCITY_PX,
        predict() returns [] so no ghost-path is drawn for idle objects.
    """

    def __init__(self,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 predict_seconds:  float = PREDICT_SECONDS,
                 min_velocity_px:  float = MIN_VELOCITY_PX):
        """
        Args:
            smoothing_window: Number of frames used for moving-average velocity.
            predict_seconds:  How many seconds ahead to project the path.
            min_velocity_px:  Speed threshold below which no path is drawn.
        """
        self.smoothing_window = smoothing_window
        self.predict_seconds  = predict_seconds
        self.min_velocity_px  = min_velocity_px

        # Maps track_id → deque of (cx, cy) tuples (sliding window)
        self._histories: dict[int, deque] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, track_id: int, center: tuple[int, int]) -> None:
        """
        Push a new observed position into the history for a given track.

        Args:
            track_id: Unique integer identifier from the tracker.
            center:   (cx, cy) pixel coordinate observed this frame.
        """
        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self.smoothing_window)
        self._histories[track_id].append(center)

    def predict(self, track_id: int, fps: float) -> list[tuple[int, int]]:
        """
        Predict future positions using a smoothed constant-velocity model.

        Velocity is the moving average of frame-to-frame displacements over
        the stored history window, providing 3-second temporal smoothing.
        The corridor is projected for exactly `predict_seconds * fps` frames.

        Args:
            track_id: The object whose trajectory to predict.
            fps:      Video frame-rate (used to convert seconds → frames).

        Returns:
            List of (cx, cy) tuples for the next (predict_seconds * fps) frames.
            Returns [] if:
                • fewer than 2 history points are available, OR
                • the object is stationary (speed < min_velocity_px px/frame).
        """
        history = self._histories.get(track_id)
        if history is None or len(history) < 2:
            return []

        # Convert history to numpy array, shape (N, 2)
        pts = np.array(history, dtype=float)

        # ── Moving-average velocity (3-second smoothing window) ──────────────
        # Use all available history up to smoothing_window frames.
        deltas   = np.diff(pts, axis=0)       # shape (N-1, 2)
        velocity = deltas.mean(axis=0)        # shape (2,) → (vx, vy)

        # ── Stationary filter ─────────────────────────────────────────────────
        speed = float(np.linalg.norm(velocity))
        if speed < self.min_velocity_px:
            return []   # Object is not moving – skip ghost-path

        # ── FPS-aware 3-second projection ─────────────────────────────────────
        predict_frames = max(1, int(round(self.predict_seconds * fps)))
        last = pts[-1]
        future = [
            (int(last[0] + velocity[0] * t),
             int(last[1] + velocity[1] * t))
            for t in range(1, predict_frames + 1)
        ]
        return future

    def get_history(self, track_id: int) -> list[tuple[int, int]]:
        """Return stored history as a plain list (oldest → newest)."""
        hist = self._histories.get(track_id)
        return list(hist) if hist else []

    def remove(self, track_id: int) -> None:
        """Evict a track that is no longer visible (optional housekeeping)."""
        self._histories.pop(track_id, None)

