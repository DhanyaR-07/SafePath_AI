# src/predictor.py
# SafePath AI - Advanced Trajectory Prediction Module
# Uses a Constant Velocity Kalman Filter for smoothing and 
# calculates dynamic braking distances based on real-world physics.

import cv2
import numpy as np
from collections import deque

# ── Safety & Physics Constants ───────────────────────────────────────────────
# These reflect professional safety standards, similar to Kavach systems.
REACTION_TIME_SEC  = 0.8   # System + Driver reaction time (seconds)
FRICTION_COEFF     = 0.3   # Average friction for warehouse concrete floors
GRAVITY            = 980.6 # cm/s^2 (Standard for centimeter-based units)
MIN_VELOCITY_UNIT  = 2.0   # Threshold to ignore stationary noise (cm/frame)
SMOOTHING_WINDOW   = 15    # Number of frames for visual history trail

class TrajectoryPredictor:
    """
    Advanced predictor utilizing a Kalman Filter for state estimation
    and dynamic path projection based on braking physics.
    """

    def __init__(self):
        # Maps track_id -> cv2.KalmanFilter object
        self.kalmans = {}
        # Maps track_id -> deque of ground coordinates for visual history
        self._histories = {}

    def _init_kalman(self, initial_x, initial_y):
        """
        Initializes a 4-state Kalman Filter: [x, y, vx, vy]
        """
        # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
        kf = cv2.KalmanFilter(4, 2)
        
        # State Transition Matrix (F): Constant Velocity Model
        # x_new = x + vx, y_new = y + vy
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

        # Measurement Matrix (H): We only observe (x, y) ground positions
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

        # Process Noise Covariance (Q): Trust in the physics model
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        
        # Measurement Noise Covariance (R): Trust in YOLO detections
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # Error Covariance (P): Initial state uncertainty
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Initial State: Start at the detected ground point with zero initial velocity
        kf.statePost = np.array([[initial_x], [initial_y], [0], [0]], np.float32)
        
        return kf

    def update(self, track_id: int, ground_coord: tuple[int, int]) -> None:
        """
        Update the state estimate using the Kalman Predict/Correct cycle.
        """
        gx, gy = ground_coord

        if track_id not in self.kalmans:
            self.kalmans[track_id] = self._init_kalman(gx, gy)
            self._histories[track_id] = deque(maxlen=SMOOTHING_WINDOW)

        # 1. Kalman Predict Phase: Project state based on physics
        self.kalmans[track_id].predict()

        # 2. Kalman Correct Phase: Update state with new detection
        measurement = np.array([[np.float32(gx)], [np.float32(gy)]])
        self.kalmans[track_id].correct(measurement)

        # Store for history trail (Bird's Eye View coordinates)
        self._histories[track_id].append((gx, gy))

    def predict(self, track_id: int, fps: float) -> list[tuple[int, int]]:
        """
        Predicts the path using Dynamic Braking Distance logic.
        """
        if track_id not in self.kalmans:
            return []

        kf = self.kalmans[track_id]
        state = kf.statePost # Current estimated [x, y, vx, vy]
        
        vx, vy = state[2][0], state[3][0]
        speed_per_frame = np.sqrt(vx**2 + vy**2)
        
        # Convert speed to units per second (e.g., cm/s)
        speed_per_sec = speed_per_frame * fps

        if speed_per_frame < MIN_VELOCITY_UNIT:
            return [] # Stationary filter

        # ── Dynamic Braking Calculation ──────────────────────────────────────
        # Derived from industrial safety principles (Kavach).
        # Total distance = (Reaction Distance) + (Braking Distance)
        # d = (v * t_reaction) + (v^2 / (2 * friction * gravity))
        
        reaction_dist = speed_per_sec * REACTION_TIME_SEC
        braking_dist  = (speed_per_sec**2) / (2 * FRICTION_COEFF * GRAVITY)
        total_dist_ground = reaction_dist + braking_dist
        
        # Map physical distance back to frame-based prediction length
        predict_frames = int(round(total_dist_ground / speed_per_frame))

        # Project the linear path forward from the current state
        curr_x, curr_y = state[0][0], state[1][0]
        future = [
            (int(curr_x + vx * t), int(curr_y + vy * t))
            for t in range(1, predict_frames + 1)
        ]
        return future

    def get_history(self, track_id: int) -> list[tuple[int, int]]:
        """Returns the smoothed ground-coordinate history."""
        return list(self._histories.get(track_id, []))

    def remove(self, track_id: int) -> None:
        """Cleanup for objects no longer in frame."""
        self.kalmans.pop(track_id, None)
        self._histories.pop(track_id, None)