# src/tracker.py
# SafePath AI - Object Detection & Tracking Module
# Uses YOLOv11 (via ultralytics) to detect 'person' and 'Forklift' (truck proxy)
# Returns track IDs and center coordinates for every detected object simultaneously.

from ultralytics import YOLO
import numpy as np

# YOLO class indices for COCO dataset.
# The COCO 'truck' class (7) serves as the forklift proxy.
# Display label is 'Forklift' for clarity in the warehouse context.
TRACKED_CLASSES = {
    0: "person",
    7: "Forklift",  # Renamed from 'truck' – displayed as 'Forklift' everywhere
}

class ObjectTracker:
    """
    Wraps YOLOv11 with ByteTrack for unlimited multi-object tracking (MOT).
    All detected 'person' and 'Forklift' instances are tracked simultaneously
    with no per-frame cap on the number of active tracks.
    """

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.4):
        """
        Args:
            model_path: Path or name of the YOLOv11 model weights.
            confidence: Minimum confidence threshold for detections.
        """
        print(f"[Tracker] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_ids = list(TRACKED_CLASSES.keys())  # [0, 7]

    def track_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection + tracking on a single BGR frame.
        All instances of every tracked class are returned – no object limit.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            List of dicts, each containing:
                - 'track_id' (int): Unique persistent ID from ByteTrack
                - 'label'    (str): 'person' or 'Forklift'
                - 'center'   (tuple[int, int]): (cx, cy) in pixel coords
                - 'bbox'     (tuple[int,int,int,int]): (x1, y1, x2, y2)
        """
        results = self.model.track(
            frame,
            persist=True,               # Maintain track IDs across frames
            classes=self.class_ids,     # Only detect person & Forklift (truck)
            conf=self.confidence,
            tracker="bytetrack.yaml",   # Robust multi-object tracker
            max_det=1000,               # High cap → effectively unlimited MOT
            verbose=False,
        )

        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                # Skip if ByteTrack hasn't assigned an ID yet
                if box.id is None:
                    continue

                track_id  = int(box.id.item())
                class_id  = int(box.cls.item())
                label     = TRACKED_CLASSES.get(class_id, "unknown")

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "track_id": track_id,
                    "label":    label,
                    "center":   (cx, cy),
                    "bbox":     (x1, y1, x2, y2),
                })

        return detections
