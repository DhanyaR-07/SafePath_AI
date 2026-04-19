# main.py
# SafePath AI – Entry Point
# Ties together the tracker, predictor, and visualizer modules
# to process a warehouse / industrial video and highlight collision risks.
#
# Usage:
#   python main.py
#   python main.py --source data/input_videos/test_video.mp4
#   python main.py --source 0          # webcam
#   python main.py --save              # write output video

import argparse
import sys
from pathlib import Path
import numpy as np

import cv2

# ── Local modules ────────────────────────────────────────────────────────────
from src.tracker   import ObjectTracker
from src.predictor import TrajectoryPredictor
from src.visualizer import draw_frame

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_VIDEO  = "data/input_videos/*.mp4"
DEFAULT_MODEL  = "yolo11n.pt"   # Swap for yolo11s.pt / yolo11m.pt as needed
OUTPUT_DIR     = Path("data/output_videos")
WINDOW_NAME    = "SafePath AI – Invisible Corridor"

# 1. Define 4 points in the video (x, y)
SRC_PTS = np.array([[450, 600], [1400, 600], [1800, 1000], [100, 1000]], dtype=np.float32)

DST_PTS = np.array([[0, 0], [1000, 0], [1000, 1500], [0, 1500]], dtype=np.float32)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafePath AI – collision prediction")
    parser.add_argument("--source", default=DEFAULT_VIDEO,
                        help="Video file path or webcam index (default: test_video.mp4)")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help="YOLOv11 weights file (default: yolo11n.pt)")
    parser.add_argument("--conf",   type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("--save",   action="store_true",
                        help="Save the annotated output video to disk")
    parser.add_argument("--no-show", action="store_true",
                        help="Suppress the live preview window")
    return parser.parse_args()


def open_video(source: str) -> cv2.VideoCapture:
    """Open a video file or webcam; exit on failure."""
    cap_src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)
    return cap


def make_writer(cap: cv2.VideoCapture, out_path: Path) -> cv2.VideoWriter:
    """Create an mp4 VideoWriter matching the input stream's resolution & FPS."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))


def main() -> None:
    args = parse_args()

    # ── Initialise components ────────────────────────────────────────────────
    tracker   = ObjectTracker(model_path=args.model, confidence=args.conf)
    predictor = TrajectoryPredictor()

    # ── Open video source ────────────────────────────────────────────────────
    cap = open_video(args.source)

    # Calculate Homography Matrix (Image -> Ground) and Inverse (Ground -> Image)
    H, _ = cv2.findHomography(SRC_PTS, DST_PTS)
    H_inv, _ = cv2.findHomography(DST_PTS, SRC_PTS)

    # ── Optional output writer ───────────────────────────────────────────────
    writer = None
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        src_stem  = Path(args.source).stem if not args.source.isdigit() else "webcam"
        out_path  = OUTPUT_DIR / f"{src_stem}_safepath.mp4"
        writer    = make_writer(cap, out_path)
        print(f"[Main] Saving annotated video → {out_path}")

    print("[Main] Processing… press 'q' to quit.")

    # Read FPS once; predictor uses it to project exactly 3 seconds ahead.
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[Main] Video FPS: {fps:.1f}  →  predicting {3 * fps:.0f} frames ahead")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] End of stream.")
            break

        # ── 1. Detect & track objects in this frame ──────────────────────────
        detections = tracker.track_frame(frame)

        # ── 2. Update predictor with new positions ────────────────────────────
        #for det in detections:
            #predictor.update(det["track_id"], det["center"])

        # ── 2. Update predictor with GROUND positions ────────────────────────
        for det in detections:
            # a. Prepare the pixel coordinate
            px_coord = np.array([[[det["center"][0], det["center"][1]]]], dtype=np.float32)
            
            # b. Transform pixel -> ground
            ground_coord = cv2.perspectiveTransform(px_coord, H)[0][0]
            
            # c. Update predictor with ground units
            predictor.update(det["track_id"], (int(ground_coord[0]), int(ground_coord[1])))    

        # ── 3. Build history & prediction maps keyed by track_id ─────────────
        #histories   = {det["track_id"]: predictor.get_history(det["track_id"])
                     #for det in detections}
        #predictions = {det["track_id"]: predictor.predict(det["track_id"], fps)
                       #for det in detections} 
        # ── 3. Build history & prediction maps (and transform back to pixels) ─
        raw_histories   = {det["track_id"]: predictor.get_history(det["track_id"]) for det in detections}
        raw_predictions = {det["track_id"]: predictor.predict(det["track_id"], fps) for det in detections}

        histories = {}
        predictions = {}

        for tid, path in raw_histories.items():
            if not path: continue
            pts = np.array([path], dtype=np.float32)
            histories[tid] = cv2.perspectiveTransform(pts, H_inv)[0].astype(int).tolist()

        for tid, path in raw_predictions.items():
            if not path: continue
            pts = np.array([path], dtype=np.float32)
            predictions[tid] = cv2.perspectiveTransform(pts, H_inv)[0].astype(int).tolist()
        # ── 4. Render the annotated frame ─────────────────────────────────────
        annotated = draw_frame(frame, detections, histories, predictions)

        # ── 5. Display / save ─────────────────────────────────────────────────
        if not args.no_show:
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Main] User quit.")
                break

        if writer:
            writer.write(annotated)

        frame_idx += 1

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[Main] Done — processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
