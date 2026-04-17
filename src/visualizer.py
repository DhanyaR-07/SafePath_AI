# src/visualizer.py
# SafePath AI - Visualisation & Collision-Risk Module
# Draws the "Invisible Corridor" (predicted ghost-paths) on each frame.
#
# Key behaviours in this version:
#   • Forklift paths get a wide semi-transparent "Danger Zone" polygon.
#   • A 'COLLISION RISK' alert fires when any person's predicted path
#     enters a forklift's Danger Zone polygon.
#   • Ghost-paths are only drawn for moving objects (velocity-filtered
#     by the predictor; empty predictions mean stationary → skip).
#   • Display label uses 'Forklift' (never 'truck').

import cv2
import numpy as np

# ── Visual style constants ───────────────────────────────────────────────────
PERSON_COLOR      = (0,   255,   0)   # Green  – person ghost-path
FORKLIFT_COLOR    = (0,   165, 255)   # Orange – forklift ghost-path
DANGER_ZONE_COLOR = (0,    50, 220)   # Deep red-blue – forklift danger polygon
HISTORY_COLOR     = (160, 160, 160)   # Light grey – past trail
ALERT_COLOR       = (0,     0, 255)   # Pure red – COLLISION RISK banner
BBOX_PERSON       = (50,  255,  50)
BBOX_FORKLIFT     = (50,  165, 255)

GHOST_THICKNESS   = 2    # px – predicted path line
HISTORY_THICKNESS = 1    # px – past trail
POINT_RADIUS      = 3    # px – dots along ghost-path
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# Half-width of the forklift Danger Zone corridor (pixels either side of path)
DANGER_ZONE_WIDTH = 55   # px – adjust to match physical forklift width in scene

# Alpha for the semi-transparent danger polygon overlay
DANGER_ZONE_ALPHA = 0.28


def draw_frame(
    frame:       np.ndarray,
    detections:  list[dict],
    histories:   dict[int, list],
    predictions: dict[int, list],
) -> np.ndarray:
    """
    Render all visual elements onto a copy of `frame`.

    Args:
        frame:       Original BGR frame from OpenCV.
        detections:  List of detection dicts from tracker
                     (keys: track_id, label, center, bbox).
        histories:   Mapping of track_id → list of past (cx,cy) positions.
        predictions: Mapping of track_id → list of predicted (cx,cy) positions.
                     Empty list means the object is stationary → nothing drawn.

    Returns:
        Annotated BGR frame (same dimensions as input).
    """
    canvas = frame.copy()

    # Split detections by class for later polygon/path checks
    forklift_dets = [d for d in detections if d["label"] == "Forklift"]
    person_dets   = [d for d in detections if d["label"] == "person"]

    # ── 1. Bounding boxes & labels ───────────────────────────────────────────
    for det in detections:
        tid   = det["track_id"]
        label = det["label"]
        x1, y1, x2, y2 = det["bbox"]
        color = BBOX_FORKLIFT if label == "Forklift" else BBOX_PERSON

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{label} #{tid}", (x1, y1 - 8),
                    FONT, 0.55, color, 2, cv2.LINE_AA)

    # ── 2. Historical trails ─────────────────────────────────────────────────
    for tid, hist in histories.items():
        if len(hist) < 2:
            continue
        for i in range(len(hist) - 1):
            cv2.line(canvas, hist[i], hist[i + 1], HISTORY_COLOR, HISTORY_THICKNESS)

    # ── 3. Build forklift Danger Zone polygons ───────────────────────────────
    #   Each forklift's predicted path is expanded into a wide corridor polygon.
    forklift_zones: list[np.ndarray] = []   # list of (N,2) int32 polygon arrays

    for det in forklift_dets:
        tid  = det["track_id"]
        path = predictions.get(tid, [])
        if len(path) < 2:
            continue
        poly = _path_to_corridor_polygon(path, DANGER_ZONE_WIDTH)
        if poly is not None:
            forklift_zones.append(poly)

    # Draw semi-transparent danger polygons
    if forklift_zones:
        overlay = canvas.copy()
        for poly in forklift_zones:
            cv2.fillPoly(overlay, [poly], DANGER_ZONE_COLOR)
        cv2.addWeighted(overlay, DANGER_ZONE_ALPHA, canvas,
                        1 - DANGER_ZONE_ALPHA, 0, canvas)
        # Draw polygon border
        for poly in forklift_zones:
            cv2.polylines(canvas, [poly], isClosed=True,
                          color=FORKLIFT_COLOR, thickness=2, lineType=cv2.LINE_AA)

    # ── 4. Check for COLLISION RISK (person path ∩ forklift danger zone) ─────
    collision_points: list[tuple[int, int]] = []

    for det in person_dets:
        tid  = det["track_id"]
        path = predictions.get(tid, [])
        for pt in path:
            for poly in forklift_zones:
                # pointPolygonTest returns > 0 if point is inside polygon
                if cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0:
                    collision_points.append(pt)
                    break   # One hit per person per forklift zone is enough
            else:
                continue
            break

    # ── 5. Ghost-paths (Invisible Corridor) ──────────────────────────────────
    #   Stationary objects have empty predictions → nothing drawn (velocity filter).
    for det in detections:
        tid   = det["track_id"]
        label = det["label"]
        path  = predictions.get(tid, [])

        if len(path) < 2:
            continue   # Stationary or not enough data – skip (velocity filtered)

        ghost_color = FORKLIFT_COLOR if label == "Forklift" else PERSON_COLOR

        for i in range(len(path) - 1):
            cv2.line(canvas, path[i], path[i + 1], ghost_color,
                     GHOST_THICKNESS, cv2.LINE_AA)

        for pt in path:
            cv2.circle(canvas, pt, POINT_RADIUS, ghost_color, -1)

    # ── 6. Collision risk alert markers ──────────────────────────────────────
    for pt in collision_points:
        # Pulsing ring
        cv2.circle(canvas, pt, 22, ALERT_COLOR, 3, cv2.LINE_AA)
        cv2.circle(canvas, pt,  8, ALERT_COLOR, -1)
        cv2.putText(canvas, "COLLISION RISK",
                    (pt[0] + 26, pt[1] + 6),
                    FONT, 0.65, ALERT_COLOR, 2, cv2.LINE_AA)

    # ── 7. Full-frame banner if any collision risk exists ─────────────────────
    if collision_points:
        _draw_alert_banner(canvas)

    # ── 8. HUD ───────────────────────────────────────────────────────────────
    _draw_hud(canvas, detections, len(collision_points))

    return canvas


# ── Internal helpers ─────────────────────────────────────────────────────────

def _path_to_corridor_polygon(
    path: list[tuple[int, int]],
    half_width: int,
) -> np.ndarray | None:
    """
    Expand a centre-line path into a filled corridor (rectangle-capped) polygon.

    For each consecutive pair of path points the perpendicular offset vector
    is computed; the four corners of each segment are accumulated.
    The result is a convex-hull polygon suitable for cv2.fillPoly.

    Args:
        path:       Ordered list of (cx, cy) tuples along the predicted path.
        half_width: Half the desired corridor width in pixels.

    Returns:
        Integer (N,1,2) numpy polygon array, or None if path is degenerate.
    """
    if len(path) < 2:
        return None

    left_edge  = []
    right_edge = []

    for i in range(len(path) - 1):
        p1 = np.array(path[i],     dtype=float)
        p2 = np.array(path[i + 1], dtype=float)

        direction = p2 - p1
        length    = np.linalg.norm(direction)
        if length < 1e-6:
            continue

        # Unit perpendicular (rotated 90°)
        perp = np.array([-direction[1], direction[0]]) / length * half_width

        left_edge.append((p1 + perp).astype(int))
        left_edge.append((p2 + perp).astype(int))
        right_edge.append((p1 - perp).astype(int))
        right_edge.append((p2 - perp).astype(int))

    if not left_edge:
        return None

    # Combine edges into a closed polygon (left side forward, right side reversed)
    pts_all = np.array(left_edge + right_edge[::-1], dtype=np.int32)

    # Convex hull gives a clean outline even for curved paths
    hull = cv2.convexHull(pts_all.reshape(-1, 1, 2))
    return hull


def _draw_alert_banner(canvas: np.ndarray) -> None:
    """Draw a prominent red banner at the top of the frame."""
    h, w = canvas.shape[:2]
    banner_h = 44

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    text = "⚠  COLLISION RISK DETECTED  ⚠"
    text_size, _ = cv2.getTextSize(text, FONT, 0.75, 2)
    tx = (w - text_size[0]) // 2
    cv2.putText(canvas, text, (tx, 30), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_hud(
    canvas:      np.ndarray,
    detections:  list[dict],
    risk_count:  int,
) -> None:
    """Overlay a minimal heads-up display in the top-left corner."""
    persons   = sum(1 for d in detections if d["label"] == "person")
    forklifts = sum(1 for d in detections if d["label"] == "Forklift")

    lines = [
        f"Persons   : {persons}",
        f"Forklifts : {forklifts}",
        f"Risk zones: {risk_count}",
    ]
    x, y0, dy = 12, 60, 24   # y0=60 leaves room for the alert banner
    for i, text in enumerate(lines):
        color = ALERT_COLOR if (i == 2 and risk_count > 0) else (230, 230, 230)
        cv2.putText(canvas, text, (x, y0 + i * dy),
                    FONT, 0.65, (0, 0, 0), 4, cv2.LINE_AA)   # drop-shadow
        cv2.putText(canvas, text, (x, y0 + i * dy),
                    FONT, 0.65, color,     2, cv2.LINE_AA)

