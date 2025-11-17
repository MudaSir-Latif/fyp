"""
Real-time / video detection script using your 7-layer pushup model
(model_pushup_7layer.h5) and scaler (scaler_pushup.save).

Place this script next to:
- ./serious/model_pushup_7layer.h5
- ./serious/scaler_pushup.save

Usage:
    python core/bicep_model/detect_with_7layer.py --video PATH_TO_VIDEO   # run on a video file
    python core/bicep_model/detect_with_7layer.py --webcam               # run on webcam

This is adapted from your notebook: it uses the same IMPORTANT_LMS ordering
and extracts mediapipe landmarks per-frame, scales with the saved scaler,
and runs the 7-layer model to predict posture label per-frame ('C' or 'L').
It also reuses the BicepPoseAnalysis helper to compute per-arm counters & errors.
"""
import argparse
import os
import datetime
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from tensorflow.keras.models import load_model

# IMPORTANT_LMS and HEADERS must match the CSV used to train the model
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
HEADERS = ["label"]
for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# mediapipe helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- utility helpers ----------
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def save_frame_as_image(frame, message: str = None, out_dir="./data/logs"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if message:
        cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    fname = os.path.join(out_dir, f"pushup_err_{now}.jpg")
    cv2.imwrite(fname, frame)
    print("Saved error frame:", fname)

def calculate_angle(point1: list, point2: list, point3: list) -> float:
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    angleInRad = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)
    return angleInDeg if angleInDeg <= 180 else 360 - angleInDeg

def extract_important_keypoints(results, important_landmarks: list) -> list:
    try:
        landmarks = results.pose_landmarks.landmark
        data = []
        for lm in important_landmarks:
            keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
            data.extend([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
        return data
    except Exception:
        # return a list of zeros if extraction fails (caller should handle)
        return [0.0] * (len(important_landmarks) * 4)

# ---------- PushupPoseAnalysis: adapted for pushup counting & simple errors ----------
class PushupPoseAnalysis:
    """
    Simple per-side pushup analyzer.

    Counting logic (default): detect elbow angle (shoulder-elbow-wrist).
      - "up" position: elbow angle > up_threshold (arms mostly straight)
      - "down" position: elbow angle < down_threshold (elbows bent)
    We increment the counter when the user returns to the "up" position from "down".

    Also tracks two simple errors:
      - SHALLOW_DEPTH: minimal elbow angle during the down phase did not go below
        `peak_contraction_threshold`.
      - HIPS_SAG: torso angle (shoulder-hip-ankle) deviates below a threshold.
    """
    def __init__(self, side: str,
                 up_threshold: float = 160.0,
                 down_threshold: float = 90.0,
                 peak_contraction_threshold: float = 80.0,
                 hips_sag_threshold: float = 160.0,
                 visibility_threshold: float = 0.65):
        self.side = side.lower()
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.hips_sag_threshold = hips_sag_threshold
        self.visibility_threshold = visibility_threshold

        self.counter = 0
        # start assuming the user is at the top
        self.stage = "up"
        self.is_visible = True
        self.detected_errors = {
            "SHALLOW_DEPTH": 0,
            "HIPS_SAG": 0,
        }

        # tracking minimal elbow angle seen during a down-phase
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None

        # joint placeholders
        self.shoulder = None
        self.elbow = None
        self.wrist = None
        self.hip = None
        self.ankle = None

    def get_joints(self, landmarks) -> bool:
        side = self.side.upper()
        try:
            joints_visibility = [
                landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
                landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
                landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility,
                landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].visibility,
                landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].visibility,
            ]
        except Exception:
            self.is_visible = False
            return False

        is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible

        if not is_visible:
            return False

        self.shoulder = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y,
        ]
        self.elbow = [
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y,
        ]
        self.wrist = [
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y,
        ]
        self.hip = [
            landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].y,
        ]
        self.ankle = [
            landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].y,
        ]
        return True

    def analyze_pose(self, landmarks, frame=None):
        """
        Returns a tuple: (elbow_angle, torso_angle)
        elbow_angle: shoulder-elbow-wrist
        torso_angle: shoulder-hip-ankle (used to detect hip sag)
        """
        self.get_joints(landmarks)
        if not self.is_visible:
            return (None, None)

        elbow_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))

        # Counting logic: detect transition from down -> up
        if elbow_angle < self.down_threshold:
            # user is at bottom
            if self.stage != "down":
                self.stage = "down"
            # update minimal elbow angle for this rep
            if elbow_angle < self.peak_contraction_angle:
                self.peak_contraction_angle = elbow_angle
                self.peak_contraction_frame = frame
        elif elbow_angle > self.up_threshold and self.stage == "down":
            # returned to top from bottom - count a rep
            self.stage = "up"
            self.counter += 1
            # evaluate shallow depth (if min angle during down was not low enough)
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle > self.peak_contraction_threshold:
                if self.peak_contraction_frame is not None:
                    # optionally save a debug frame
                    pass
                self.detected_errors["SHALLOW_DEPTH"] += 1
            # reset
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None

        # Torso angle to detect hips sagging: smaller angle -> more sag
        torso_angle = int(calculate_angle(self.shoulder, self.hip, self.ankle))
        if torso_angle < self.hips_sag_threshold:
            # count as a hips sag event once per occurrence
            self.detected_errors["HIPS_SAG"] += 1

        return (elbow_angle, torso_angle)

# ---------- detection / main loop ----------
def detect(model_path: str, scaler_path: str, video_source: str = None, use_webcam: bool = False, rescale_percent: int = 50):
    # load scaler and model
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(scaler_path)

    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    print("Loaded model:", model_path)
    print("Loaded scaler:", scaler_path)

    # init analyzers (pushup-specific)
    left_arm = PushupPoseAnalysis(side="left")
    right_arm = PushupPoseAnalysis(side="right")

    # video capture
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # rescale for speed
            frame = rescale_frame(frame, percent=rescale_percent)
            video_dimensions = [frame.shape[1], frame.shape[0]]

            # mediapipe expects RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            if not results.pose_landmarks:
                # draw info and continue
                cv2.putText(frame, "No human detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Pushup Detection (7-layer)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # draw pose
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # analyze arms
            landmarks = results.pose_landmarks.landmark
            left_angles = left_arm.analyze_pose(landmarks, frame=image_bgr)
            right_angles = right_arm.analyze_pose(landmarks, frame=image_bgr)

            # extract model input
            row = extract_important_keypoints(results, IMPORTANT_LMS)
            X = np.array([row], dtype=np.float32)
            # if scaler expects 2D shape
            X_scaled = scaler.transform(X)

            # model prediction (handle binary single-output or 2-output softmax)
            preds = model.predict(X_scaled, verbose=0)
            if preds.ndim == 2 and preds.shape[1] == 2:
                pred_prob = float(np.max(preds))
                pred_class = int(np.argmax(preds, axis=1)[0])
            elif preds.ndim == 2 and preds.shape[1] == 1:
                pred_prob = float(preds[0,0])
                pred_class = 1 if pred_prob >= 0.5 else 0
            elif preds.ndim == 1:
                # shape (2,) or (,) - try to handle
                if preds.size == 2:
                    pred_prob = float(np.max(preds))
                    pred_class = int(np.argmax(preds))
                else:
                    pred_prob = float(preds[0])
                    pred_class = 1 if pred_prob >= 0.5 else 0
            else:
                pred_prob = 0.0
                pred_class = 0

            label_str = "C" if pred_class == 0 else "L"

            # overlay info
            cv2.rectangle(image_bgr, (0,0), (420,48), (245,117,16), -1)
            cv2.putText(image_bgr, f"Pred: {label_str} ({pred_prob:.2f})", (10,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(image_bgr, f"R:{right_arm.counter if right_arm.is_visible else 'UNK'} L:{left_arm.counter if left_arm.is_visible else 'UNK'}",
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # show some angles if visible
            if left_arm.is_visible and left_angles[0] is not None:
                elbow_pos = tuple((np.multiply(left_arm.elbow, video_dimensions).astype(int)).tolist())
                cv2.putText(image_bgr, f"{left_angles[0]}", elbow_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if right_arm.is_visible and right_angles[0] is not None:
                elbow_pos = tuple((np.multiply(right_arm.elbow, video_dimensions).astype(int)).tolist())
                cv2.putText(image_bgr, f"{right_angles[0]}", elbow_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # optional: save frames when predictions are 'L' with high probability
            if label_str == "L" and pred_prob >= 0.9:
                save_frame_as_image(image_bgr, message=f"High-conf L ({pred_prob:.2f})")

            cv2.imshow("Pushup Detection (7-layer)", image_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to video file (if omitted and --webcam not set, script will error)")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as source")
    parser.add_argument("--model", type=str, default="./serious/model_pushup_7layer_khuari.h5", help="Path to trained model")
    parser.add_argument("--scaler", type=str, default="./serious/scaler_pushup_7layer_khuari.save", help="Path to saved scaler (joblib)")
    parser.add_argument("--rescale", type=int, default=50, help="Frame rescale percent")
    args = parser.parse_args()

    if not args.webcam and not args.video:
        parser.error("Specify --video PATH or --webcam")

    detect(model_path=args.model, scaler_path=args.scaler, video_source=args.video, use_webcam=args.webcam, rescale_percent=args.rescale)