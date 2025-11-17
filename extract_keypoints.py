# """
# Batch-extract mediapipe pose landmarks from videos and label frames automatically.

# Usage examples (at bottom of file):
# - label by folder names ("correct" / "incorrect")
# - or label by filename containing 'C'/'W' or 'correct'/'incorrect'

# The CSV format matches the original notebook: first column 'label', then
# '<lm>_x', '<lm>_y', '<lm>_z', '<lm>_v' for IMPORTANT_LMS order.
# """
# import os
# import csv
# import cv2
# import numpy as np
# import mediapipe as mp

# # --- configuration / important landmarks (same as your notebook) ---
# IMPORTANT_LMS = [
#     "NOSE",
#     "LEFT_SHOULDER",
#     "RIGHT_SHOULDER",
#     "LEFT_ELBOW",
#     "RIGHT_ELBOW",
#     "LEFT_WRIST",
#     "RIGHT_WRIST",
#     "LEFT_HIP",
#     "RIGHT_HIP",
#     "LEFT_KNEE",
#     "RIGHT_KNEE",
#     "LEFT_ANKLE",
#     "RIGHT_ANKLE",
#     "LEFT_HEEL",
#     "RIGHT_HEEL",
#     "LEFT_FOOT_INDEX",
#     "RIGHT_FOOT_INDEX",
# ]

# HEADERS = ["label"]
# for lm in IMPORTANT_LMS:
#     HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # --- helpers (kept small and self-contained) ---
# def rescale_frame(frame, percent=60):
#     width = int(frame.shape[1] * percent / 100)
#     height = int(frame.shape[0] * percent / 100)
#     return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# def init_csv(dataset_path: str):
#     if os.path.exists(dataset_path):
#         return
#     with open(dataset_path, mode="w", newline="") as f:
#         csv_writer = csv.writer(f)
#         csv_writer.writerow(HEADERS)

# def export_landmark_row(csv_path: str, pose_landmarks, label: str):
#     try:
#         landmarks = pose_landmarks.landmark
#         keypoints = []
#         for lm in IMPORTANT_LMS:
#             kp = landmarks[mp_pose.PoseLandmark[lm].value]
#             keypoints.extend([kp.x, kp.y, kp.z, kp.visibility])
#         row = [label] + list(keypoints)
#         with open(csv_path, mode="a", newline="") as f:
#             csv_writer = csv.writer(f)
#             csv_writer.writerow(row)
#     except Exception:
#         # ignore frames where pose isn't available or other issues
#         pass

# def infer_label_from_folder(folder_name: str):
#     # simple mapping, customize as needed
#     name = folder_name.lower()
#     # IMPORTANT: check for 'incorrect' before 'correct' because
#     # 'incorrect' contains the substring 'correct' and would match
#     # the wrong branch otherwise.
#     if "incorrect" in name or "error" in name or "l" == name or "lean" in name:
#         return "L"
#     if "correct" in name or "c" == name:
#         return "C"
#     # default: return folder basename (uppercased first char)
#     return folder_name[:1].upper()

# def infer_label_from_filename(filename: str):
#     low = filename.lower()
#     # check for 'incorrect' first (it contains 'correct' as substring)
#     if "incorrect" in low or "_l" in low or "-l" in low or "lean" in low or "_wrong" in low:
#         return "L"
#     if "correct" in low or "_c" in low or "-c" in low or "_correct" in low:
#         return "C"
#     # fallback: check single letter 'c' or 'l' before extension
#     base = os.path.splitext(filename)[0]
#     if base.endswith("_c") or base.endswith("-c") or base.endswith("c"):
#         return "C"
#     if base.endswith("_l") or base.endswith("-l") or base.endswith("l"):
#         return "L"
#     return None

# # --- main processing function ---
# def process_videos(video_paths, output_csv="train.csv", label_source="folder", frame_step=5, scale_percent=60):
#     """
#     video_paths: list of paths OR list of folders containing videos
#     label_source: "folder" to use folder name -> label (recommended),
#                   "filename" to infer label from filename
#     frame_step: sample every Nth frame (1 = keep all)
#     """
#     init_csv(output_csv)
#     total_saved = 0
#     # Initialize mediapipe once (faster)
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         for path in video_paths:
#             if os.path.isdir(path):
#                 # iterate videos inside directory
#                 for fname in sorted(os.listdir(path)):
#                     if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
#                         continue
#                     video_file = os.path.join(path, fname)
#                     if label_source == "folder":
#                         label = infer_label_from_folder(os.path.basename(path))
#                     else:
#                         label = infer_label_from_filename(fname) or "C"
#                     saved = process_single_video(video_file, pose, output_csv, label, frame_step, scale_percent)
#                     total_saved += saved
#                     print(f"Processed {video_file} -> saved: {saved} rows (label={label})")
#             else:
#                 # single file
#                 fname = os.path.basename(path)
#                 if label_source == "folder":
#                     label = infer_label_from_folder(os.path.basename(os.path.dirname(path)))
#                 else:
#                     label = infer_label_from_filename(fname) or "C"
#                 saved = process_single_video(path, pose, output_csv, label, frame_step, scale_percent)
#                 total_saved += saved
#                 print(f"Processed {path} -> saved: {saved} rows (label={label})")

#     print(f"Total saved rows: {total_saved}")
#     return total_saved

# def process_single_video(video_path, pose, output_csv, label, frame_step=5, scale_percent=60):
#     cap = cv2.VideoCapture(video_path)
#     frame_idx = 0
#     saved_count = 0
#     if not cap.isOpened():
#         print("Cannot open:", video_path)
#         return 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_idx += 1
#         if frame_idx % frame_step != 0:
#             continue
#         try:
#             frame = rescale_frame(frame, percent=scale_percent)
#             # optionally flip if videos are mirrored in your dataset; adjust if needed
#             frame = cv2.flip(frame, 1)

#             # mediapipe expects RGB
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image_rgb.flags.writeable = False
#             results = pose.process(image_rgb)
#             if not results.pose_landmarks:
#                 continue
#             # write a row
#             export_landmark_row(output_csv, results.pose_landmarks, label)
#             saved_count += 1
#         except Exception:
#             # keep processing other frames
#             continue
#     cap.release()
#     return saved_count

# # --- Example usage ---
# if __name__ == "__main__":
#     # Example A: label by folder name
#     # Put videos of correct form under ../data/db_curl/correct/
#     # Put videos of incorrect form under ../data/db_curl/incorrect/
#     folders = [
#         "D:/Fyp/First/both_forms/correct",
#         "D:/Fyp/First/both_forms/incorrect"
#         # "D:\Fyp\First\my_correct",
#         # "D:\Fyp\First\my_wrong"
#     ]
#     # Example B: label from filename instead (if you have e.g. video_C1.mp4, video_L2.mp4)
#     # files = ["../data/db_curl/video_C1.mp4", "../data/db_curl/video_L1.mp4"]

#     OUTPUT = "./train_from_videos.csv"
#     # choose label_source="folder" or "filename"
#     process_videos(folders, output_csv=OUTPUT, label_source="folder", frame_step=5, scale_percent=60)




"""
Batch-extract mediapipe pose landmarks from videos and label frames automatically.

Usage examples (bottom of file):
- label by folder names ("correct" / "incorrect")
-- or label by filename containing 'C'/'W' or 'correct'/'incorrect'

Output CSV format:
-- first column 'label' ('C' or 'W'), then '<lm>_x', '<lm>_y', '<lm>_z', '<lm>_v' for IMPORTANT_LMS order.

Adjust frame_step, scale_percent, and label_source as needed.
"""
import os
import csv
import cv2
import numpy as np
import sys
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- configuration / important landmarks (same ordering used by model) ---
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

# --- helpers ---
def rescale_frame(frame, percent=60):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def init_csv(dataset_path: str):
    if os.path.exists(dataset_path):
        return
    with open(dataset_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(HEADERS)

def _prompt_user_for_frame(frame_bgr, pose_landmarks, label):
    """Show the frame with drawn landmarks and ask user to confirm.

    Returns:
      'y' = save with provided label
      'n' = skip (don't save)
      'f' = flip label and save
      'q' = quit program
    """
    disp = frame_bgr.copy()
    try:
        mp_drawing.draw_landmarks(disp, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    except Exception:
        # drawing failure shouldn't block review
        pass

    text = f"Label: {label}  (y=save, n=skip, f=flip+save, q=quit)"
    cv2.putText(disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    window_name = "Frame confirmation - press key"
    cv2.imshow(window_name, disp)
    while True:
        k = cv2.waitKey(0)
        if k == -1:
            continue
        # Get lower-case ascii character if possible
        try:
            ch = chr(k & 0xFF).lower()
        except Exception:
            ch = ""
        if ch in ("y", "n", "f", "q"):
            cv2.destroyWindow(window_name)
            return ch
        # ignore other keys and continue waiting

def export_landmark_row(csv_path: str, pose_landmarks, label: str):
    try:
        landmarks = pose_landmarks.landmark
        keypoints = []
        for lm in IMPORTANT_LMS:
            kp = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.extend([kp.x, kp.y, kp.z, kp.visibility])
        row = [label] + list(keypoints)
        with open(csv_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
    except Exception:
        # ignore frames where pose isn't available or other issues
        pass

def infer_label_from_folder(folder_name: str):
    # simple mapping, customize as needed
    name = folder_name.lower()
    # IMPORTANT: check for 'incorrect' before 'correct' because
    # 'incorrect' contains the substring 'correct' and would match
    # the wrong branch otherwise.
    if "incorrect" in name or "error" in name or "w" == name or "lean" in name or "wrong" in name:
        return "W"
    if "correct" in name or "c" == name:
        return "C"
    # default: return folder basename (uppercased first char)
    # return folder_name[:1].upper()
    return 'C'

def infer_label_from_filename(filename: str):
    low = filename.lower()
    # check for 'incorrect' first (it contains 'correct' as substring)
    if "incorrect" in low or "_w" in low or "-w" in low or "lean" in low or "_wrong" in low:
        return "W"
    if "correct" in low or "_c" in low or "-c" in low or "_correct" in low:
        return "C"
    # fallback: check single letter 'c' or 'w' before extension
    base = os.path.splitext(filename)[0]
    if base.endswith("_c") or base.endswith("-c") or base.endswith("c"):
        return "C"
    if base.endswith("_w") or base.endswith("-w") or base.endswith("w"):
        return "W"
    return None

# --- main processing function ---
def process_videos(video_paths, output_csv="pushup_train.csv", label_source="folder", frame_step=5, scale_percent=60, flip_horizontal=False, confirm_frames=False):
    """
    video_paths: list of paths OR list of folders containing videos
    label_source: "folder" to use folder name -> label (recommended),
                  "filename" to infer label from filename
    frame_step: sample every Nth frame (1 = keep all)
    flip_horizontal: whether to horizontally flip frames (useful if dataset mirrored)
    """
    init_csv(output_csv)
    total_saved = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for path in video_paths:
            if os.path.isdir(path):
                for fname in sorted(os.listdir(path)):
                    if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        continue
                    video_file = os.path.join(path, fname)
                    if label_source == "folder":
                        label = infer_label_from_folder(os.path.basename(path))
                    else:
                        label = infer_label_from_filename(fname) or "C"
                    saved = process_single_video(video_file, pose, output_csv, label, frame_step, scale_percent, flip_horizontal, confirm_frames)
                    total_saved += saved
                    print(f"Processed {video_file} -> saved: {saved} rows (label={label})")
            else:
                fname = os.path.basename(path)
                if label_source == "folder":
                    label = infer_label_from_folder(os.path.basename(os.path.dirname(path)))
                else:
                    label = infer_label_from_filename(fname) or "C"
                saved = process_single_video(path, pose, output_csv, label, frame_step, scale_percent, flip_horizontal, confirm_frames)
                total_saved += saved
                print(f"Processed {path} -> saved: {saved} rows (label={label})")

    print(f"Total saved rows: {total_saved}")
    return total_saved

def process_single_video(video_path, pose, output_csv, label, frame_step=5, scale_percent=60, flip_horizontal=False, confirm_frames=False):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0
    if not cap.isOpened():
        print("Cannot open:", video_path)
        return 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue
        try:
            frame = rescale_frame(frame, percent=scale_percent)
            if flip_horizontal:
                frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            if not results.pose_landmarks:
                continue

            if confirm_frames:
                decision = _prompt_user_for_frame(frame, results.pose_landmarks, label)
                if decision == "q":
                    cv2.destroyAllWindows()
                    print("Quitting on user request.")
                    sys.exit(0)
                if decision == "y":
                    export_landmark_row(output_csv, results.pose_landmarks, label)
                    saved_count += 1
                elif decision == "f":
                    flipped = "W" if label == "C" else "C"
                    export_landmark_row(output_csv, results.pose_landmarks, flipped)
                    saved_count += 1
                else:
                    # skip (n)
                    pass
            else:
                export_landmark_row(output_csv, results.pose_landmarks, label)
                saved_count += 1
        except Exception:
            continue
    cap.release()
    return saved_count


def process_images(image_paths, output_csv="pushup_train.csv", label_source="folder", scale_percent=60, flip_horizontal=False, confirm_frames=False):
    """
    Process images (or folders of images) and export pose keypoints into CSV.

    image_paths: list of paths or folders containing images
    label_source: "folder" to use folder name -> label, or "filename" to infer from filename
    Returns total number of saved rows.
    """
    init_csv(output_csv)
    total_saved = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for path in image_paths:
            if os.path.isdir(path):
                for fname in sorted(os.listdir(path)):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        continue
                    img_file = os.path.join(path, fname)
                    if label_source == "folder":
                        label = infer_label_from_folder(os.path.basename(path))
                    else:
                        label = infer_label_from_filename(fname) or "C"
                    saved = process_single_image(img_file, pose, output_csv, label, scale_percent, flip_horizontal, confirm_frames)
                    total_saved += saved
                    print(f"Processed {img_file} -> saved: {saved} rows (label={label})")
            else:
                fname = os.path.basename(path)
                if label_source == "folder":
                    label = infer_label_from_folder(os.path.basename(os.path.dirname(path)))
                else:
                    label = infer_label_from_filename(fname) or "C"
                saved = process_single_image(path, pose, output_csv, label, scale_percent, flip_horizontal, confirm_frames)
                total_saved += saved
                print(f"Processed {path} -> saved: {saved} rows (label={label})")

    print(f"Total saved rows: {total_saved}")
    return total_saved


def process_single_image(image_path, pose, output_csv, label, scale_percent=60, flip_horizontal=False, confirm_frames=False):
    """Process a single image file and append a CSV row if pose was detected."""
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot open image:", image_path)
        return 0
    try:
        frame = rescale_frame(img, percent=scale_percent)
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return 0

        if confirm_frames:
            decision = _prompt_user_for_frame(frame, results.pose_landmarks, label)
            if decision == "q":
                cv2.destroyAllWindows()
                print("Quitting on user request.")
                sys.exit(0)
            if decision == "y":
                export_landmark_row(output_csv, results.pose_landmarks, label)
                return 1
            if decision == "f":
                flipped = "W" if label == "C" else "C"
                export_landmark_row(output_csv, results.pose_landmarks, flipped)
                return 1
            # else: skip
            return 0
        else:
            export_landmark_row(output_csv, results.pose_landmarks, label)
            return 1
    except Exception:
        return 0

# --- Command-line interface ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract pose keypoints from videos or images and export to CSV.")
    parser.add_argument("paths", nargs='+', help="One or more files or directories to process")
    parser.add_argument("--mode", choices=["videos", "images"], default="videos", help="Process videos or images (default: videos)")
    parser.add_argument("--output", default="./serious/pushup_train_khuari.csv", help="Output CSV path (default: ./pushup_train.csv)")
    parser.add_argument("--label-source", choices=["folder", "filename"], default="folder", help="Infer label from folder name or filename (default: folder)")
    parser.add_argument("--frame-step", type=int, default=5, help="Sample every Nth frame when processing videos (default: 5)")
    parser.add_argument("--scale-percent", type=int, default=60, help="Rescale frames/images by percent (default: 60)")
    parser.add_argument("--flip", action="store_true", help="Flip frames/images horizontally before processing")
    parser.add_argument("--confirm-frames", action="store_true", help="Interactively confirm each frame before saving (y/n/f/q)")

    args = parser.parse_args()

    # Expand possible relative paths to absolute for clarity
    paths = [os.path.abspath(p) for p in args.paths]

    if args.mode == "videos":
        process_videos(
            paths,
            output_csv=args.output,
            label_source=args.label_source,
            frame_step=args.frame_step,
            scale_percent=args.scale_percent,
            flip_horizontal=args.flip,
            confirm_frames=args.confirm_frames,
        )
    else:
        process_images(
            paths,
            output_csv=args.output,
            label_source=args.label_source,
            scale_percent=args.scale_percent,
            flip_horizontal=args.flip,
            confirm_frames=args.confirm_frames,
        )