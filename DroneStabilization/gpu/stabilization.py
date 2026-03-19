# stabilization_main.py

import cv2
import numpy as np
import csv
from pathlib import Path
import math

from detect_features import GPUFeatureExtractor
from affine_cpu import estimate_affine_cpu
from gpu_transforms import GPUWarpAffine

def video_stabilization_multi_gpu(
    video_path: Path,
    roi_coords: tuple,
    reference_frame_index: int,
    shifts_csv: Path,
    output_path: Path | None = None,
    *,
    method="affine-gpu",
    gpu_sift_id=0,         # GPU0 → SIFT & matching
    gpu_warp_id=1,         # GPU1 → warpAffine
    sift_nfeatures=2000,
    ratio_thresh=0.75,
    min_matches=12,
    ransac_thresh=3.0,
    partial_affine=True,
    prefer_codecs=("mp4v", "avc1", "XVID"),
    default_fps=25.0,
    log_every=100,
):
    """
    Full GPU-accelerated affine stabilization using two GPUs.
    """

    video_path = Path(video_path)
    assert video_path.exists()

    # -------------------
    # VIDEO METADATA
    # -------------------
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -------------------
    # REFERENCE FRAME
    # -------------------
    ref_frame = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_index)
    ok, ref_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read reference frame")

    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    x, y, w, h = map(int, roi_coords)
    ref_roi = ref_gray[y:y+h, x:x+w]
    if ref_roi.size == 0:
        raise RuntimeError("ROI invalid")

    # -------------------
    # GPU MODULES
    # -------------------
    fe = GPUFeatureExtractor(device_id=gpu_sift_id, nfeatures=sift_nfeatures)
    warper = GPUWarpAffine(device_id=gpu_warp_id)

    print(f"[INFO] Using GPUs: SIFT={gpu_sift_id}, warpAffine={gpu_warp_id}")

    # Extract features from reference ROI (GPU0)
    kp_ref, des_ref = fe.extract(ref_roi)
    if des_ref is None or len(kp_ref) < min_matches:
        raise RuntimeError("Not enough features in reference ROI")

    # Convert ref keypoints to FULL IMAGE coordinates
    pts_ref_full = np.float32([ (kp.pt[0] + x, kp.pt[1] + y) for kp in kp_ref ])

    # -------------------
    # VIDEO WRITER
    # -------------------
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_stabilized.mp4"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Failed to open writer")

    # -------------------
    # CSV OUTPUT
    # -------------------
    shifts_csv = Path(shifts_csv)
    shifts_csv.parent.mkdir(parents=True, exist_ok=True)
    fcsv = shifts_csv.open("w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["frame", "a11","a12","a13","a21","a22","a23","inliers","matched"])

    # -------------------
    # PROCESS FRAMES (STREAMING)
    # -------------------
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mov_roi = gray[y:y+h, x:x+w]

        # GPU0: SIFT
        kp_mov, des_mov = fe.extract(mov_roi)
        M = None
        matched = 0
        inliers = 0

        if des_mov is not None and len(kp_mov) >= min_matches:
            # Full image coords
            pts_mov_full = np.float32([ (kp.pt[0] + x, kp.pt[1] + y) for kp in kp_mov ])

            knn = fe.match(des_ref, des_mov, k=2)
            good = []
            for m, n in knn:
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

            matched = len(good)

            if matched >= min_matches:
                src = np.float32([ pts_mov_full[m.trainIdx] for m in good ])
                dst = np.float32([ pts_ref_full[m.queryIdx] for m in good ])

                M, inlier_mask = estimate_affine_cpu(src, dst, ransac_thresh, partial_affine)
                if inlier_mask is not None:
                    inliers = int(inlier_mask.sum())

        # Fallback to translation
        if M is None:
            # Use phase correlation as fallback
            win = cv2.createHanningWindow((w, h), cv2.CV_64F)
            A = ref_roi.astype(np.float64) * win
            B = mov_roi.astype(np.float64) * win
            (shift_y, shift_x), _ = cv2.phaseCorrelate(B, A)
            M = np.array([[1,0,shift_x],[0,1,shift_y]],dtype=np.float32)

        # GPU1: warpAffine
        stabilized = warper.warp(frame, M, W, H)

        writer.write(stabilized)

        # Log matrix
        a11, a12, a13 = float(M[0,0]), float(M[0,1]), float(M[0,2])
        a21, a22, a23 = float(M[1,0]), float(M[1,1]), float(M[1,2])
        wcsv.writerow([frame_idx, a11,a12,a13,a21,a22,a23, inliers, matched])

        if frame_idx % log_every == 0:
            print(f"[INFO] Frame {frame_idx}/{N}: M={M.flatten()}")

        frame_idx += 1

    cap.release()
    writer.release()
    fcsv.close()

    print(f"[INFO] Complete. Output video: {out_path}")
    return out_path