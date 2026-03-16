import cv2
import numpy as np
import pandas as pd
import csv
from pathlib import Path
import math

def frame_ripper(video_path):
    '''
    Docstring for frame_ripper
    
    :param video_path: path to video as Pathlib obj

    :retval frame: middle frame of video for creating a bounding box
    '''
    # where video is expected to be the filename of a video
    source = cv2.VideoCapture(str(video_path))
    frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = int(frames/2)
    source.set(cv2.CAP_PROP_POS_FRAMES, middle_frame-1)
    res, frame = source.read()
    if not res:
        IndexError("No frame found")
    else:
        return (frame, middle_frame - 1)
    
#TODO: could use response to switch to sift ROI tracking as a form of enhanced interpolation
def video_stabilization(
    video_path: Path,
    roi_coords: tuple,                    # (x, y, w, h)
    reference_frame_index: int,
    shifts_csv: Path,
    output_path: Path | None = None,
    default_fps: float = 25.0,
    ensure_even_mp4: bool = True,
    prefer_codecs: tuple = ("mp4v", "avc1", "XVID"),
    log_every: int = 100,
    estimation_scale: float | None = None,  # used in translation path (phase correlation)
    *,
    # >>> NEW: registration method & params
    method: str = "translation",           # "translation" or "affine"
    sift_nfeatures: int = 2000,            # SIFT/ORB: number of features to keep
    ratio_thresh: float = 0.75,            # Lowe's ratio for SIFT matching
    min_matches: int = 12,                 # minimum matches to attempt affine
    ransac_thresh: float = 3.0,            # RANSAC reprojection threshold (px)
    use_partial_affine: bool = True        # True -> estimateAffinePartial2D (no shear), False -> estimateAffine2D
):
    """
    Streamed stabilization using either:
      - translation (phase correlation on ROI), or
      - affine (SIFT feature matching in ROI + RANSAC affine).

    Writes a stabilized video and a CSV:
      - translation: (frame, dx, dy)
      - affine: (frame, a11, a12, a13, a21, a22, a23, inliers, matched, dx, dy)
    """

    # ------------------
    # Prepare paths
    # ------------------
    video_path = Path(video_path)
    assert video_path.exists(), f"Input video not found: {video_path}"

    if output_path is None:
        output_path = video_path.parents[2] / "results" / "stabilized_videos" / f"{video_path.stem}_stabilized{video_path.suffix}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------
    # Open, read metadata
    # ------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = float(default_fps)

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Source opened OK: {video_path}")
    print(f"[INFO] W={W}, H={H}, N={N}, FPS={fps:.3f}")

    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError("Video reports zero width/height—cannot proceed.")

    # -----------------------
    # PASS 1: get ref frame
    # -----------------------
    ref_bgr = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx == reference_frame_index:
            ref_bgr = frame.copy()
            break
        idx += 1
    cap.release()

    if ref_bgr is None:
        # Clamp and try again if index out of range
        reference_frame_index = max(0, min(N - 1, reference_frame_index))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Re-open failed: {video_path}")
        for i in range(reference_frame_index + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError("Unable to read reference frame after clamping.")
        ref_bgr = frame.copy()
        cap.release()

    # ROI clamp
    x, y, w, h = map(int, roi_coords)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_roi = ref_gray[y:y+h, x:x+w]
    if ref_roi.size == 0:
        raise ValueError("ROI is empty after clamping; check roi_coords.")

    # -----------------------
    # Prepare writer (with fallbacks and even-dim padding if MP4)
    # -----------------------
    is_mp4_target = output_path.suffix.lower() == ".mp4"
    outW, outH = W, H
    pad_right = pad_bottom = 0
    if is_mp4_target and ensure_even_mp4:
        if outW % 2 != 0:
            pad_right = 1
            outW += 1
        if outH % 2 != 0:
            pad_bottom = 1
            outH += 1
        if pad_right or pad_bottom:
            print(f"[INFO] Padding for MP4 even dims: ({W},{H}) -> ({outW},{outH})")

    def try_open_writer(path: Path, fourcc_tag: str, width: int, height: int, fps_val: float):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
        wr = cv2.VideoWriter(str(path), fourcc, fps_val, (width, height))
        return wr

    attempts = []
    for tag in prefer_codecs:
        if tag in ("mp4v", "avc1"):
            attempts.append((tag, output_path.with_suffix(".mp4"), outW, outH))
        elif tag == "XVID":
            attempts.append((tag, output_path.with_suffix(".avi"), W, H))
        else:
            attempts.append((tag, output_path.with_suffix(".avi"), W, H))

    writer = None
    chosen = None
    for tag, path_try, w_try, h_try in attempts:
        wr = try_open_writer(path_try, tag, w_try, h_try, fps)
        print(f"[INFO] Trying writer {tag} -> {path_try} @ {w_try}x{h_try} … opened={wr.isOpened()}")
        if wr.isOpened():
            writer = wr
            chosen = (tag, path_try, w_try, h_try)
            break
        else:
            try: wr.release()
            except: pass

    if writer is None:
        raise RuntimeError("Failed to open VideoWriter (mp4v/avc1/XVID). Try AVI/MJPG or install FFmpeg-enabled OpenCV.")

    fourcc_tag, out_path, writeW, writeH = chosen
    print(f"[INFO] Using writer: {fourcc_tag} -> {out_path} ({writeW}x{writeH} @ {fps:.3f} fps)")

    # -----------------------
    # Precompute data per method
    # -----------------------
    # Translation path precompute (phase correlation with optional downscale)
    if method == "translation":
        scale = 1.0
        if estimation_scale and 0 < estimation_scale < 1.0:
            scale = float(estimation_scale)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            ref_roi_small = cv2.resize(ref_roi, new_size, interpolation=cv2.INTER_AREA)
            window = cv2.createHanningWindow((ref_roi_small.shape[1], ref_roi_small.shape[0]), cv2.CV_64F)
            refA = ref_roi_small.astype(np.float64) * window
        else:
            window = cv2.createHanningWindow((ref_roi.shape[1], ref_roi.shape[0]), cv2.CV_64F)
            refA = ref_roi.astype(np.float64) * window

    # Affine path precompute (SIFT/ORB features on reference ROI)
    elif method == "affine":
        # Create feature extractor (SIFT preferred; fallback to ORB if unavailable)
        sift = None
        bf_norm = None
        if hasattr(cv2, "SIFT_create"):
            sift = cv2.SIFT_create(nfeatures=int(sift_nfeatures))
            bf_norm = cv2.NORM_L2
            feat_name = "SIFT"
        else:
            # Fallback to ORB
            sift = cv2.ORB_create(nfeatures=int(sift_nfeatures))
            bf_norm = cv2.NORM_HAMMING
            feat_name = "ORB"
        print(f"[INFO] Affine mode using {feat_name} (nfeatures={sift_nfeatures})")

        # Detect/describe in reference ROI
        kp_ref, des_ref = sift.detectAndCompute(ref_roi, None)
        if des_ref is None or len(kp_ref) < min_matches:
            raise RuntimeError(f"Not enough reference features: {len(kp_ref)}; try a more textured ROI or lower min_matches.")

        # Convert keypoints to full-image coordinates (add ROI offset)
        def kp_to_xy(kps):
            pts = np.float32([kp.pt for kp in kps])  # (x_roi, y_roi)
            pts[:, 0] += x
            pts[:, 1] += y
            return pts

        pts_ref_full = kp_to_xy(kp_ref)
        # BF matcher (we'll use knnMatch + ratio test)
        bf = cv2.BFMatcher(bf_norm, crossCheck=False)

    else:
        raise ValueError("method must be 'translation' or 'affine'")

    # -----------------------
    # PASS 2: stream frames, compute transform per frame, write video & CSV
    # -----------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to re-open video: {video_path}")

    # Prepare CSV (stream write)
    shifts_csv = Path(shifts_csv)
    shifts_csv.parent.mkdir(exist_ok=True, parents=True)
    fcsv = shifts_csv.open("w", newline="")
    wcsv = csv.writer(fcsv)
    if method == "translation":
        wcsv.writerow(("frame", "dx", "dy"))
    else:
        wcsv.writerow(("frame", "a11", "a12", "a13", "a21", "a22", "a23", "inliers", "matched", "dx", "dy"))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mov_roi = gray[y:y+h, x:x+w]

        if method == "translation":
            if estimation_scale and 0 < estimation_scale < 1.0:
                mov_roi_small = cv2.resize(mov_roi, (refA.shape[1], refA.shape[0]), interpolation=cv2.INTER_AREA)
                B = mov_roi_small.astype(np.float64) * window
            else:
                B = mov_roi.astype(np.float64) * window

            # Phase correlation returns (dy, dx)
            (shift_y, shift_x), response = cv2.phaseCorrelate(B, refA)
            dx = float(shift_x) / (float(estimation_scale) if estimation_scale and 0 < estimation_scale < 1.0 else 1.0)
            dy = float(shift_y) / (float(estimation_scale) if estimation_scale and 0 < estimation_scale < 1.0 else 1.0)

            # Build transform matrix and warp
            M = np.array([[1, 0, dx],
                          [0, 1, dy]], dtype=np.float32)

            stabilized = cv2.warpAffine(frame, M, (W, H),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)

            # Padding for MP4 even dims (if needed)
            if fourcc_tag in ("mp4v", "avc1") and (pad_right or pad_bottom):
                stabilized = cv2.copyMakeBorder(stabilized, 0, pad_bottom, 0, pad_right,
                                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            writer.write(stabilized)
            wcsv.writerow((frame_idx, dx, dy))

        else:
            # --- Affine path via SIFT/ORB & RANSAC ---
            kp_mov, des_mov = sift.detectAndCompute(mov_roi, None)
            matched = 0
            inliers = 0
            M = None

            if des_mov is not None and len(kp_mov) >= min_matches:
                # Convert moving kps to full-image coords
                pts_mov_roi = np.float32([kp.pt for kp in kp_mov])
                pts_mov_full = pts_mov_roi.copy()
                pts_mov_full[:, 0] += x
                pts_mov_full[:, 1] += y

                # KNN match
                knn = bf.knnMatch(des_ref, des_mov, k=2)  # ref -> mov
                good = []
                for m, n in knn:
                    if m.distance < ratio_thresh * n.distance:
                        good.append(m)

                matched = len(good)
                if matched >= min_matches:
                    # Build correspondence arrays (moving -> reference)
                    src = np.float32([pts_mov_full[m.trainIdx] for m in good])  # moving points
                    dst = np.float32([pts_ref_full[m.queryIdx] for m in good])  # reference points

                    # Estimate affine
                    if use_partial_affine:
                        M, inlier_mask = cv2.estimateAffinePartial2D(
                            src, dst, method=cv2.RANSAC,
                            ransacReprojThreshold=float(ransac_thresh),
                            maxIters=2000, confidence=0.99, refineIters=10
                        )
                    else:
                        M, inlier_mask = cv2.estimateAffine2D(
                            src, dst, method=cv2.RANSAC,
                            ransacReprojThreshold=float(ransac_thresh),
                            maxIters=2000, confidence=0.99, refineIters=10
                        )
                    if inlier_mask is not None:
                        inliers = int(inlier_mask.sum())
                else:
                    # Not enough good matches
                    M = None

            # Fallback if no transform obtained: try translation on this frame
            if M is None:
                # (Small, robust fallback)
                win = cv2.createHanningWindow((ref_roi.shape[1], ref_roi.shape[0]), cv2.CV_64F)
                A = ref_roi.astype(np.float64) * win
                B = mov_roi.astype(np.float64) * win
                (shift_y, shift_x), _ = cv2.phaseCorrelate(B, A)
                dx = float(shift_x)
                dy = float(shift_y)
                M = np.array([[1, 0, dx],
                              [0, 1, dy]], dtype=np.float32)

            # Apply affine/translation matrix
            stabilized = cv2.warpAffine(frame, M, (W, H),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)

            if fourcc_tag in ("mp4v", "avc1") and (pad_right or pad_bottom):
                stabilized = cv2.copyMakeBorder(stabilized, 0, pad_bottom, 0, pad_right,
                                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            writer.write(stabilized)

            # Log affine params (a11 a12 a13; a21 a22 a23) + dx dy from matrix
            a11, a12, a13 = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
            a21, a22, a23 = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
            wcsv.writerow((frame_idx, a11, a12, a13, a21, a22, a23, inliers, matched, a13, a23))

        if (frame_idx % log_every) == 0:
            if method == "translation":
                print(f"[INFO] Frame {frame_idx}/{N}  (translation)")
            else:
                print(f"[INFO] Frame {frame_idx}/{N}  (affine)")

        frame_idx += 1

    # Cleanup
    writer.release()
    cap.release()
    fcsv.close()

    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    print(f"[INFO] Done. Wrote: {out_path}  bytes={size_bytes}")
    print(f"[INFO] CSV: {shifts_csv}")

    return out_path


def batch_stabilize(video_folder, mask_csv, shifts_csv):
    video_folder = Path(video_folder)
    video_list = [x for x in video_folder.iterdir()]
    masked_df = pd.read_csv(Path(mask_csv))
    for vid in video_list:
        print(vid)
        row = tuple(masked_df.loc[masked_df['video'] == str(vid)].iloc[0])
        # get roi coords
        roi_coords = row[1:5]
        #get ref frame
        ref_frame = row[5]
        # name individual shifts file
        vid_stem = Path(vid).stem
        shifts_csv = Path(shifts_csv)
        shifts_stem = shifts_csv.stem
        new_name = "_".join((vid_stem, shifts_stem))
        renamed_shifts_csv = shifts_csv.with_stem(new_name)
        # call function
        video_stabilization(vid, roi_coords, ref_frame, renamed_shifts_csv)
    
    