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
    estimation_scale: float | None = None,  # e.g., 0.5 to downscale ROI for estimation
):
    """
    Streamed, two-pass translation stabilization using phase correlation on a user ROI.
    Pass 1: read reference frame only (sequential).
    Pass 2: stream frames, compute (dx, dy), warp, and write each frame immediately.
    """

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

    # Optional downscale for estimation (faster on large ROIs)
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
    # PASS 2: stream frames, compute dx/dy, write video & CSV
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
    wcsv.writerow(("frame", "dx", "dy"))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mov_roi = gray[y:y+h, x:x+w]

        if scale != 1.0:
            mov_roi_small = cv2.resize(mov_roi, (refA.shape[1], refA.shape[0]), interpolation=cv2.INTER_AREA)
            B = mov_roi_small.astype(np.float64) * window
        else:
            B = mov_roi.astype(np.float64) * window

        # Phase correlation returns (dy, dx)
        (shift_y, shift_x), response = cv2.phaseCorrelate(B, refA)
        dx = float(shift_x) / scale
        dy = float(shift_y) / scale

        # Apply translation to full frame
        M = np.array([[1, 0, dx],
                      [0, 1, dy]], dtype=np.float32)
        stabilized = cv2.warpAffine(frame, M, (W, H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)

        if fourcc_tag in ("mp4v", "avc1") and (pad_right or pad_bottom):
            stabilized = cv2.copyMakeBorder(stabilized, 0, pad_bottom, 0, pad_right,
                                            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        writer.write(stabilized)
        wcsv.writerow((frame_idx, dx, dy))

        if (frame_idx % log_every) == 0:
            print(f"[INFO] Frame {frame_idx}/{N}  dx={dx:.3f}  dy={dy:.3f}")

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
    
    