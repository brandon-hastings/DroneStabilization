from pathlib import Path
from utils.general_utils import check_resume
import os
import cv2
import csv
import numpy as np
from typing import List, Tuple, Optional


def _read_frames_at_indices(video_path: Path, indices: List[int], resize_to: Optional[Tuple[int, int]] = None):
    """
    Reads frames at (roughly) the given indices by streaming forward.
    Returns a list of BGR frames or None where a frame couldn't be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = [None] * len(indices)
    target_iter = iter(sorted(enumerate(indices), key=lambda x: x[1]))
    try:
        idx_pos, target_idx = next(target_iter)  # (slot_in_list, frame_number)
    except StopIteration:
        cap.release()
        return frames

    i = 0
    ok = True
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        if i == target_idx:
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            frames[idx_pos] = frame
            try:
                idx_pos, target_idx = next(target_iter)
            except StopIteration:
                break
        i += 1
    cap.release()
    return frames

def _laplacian_energy(gray: np.ndarray) -> np.ndarray:
    # Texture score: energy of Laplacian (edges/texture)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return lap * lap  # squared magnitude

def _box_mean(mat: np.ndarray, ksize: Tuple[int, int]) -> np.ndarray:
    # Fast sliding-window mean
    return cv2.boxFilter(mat, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)

def _global_align_pc(src_gray: np.ndarray, ref_gray: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Align src to ref by global translation using phase correlation (on full frame).
    Returns aligned image and (dx, dy) applied to src (src moved by dx,dy to match ref).
    """
    A = ref_gray.astype(np.float64)
    B = src_gray.astype(np.float64)
    # Hanning window to reduce edge effects
    win = cv2.createHanningWindow((A.shape[1], A.shape[0]), cv2.CV_64F)
    (shift_y, shift_x), response = cv2.phaseCorrelate(B * win, A * win)
    dx, dy = float(shift_x), float(shift_y)
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(src_gray, M, (src_gray.shape[1], src_gray.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned, (dx, dy)

def _residual_flow_mag(aligned_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    """
    Residual motion magnitude after global translation removal using dense optical flow.
    Lower magnitude => more stable region.
    """
    flow = cv2.calcOpticalFlowFarneback(ref_gray, aligned_gray,
                                        None, pyr_scale=0.5, levels=3, winsize=21,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2).astype(np.float32)
    return mag

def _nonmax_suppression_2d(score: np.ndarray, k: int, radius: int) -> List[Tuple[int, int, float]]:
    """
    Pick top-k points (y,x,score) with naive non-max suppression using a disk/square mask of given radius.
    """
    picked = []
    sc = score.copy()
    H, W = sc.shape
    for _ in range(k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sc)
        if max_val <= 0:
            break
        x, y = max_loc
        picked.append((y, x, float(max_val)))
        # Suppress neighborhood
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        sc[y0:y1, x0:x1] = 0.0
    return picked

def suggest_rois(
    video_path: Path,
    roi_size: Tuple[int, int] = (256, 256),
    k: int = 3,
    samples: int = 8,
    analysis_scale: float = 0.5,
    stride: Optional[int] = None,
    border_margin: int = 80,
    stability_weight: float = 1.0,
    texture_weight: float = 1.0,
    return_preview: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """
    Suggest top-k ROI rectangles for stabilization.

    Scoring = (normalized texture) / (1 + normalized residual-motion),
    where residual-motion is measured after removing global translation.

    Parameters
    ----------
    video_path : Path
        Path to the input video.
    roi_size : (w, h)
        Size of ROI in ORIGINAL resolution. (Will be scaled for analysis.)
    k : int
        Number of ROIs to return.
    samples : int
        Number of frames sampled across the video (including the first as reference).
    analysis_scale : float
        Downscale factor for analysis (0.3–0.7 recommended for 4K). ROIs map back to original size.
    stride : Optional[int]
        Step (in pixels at analysis scale) when scanning windows. Default: roi_w//4.
    border_margin : int
        Margin (in ORIGINAL pixels) to avoid near edges.
    stability_weight : float
        Weight of stability term (higher => prefer more stable regions).
    texture_weight : float
        Weight of texture term (higher => prefer more textured regions).
    return_preview : bool
        If True, returns a preview BGR image with candidate boxes.

    Returns
    -------
    rois : List[(x, y, w, h)]
        Top-k ROI rectangles in ORIGINAL resolution.
    preview_bgr : np.ndarray or None
        A preview image (first sampled frame) with boxes drawn (if return_preview).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if W <= 0 or H <= 0 or N <= 0:
        raise RuntimeError("Video metadata invalid (W/H/N).")

    # Downscale target for analysis
    scale = float(analysis_scale)
    aw, ah = max(1, int(W * scale)), max(1, int(H * scale))
    roi_w_orig, roi_h_orig = roi_size
    roi_w = max(8, int(roi_w_orig * scale))
    roi_h = max(8, int(roi_h_orig * scale))
    bmargin = int(border_margin * scale)

    # Sample frame indices (include first and last)
    idxs = np.linspace(0, max(0, N - 1), num=samples, dtype=int).tolist()
    # get ref frame index, round down in case of odd sample number
    ref_frame = idxs[int(samples/2)]
    frames = _read_frames_at_indices(video_path, idxs, resize_to=(aw, ah))
    # Convert to gray and check
    grays = []
    ref_gray = None
    ref_bgr_preview = None
    for i, f in enumerate(frames):
        if f is None:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        grays.append((idxs[i], g))
        if ref_gray is None:
            ref_gray = g
            ref_bgr_preview = f.copy()

    if ref_gray is None:
        raise RuntimeError("Could not decode any sampled frames.")

    # Texture map: mean Laplacian energy across samples
    tex_accum = np.zeros_like(ref_gray, dtype=np.float32)
    count_tex = 0
    for _, g in grays:
        tex_accum += _laplacian_energy(g)
        count_tex += 1
    tex_mean = tex_accum / max(1, count_tex)

    # Stability map: mean residual flow magnitude after global align
    stab_accum = np.zeros_like(ref_gray, dtype=np.float32)
    count_stab = 0
    for i, g in grays[1:]:  # compare to reference
        aligned, _ = _global_align_pc(g, ref_gray)
        mag = _residual_flow_mag(aligned, ref_gray)
        stab_accum += mag
        count_stab += 1
    if count_stab > 0:
        stab_mean = stab_accum / count_stab
    else:
        # If only one frame decoded, fake zero motion everywhere
        stab_mean = np.zeros_like(ref_gray, dtype=np.float32)

    # Convert to window-level maps using box mean over ROI size
    tex_win = _box_mean(tex_mean, (roi_w, roi_h))   # high is good
    stab_win = _box_mean(stab_mean, (roi_w, roi_h)) # low is good

    # Normalize maps (robust min-max with small epsilon)
    def normalize01(m):
        m = m.astype(np.float32)
        lo, hi = np.percentile(m, 5), np.percentile(m, 95)
        if hi <= lo:
            hi = lo + 1e-6
        out = np.clip((m - lo) / (hi - lo), 0, 1)
        return out

    tex_n = normalize01(tex_win)
    stab_n = normalize01(stab_win)

    # Combine: prefer high texture, low residual motion
    # Score = tex^tw / (1 + stab^sw)
    tw = float(texture_weight)
    sw = float(stability_weight)
    score = (np.power(tex_n, tw) / (1.0 + np.power(stab_n, sw))).astype(np.float32)

    # Border mask to avoid edges (set score=0 near borders)
    mask = np.zeros_like(score, dtype=np.float32)
    mask[bmargin:ah - bmargin, bmargin:aw - bmargin] = 1.0
    score *= mask

    # (Optional) reduce available positions to a grid via stride
    if stride is None:
        stride = max(8, min(roi_w, roi_h) // 4)
    # Reduce score by taking every 'stride' pixel to speed up NMS on large frames
    score_sparse = score.copy()

    # Non-max suppression over the score map
    # NMS radius ~ half ROI size at analysis scale
    nms_radius = int(0.5 * min(roi_w, roi_h))
    picks = _nonmax_suppression_2d(score_sparse, k=k, radius=nms_radius)

    rois: List[Tuple[int, int, int, int]] = []
    for (y_c, x_c, s) in picks:
        # Convert the window center (analysis scale) to top-left in ORIGINAL scale
        x0a = int(x_c - roi_w // 2)
        y0a = int(y_c - roi_h // 2)
        x0 = int(x0a / scale)
        y0 = int(y0a / scale)
        w0 = roi_w_orig
        h0 = roi_h_orig
        # Clamp to original frame bounds
        x0 = max(0, min(x0, W - w0))
        y0 = max(0, min(y0, H - h0))
        rois.append((x0, y0, w0, h0))

    # Build preview
    preview = None
    if return_preview and ref_bgr_preview is not None:
        preview = ref_bgr_preview.copy()
        # Scale ROIs to preview size
        color_list = [(0, 255, 0), (0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 128, 255)]
        for i, (x0, y0, w0, h0) in enumerate(rois):
            x1p = int(x0 * scale)
            y1p = int(y0 * scale)
            w1p = int(w0 * scale)
            h1p = int(h0 * scale)
            c = color_list[i % len(color_list)]
            cv2.rectangle(preview, (x1p, y1p), (x1p + w1p, y1p + h1p), c, 2)
            cv2.putText(preview, f"ROI {i+1}", (x1p + 5, y1p + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)


    return rois, ref_frame, preview


def batch_suggest_rois(video_folder: Path, output_csv: Path):
    # store settings
    video_folder = video_folder
    output_csv = output_csv

    # list of images to process
    video_list = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith((".mov", ".mp4"))
        ]

    video_list = check_resume(video_list, output_csv)

    if len(video_list) == 0:
        raise ValueError("No videos found in folder!")
    
    labels = {}
    
    for video in video_list:
        rois, ref_frame, _ = suggest_rois(
            video_path=video,
            roi_size=(256, 256),
            k=3,
            samples=100,              # at least 6
            analysis_scale=0.5,     # 0.4–0.6 is a good range for 4K
            border_margin=300,      # avoid edges
            stability_weight=1.0,
            texture_weight=1.0,
            return_preview=False
        )
        # join rois and ref_frame
        rois_as_list = list(rois[0])
        rois_as_list.append(ref_frame)
        labels[video] = rois_as_list

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    # was mode "w", changed to "a" to support return to labelling
    with output_csv.open(mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "x", "y", "w", "h", "ref_frame"])
        for vid, box in labels.items():
            writer.writerow([vid] + box)

    print(f"Saved labels to {output_csv}")

    


if __name__ == "__main__":

    pass
    # batch_suggest_rois(video_folder, output_csv)


    # # Suggest top-3 ROIs of size 256x256 (in original resolution)
    # rois, ref_frame, preview = suggest_rois(
    #     video_path=video,
    #     roi_size=(256, 256),
    #     k=3,
    #     samples=100,              # increase to 10–12 for longer clips if you want
    #     analysis_scale=0.5,     # 0.4–0.6 is a good range for 4K
    #     border_margin=120,      # avoid edges
    #     stability_weight=1.0,
    #     texture_weight=1.0,
    #     return_preview=True
    # )

    # print("Suggested ROIs:", rois)  # e.g., [(x,y,w,h), ...]
    # print("Reference frame:", ref_frame)
    # if preview is not None:
    #     cv2.imwrite("roi_candidates_preview.jpg", preview)

    # # Pick the best ROI (first one) to feed your stabilizer:
    # best_roi = rois[0] if rois else (0, 0, 256, 256)