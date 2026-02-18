import cv2
import numpy as np
import pandas as pd
import csv
from pathlib import Path

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
    
def video_stabilization(video_path: Path, roi_coords: tuple, reference_frame_index: int, shifts_csv: Path):
    # --- Config ---
    org_name = video_path.stem
    new_name = "_".join((org_name, "stabilized"))
    stabilized_video = video_path.with_stem(new_name)
    print(stabilized_video)
    mode = "translation"  # or "affine"
    use_phase_correlation = True  # else template matching

    # --- Load video ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Read reference frame ---
    def read_frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        return frame if ret else None

    ref_bgr = read_frame(reference_frame_index)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    # --- Define ROI/mask ---
    # stored as x1,y1,w,h
    x, y, w, h = roi_coords
    ref_roi = ref_gray[y:y+h, x:x+w]


    # Optional windowing to reduce edge effects
    window = cv2.createHanningWindow((ref_roi.shape[1], ref_roi.shape[0]), cv2.CV_64F)

    # --- Prepare writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(stabilized_video), fourcc, fps, (W, H))

    # --- Logging ---
    rows = [("frame", "dx", "dy")]  # extend with affine fields if needed

    # --- Process all frames ---
    for i in range(N):
        frame = read_frame(i)
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mov_roi = gray[y:y+h, x:x+w]

        # --- Estimate motion ---
        if use_phase_correlation:
            # Convert to float64 as required by phaseCorrelate
            A = ref_roi.astype(np.float64)
            B = mov_roi.astype(np.float64)
            # Apply window
            Aw = A * window
            Bw = B * window
            (shift_y, shift_x), response = cv2.phaseCorrelate(Bw, Aw)  # (dy, dx)
            dx, dy = float(shift_x), float(shift_y)
        else:
            res = cv2.matchTemplate(mov_roi, ref_roi, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
            # Template top-left in mov that best matches ref
            # Convert to dx, dy relative to ROI anchor (0,0)
            dx = (x + maxLoc[0]) - x
            dy = (y + maxLoc[1]) - y

        # --- Apply transform to original full frame ---
        M = np.array([[1, 0, dx],
                    [0, 1, dy]], dtype=np.float32)
        stabilized = cv2.warpAffine(frame, M, (W, H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)

        writer.write(stabilized)
        rows.append((i, dx, dy))

    # --- Cleanup & save CSV ---
    writer.release()
    cap.release()

    shifts_csv.parent.mkdir(exist_ok=True, parents=True)
    with shifts_csv.open(mode="w", newline="") as f:
        csv.writer(f).writerows(rows)


def batch_stabilize(video_folder, mask_csv, shifts_csv):
    video_folder = Path(video_folder)
    video_list = [x for x in video_folder.iterdir()]
    masked_df = pd.read_csv(Path(mask_csv))
    for vid in video_list:
        print(vid)
        row = tuple(masked_df.loc[masked_df['video'] == str(vid)].iloc[0])
        h = abs(row[4] - row[2])
        w = abs(row[3] - row[1])
        roi_coords = (row[1], row[2], w, h)
        #get ref frame
        ref_frame = row[5]
        #name individual shifts file
        vid_stem = Path(vid).stem
        shifts_csv = Path(shifts_csv)
        shifts_stem = shifts_csv.stem
        new_name = "_".join((vid_stem, shifts_stem))
        renamed_shifts_csv = shifts_csv.with_stem(new_name)
        # call function
        video_stabilization(vid, roi_coords, ref_frame, renamed_shifts_csv)
    
    