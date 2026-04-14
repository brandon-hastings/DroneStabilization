import pandas as pd
from pathlib import Path


def check_resume(videos_list, csv_file):
    csv_file = Path(csv_file)
    # if csv exist, get videos already done and subtract from video list
    if csv_file.exists():
        df = pd.read_csv(csv_file, header=0)
        processed_videos = df["video"].to_list()
        # remove already processed videos
        todo_videos = list(set(videos_list) - set(processed_videos))
        if len(todo_videos) == 0:
            raise IndexError("No videos left to label after comparision with previous maks_labels.csv file")
        return todo_videos
    else:
        return videos_list
    
def str_to_bool(s):
    """
    Convert a string to a boolean value. Raises ValueError for invalid inputs.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")

    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}

    s_lower = s.strip().lower()

    if s_lower in truthy:
        return True
    elif s_lower in falsy:
        return False
    else:
        raise ValueError(f"Cannot convert '{s}' to boolean.")

# use_gpu is currently exposed in the config file only
def ffmpeg_params(width: int | str, height: int | str, fps: int | float | str, output_filename: str, use_gpu: bool) -> list:
    width, height, fps = str(width), str(height), str(fps)
    if use_gpu:
        ffmpeg_cmd = [
            'ffmpeg',
            '-y', #overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            "-pix_fmt", "bgr24",
            '-s', f'{width}x{height}',
            '-r', fps,
            '-i', '-',
            # NVENC encoding
            '-c:v', 'h264_nvenc',
            '-rc', 'vbr_hq',
            '-b:v', '50M',
            '-maxrate', '50M',
            '-pix_fmt', 'yuv420p',
            output_filename
        ]
    else:
        ffmpeg_cmd = [
            'ffmpeg',
            '-y', #overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            "-pix_fmt", "bgr24",
            '-s', f'{width}x{height}',
            '-r', fps,
            '-i', '-',
            # Encoding
            '-c:v', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            output_filename
        ]

    return ffmpeg_cmd