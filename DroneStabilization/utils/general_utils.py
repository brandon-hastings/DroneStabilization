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

# TODO: add exposed option to not use gpu encoding
def ffmpeg_params(width: int | str, height: int | str, fps: int | float | str, output_filename: str) -> list:
    width, height, fps = str(width), str(height), str(fps)
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

    return ffmpeg_cmd