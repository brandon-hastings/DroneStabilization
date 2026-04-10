import pysrt
import sys
import re
import pandas as pd
import os
from pathlib import Path

def parse_camera_parameters(sub):
    params_dict = {}
    time = re.search(r'(\d{2}):(\d{2}):(\d{2}).(\d{3})', sub.text).group(0)
    camera_specs = re.findall(r'\[.*?\]', sub.text)
    for i in camera_specs[:-1]:
        key, value = i.split(": ")
        params_dict[key[1:]] = value[:-1]
    # split last altitude column
    alt_data = camera_specs[-1].split(" ")
    params_dict[alt_data[0][1:-1]] = alt_data[1]
    params_dict[alt_data[2][:-1]] = alt_data[3][:-1]
    # add frame time
    params_dict["frame_time"] = time
    return params_dict

def parse_srt(srt_file):
    srt_subs = pysrt.open(srt_file)
    video_length = str(srt_subs[-1].end)
    date = re.search(r'(\d{4})-(\d{2})-(\d{2})', srt_subs[0].text).group(0)
    df = None
    for i in range(0, len(srt_subs)):
        cam_params = parse_camera_parameters(srt_subs[i])
        if i == 0:
            df = pd.DataFrame(cam_params, index=[0])
        else:
            df.loc[len(df)] = cam_params
    df["video_length"], df["date"], = video_length, date
    df
    
    column_type_mapping = {
        "iso": "int32",
        "shutter": "string",
        "fnum": "float32",
        "ev": "int32",
        "ct": "int32",
        "color_md ": "string",
        "focal_len": "float32",
        "latitude": "float32",
        "longitude": "float32", 
        "rel_alt": "float32",
        "abs_alt": "float32"
    }
    dfn = df.astype(column_type_mapping)
    return dfn

def batch_parse_srt(folder_path):
    full_df = None
    for file in os.listdir(folder_path):
        file_name = os.path.normpath(file)
        if str(file_name).endswith(".SRT"):
            print(file_name)
            df = parse_srt(os.path.join(folder_path, file_name))
            df["filename"] = str(file_name)
            print(df.head)
            if full_df is None:
                full_df = df
            elif type(full_df) == pd.DataFrame:
                full_df = pd.concat([full_df, df])
    return full_df


if __name__ == "__main__":
    experiment_metadata = batch_parse_srt(sys.argv[1])
    experiment_metadata.to_csv(Path("/Users/brandonhastings/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Gibbs_lab/2026/Drone_data/metadata/exp_metadata.csv"))
