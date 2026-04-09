import pysrt
import sys
import re
import pandas as pd

def parse_camera_parameters(sub):
    params_dict = {}
    camera_specs = re.findall(r'\[.*?\]', sub.text)
    for i in camera_specs[:-1]:
        key, value = i.split(": ")
        params_dict[key[1:]] = value[:-1]
    # split last altitude column
    alt_data = camera_specs[-1].split(" ")
    params_dict[alt_data[0][1:-1]] = alt_data[1]
    params_dict[alt_data[2][:-1]] = alt_data[3][:-1]
    return params_dict

def parse_srt(srt_file):
    srt_subs = pysrt.open(srt_file)
    video_length = srt_subs[-1].end
    date = re.search(r'(\d{4})-(\d{2})-(\d{2})', srt_subs[0].text)
    start_time = re.search(r'(\d{2}):(\d{2}):(\d{2}).(\d{3})', srt_subs[0].text)
    df = None
    for i in range(0, len(srt_subs)):
        cam_params = parse_camera_parameters(srt_subs[i])
        if i == 0:
            df = pd.DataFrame(cam_params, index=[0])
        else:
            df.loc[len(df)] = cam_params
    
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
    print(dfn.describe)

if __name__ == "__main__":
    parse_srt(sys.argv[1])
