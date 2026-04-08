import pandas as pd
import numpy as np
import os
from pathlib import Path

# import each shifts file as a pandas df
def summarize_shifts(folder_path):
    flags = []
    for file in os.listdir(folder_path):
        file_name = os.path.normpath(file)
        if str(file_name).endswith("dxdy_shifts.csv"):
            # first order differencing on dx and dy shifts columns (last two)
            df = pd.read_csv(file_name)
            dx = df.iloc[:,-2]
            dy = df.iloc[:,-1]
            dx_diff = dx.diff()
            dy_diff = dy.diff()
            # check if variance is greater than 1 in either domain
            if abs(dx_diff) > 1 or abs(dy_diff) > 1:
                score = 1
            else:
                score = 0
            flags.append((file_name, score))
    


            
        



# calculate variance from mean for the full time series
# store the value of dx var and dy var in new dataframe (or maybe in csv masks)
# use as metric to assess stability using threshold
# eventually use simple supervised classification, with inliers metric as well

