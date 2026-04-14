import pandas as pd
import os
import sys
from pathlib import Path


def analyse_shifts(dataframe_obj):
    df = dataframe_obj
    print(df.shape)
    df_new = df.loc[(df['dx_diff_std'] < 1) & ( df['dy_diff_std'] < 1)]
    print(df_new.shape)


# import each shifts file as a pandas df
def summarize_shifts(folder_path):
    flags = []
    for file in os.listdir(folder_path):
        file_name = os.path.normpath(file)
        if str(file_name).endswith("dxdy_shifts.csv"):
            # first order differencing on dx and dy shifts columns (last two)
            df = pd.read_csv(os.path.join(folder_path, file_name))
            print(f"Created dataframe from {file_name}")
            dx = df.iloc[:,-2]
            dy = df.iloc[:,-1]
            dx_diff = dx.diff()
            dy_diff = dy.diff()
            # check if variance is greater than 1 in either domain
            # if (max(dx_diff) - min(dx_diff)) > 2:
            #     score_x = 1
            # else:
            #     score_x = 0
            # if (max(dy_diff) - min(dy_diff)) > 2:
            #     score_y = 1
            # else:
            #     score_y = 0

            # put variance in column
            dx_diff_variance = dx_diff.max() - dx_diff.min()
            dy_diff_variance = dy_diff.max() - dy_diff.min()

            # get standard deviation of diffs
            dx_diff_std = dx_diff.std()
            dy_diff_std = dy_diff.std()

            flags.append((os.path.join(folder_path, file_name), dx_diff_std, dy_diff_std, dx_diff_variance, dy_diff_variance))
    df = pd.DataFrame(flags, columns=['filename', 'dx_diff_std', 'dy_diff_std', 'dx_diff_variance', 'dy_diff_variance'])
    analyse_shifts(df)
    outpath =  Path(folder_path) / 'summary.csv'
    df.to_csv(outpath)
    print(f'Saved to: {outpath}')

# calculate variance from mean for the full time series
# store the value of dx var and dy var in new dataframe (or maybe in csv masks)
# use as metric to assess stability using threshold
# eventually use simple supervised classification, with inliers metric as well

if __name__ == "__main__":
    summarize_shifts(sys.argv[1])
