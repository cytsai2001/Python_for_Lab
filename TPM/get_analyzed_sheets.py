'/home/hwlis/data_CYT/CYT/20221123/slide1/3-1-1162bp-PS-150mMKCl-50nＭRec10'

from TPM.BinaryImage import BinaryImage
from TPM.DataToSave import DataToSave
from TPM.localization import select_folder
import time
import pandas as pd
import numpy as np
import os
from glob import glob
import psutil


analyzed_mode = 'all'  ## if analyzed_mode = 'all', analyze all frames of .csv file
criteria_mode = 'PS'   ## if criteria_mode = 'QD', criteria will include sx_sy; else, criteria will only consider BM and ratio.
frame_n = 100
frame_start = 0
BM_lower = 10
BM_upper = 130
ratio_lower = 0.8
ratio_upper = 1.2
sx_sy_lower = 5
sx_sy_upper = 30

def get_analyzed_sheet(path_folder, analyzed_mode, criteria_mode, frame_n,
                       BM_lower, BM_upper, ratio_lower, ratio_upper, sx_sy_lower, sx_sy_upper):
    path_data = glob(os.path.join(path_folder, '*-fitresults.csv'))[0]
    t1 = time.time()
    Glimpse_data = BinaryImage(path_folder)
    if analyzed_mode == 'all':
        frame_n = Glimpse_data.frames_acquired
    print('read_csv start')
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    df = pd.read_csv(path_data)
    print('read_csv end')
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    bead_number = int(max(1 + df['aoi']))
    tracking_results = np.array(df)
    tracking_results = tracking_results[0:bead_number*frame_n, :]
    localization_results = np.zeros((bead_number, 1))
    print('construct Save_df start')
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    # construction of Save_df is very slow, but does not consume large memory.
    Save_df = DataToSave(tracking_results, localization_results, path_folder, frame_start=frame_start,
                         med_fps=Glimpse_data.med_fps, window=20, factor_p2n=10000/180,
                         BM_lower=BM_lower, BM_upper=BM_upper, ratio_lower=ratio_lower, ratio_upper=ratio_upper,
                         sx_sy_lower=sx_sy_lower, sx_sy_upper=sx_sy_upper, criteria_mode=criteria_mode
                         )
    print('construct Save_df end')
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    Save_df.save_all_dict_df_to_excel()
    Save_df.save_selected_dict_df_to_excel()
    # Save_df.save_removed_dict_df_to_excel()
    time_spent = time.time() - t1
    print('spent ' + str(time_spent) + ' s')
    return Save_df


if __name__ == "__main__":
    path_folder = select_folder()
    Save_df = get_analyzed_sheet(path_folder, analyzed_mode, criteria_mode, frame_n,
                                 BM_lower, BM_upper, ratio_lower, ratio_upper, sx_sy_lower, sx_sy_upper)

    # path_folders = glob(os.path.join(path_folder, '*'))
    # for path_folder in path_folders:
    #     get_analyzed_sheet(path_folder, analyzed_mode, frame_n)

# TODO: https://github.com/pandas-dev/pandas/issues/41681#issue-902703572
