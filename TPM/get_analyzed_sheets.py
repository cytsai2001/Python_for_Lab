

from TPM.BinaryImage import BinaryImage
from TPM.DataToSave import DataToSave
from TPM.localization import select_folder
import time
import pandas as pd
import numpy as np
import os
from glob import glob

analyzed_mode = 'all'  ## if analyzed_mode = 'all', analyze all frames of .csv file
criteria_mode = 'QD'   ## if criteria_mode = 'QD', criteria will include sx_sy; else, criteria will only consider BM and ratio.
frame_n = 30000
frame_start = 0
BM_lower = 40
BM_upper = 400
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
    df = pd.read_csv(path_data)
    bead_number = int(max(1 + df['aoi']))
    tracking_results = np.array(df)
    tracking_results = tracking_results[0:bead_number*frame_n, :]
    localization_results = np.zeros((bead_number, 1))
    Save_df = DataToSave(tracking_results, localization_results, path_folder, frame_start=frame_start,
                         med_fps=Glimpse_data.med_fps, window=20, factor_p2n=10000/180,
                         BM_lower=BM_lower, BM_upper=BM_upper, ratio_lower=ratio_lower, ratio_upper=ratio_upper,
                         sx_sy_lower=sx_sy_lower, sx_sy_upper=sx_sy_upper, criteria_mode=criteria_mode
                         )
    Save_df.save_all_dict_df_to_excel()
    Save_df.save_selected_dict_df_to_excel()
    Save_df.save_removed_dict_df_to_excel()
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
