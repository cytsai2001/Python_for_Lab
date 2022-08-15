from TPM.localization import select_folder
from glob import glob
import random
import string
import numpy as np
import os
import datetime
import pandas as pd
import scipy.linalg as la
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

### get analyzed sheet names

##  path_dat:list of path; sheet_names:list of string, axis=0(add vertically)
def get_df_dict(path_data, sheet_names, axis):
    df_dict = dict()
    for i, path in enumerate(path_data):
        for sheet_name in sheet_names:
            if i == 0:  ## initiate df_dict
                df = pd.read_excel(path, sheet_name=sheet_name)
                df_dict[f'{sheet_name}'] = df
            else:  ## append df_dict
                df = pd.read_excel(path, sheet_name=sheet_name)
                df_dict[f'{sheet_name}'] = pd.concat([df_dict[f'{sheet_name}'], df], axis=axis)
    return df_dict


def get_analyzed_sheet_names():
    return ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing',
            'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared',
            'avg_attrs', 'std_attrs']


##  concatenate cetain attr from all df, output: 1D
def get_attr(df_dict, column_name):
    data = []
    n = len(df_dict)
    for i in range(n):
        df = df_dict[f'{i}']
        data = np.append(data, np.array(df[column_name]))
        data = data.reshape(len(data), 1)
    return data


##  add 2n-word random texts(n-word number and n-word letter)
def gen_random_code(n):
    digits = "".join([random.choice(string.digits) for i in range(n)])
    chars = "".join([random.choice(string.ascii_letters) for i in range(n)])
    return digits + chars


### getting date
def get_date():
    filename_time = datetime.datetime.today().strftime('%Y-%m-%d')  # yy-mm-dd
    return filename_time


### normalize to mean = 0, std = 1
def normalize_data(data):
    data = np.array(data)
    data_nor = []
    for datum in data.T:
        mean = np.mean(datum)
        std = np.std(datum, ddof=1)
        datum_nor = (datum - mean) / std
        data_nor += [datum_nor]
    return np.nan_to_num(np.array(data_nor).T)


### get analyzed sheet names, add median
def get_analyzed_sheet_names():
    return ['BMx_sliding', 'BMy_sliding', 'BMx_fixing', 'BMy_fixing',
            'sx_sy', 'xy_ratio_sliding', 'xy_ratio_fixing', 'sx_over_sy_squared']


### get reshape sheet names
def get_reshape_sheet_names():
    return ['amplitude', 'sx', 'sy', 'x', 'y', 'theta_deg', 'offset', 'intensity', 'intensity_integral', 'ss_res']


def get_data_from_excel(path_folder, sheet_names, excel_name, axis):
    # path_folders = glob(os.path.join(path_folder, '*'))
    # path_data = [glob(os.path.join(x, '*' + excel_name))[0] for x in path_folders if
    #              glob(os.path.join(x, '*' + excel_name)) != []]
    path_data = [glob(os.path.join(path_folder, '*' + excel_name))[0]]
    # sheet_names = ['med_attrs', 'std_attrs', 'avg_attrs']
    df_attrs_dict = get_df_dict(path_data, sheet_names, axis)
    return df_attrs_dict

##  add 2n-word random texts(n-word number and n-word letter)
def gen_random_code(n):
    digits = "".join([random.choice(string.digits) for i in range(n)])
    chars = "".join([random.choice(string.ascii_letters) for i in range(n)])
    return digits + chars


# excel_name = 'snapshot-fitresults_reshape_analyzed_selected.xlsx'
excel_name = 'fitresults_reshape_analyzed.xlsx'
#
path_folder = select_folder()
df_attrs_dict = get_data_from_excel(path_folder, sheet_names=['med_attrs', 'std_attrs', 'avg_attrs'],
                                    excel_name=excel_name, axis=0)
df_analyzed_dict = get_data_from_excel(path_folder, sheet_names=get_analyzed_sheet_names(), excel_name=excel_name,
                                       axis=1)

##  select statistical attributes for clustering analysis
select_columns = ['BMx_fixing', 'BMy_fixing', 'sx_sy']
sheet_names = ['med_attrs', 'std_attrs', 'avg_attrs']
df_select_attrs_dict = dict()
df_select_attrs_nor_dict = dict()
for sheet_name in sheet_names:
    df_select = df_attrs_dict[f'{sheet_name}'][select_columns]

    df_select_attrs_dict[f'{sheet_name}'] = df_attrs_dict[f'{sheet_name}'][select_columns]
    df_select_attrs_nor_dict[f'{sheet_name}'] = pd.DataFrame(data=normalize_data(df_select), columns=df_select.columns,
                                                             index=df_select.index)

beads_name = df_attrs_dict['med_attrs']['Unnamed: 0']
sx_sy = df_select_attrs_dict['med_attrs']['BMx_fixing']
random_string = gen_random_code(3)

fig, ax = plt.subplots()
ax.hist(sx_sy, range=(0, 100), bins=100, density=True)
ax.set_xlim((0, 100))
ax.set_xlabel("BMx (nm)")
ax.set_ylabel("Probability density")
ax.text(0.05, 0.9, f"N = {len(sx_sy)}", transform=ax.transAxes)
fig.savefig(os.path.join(path_folder, random_string + '-Area_histogram.png'))