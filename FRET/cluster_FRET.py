from basic.select import get_mat, get_files
from EM_Algorithm.EM import EM
import numpy as np
import pandas as pd
import random
import string
import os

def get_params(dwell):
    EM_p = EM(dwell)
    n_components_p = EM_p.opt_components(tolerance=1e-2, mode='PEM', criteria='BIC', figure=False)
    f_tau, tau, s_tau, converged_p = EM_p.PEM(n_components_p)
    return EM_p, f_tau, tau, s_tau, converged_p

def collect_params(*args):
    output = []
    for arg in args:
        arg[0] += [arg[1]]
        output += [arg[0]]
    return output

def gen_random_code(n):
    digits = "".join([random.choice(string.digits) for i in range(n)])
    chars = "".join([random.choice(string.ascii_letters) for i in range(n)])
    return digits + chars

def reshape_results(*args):
    all_output = []
    for arg in args:
        n = []
        for x in arg:
            n += [len(x)]
        n = max(n)
        output = np.zeros((len(args[0]), n))
        for i,x in enumerate(arg):
            l = len(x)
            output[i,:l] = x
        all_output += [output]
    return all_output

def get_col(name,n):
    output = [f'component_{name}_{i+1}' for i in range(n)]
    return output


all_path = get_files('*.mat')

for path in all_path:
    name = os.path.split(path)[-1]
    data = get_mat(path)
    dwell = data['Transmat']

    EM_on, EM_off = [], []
    f_on, f_off = [], []
    tau_on, tau_off = [], []
    # s_on, s_off = [], []

    m_shape = len(dwell)
    for i in range(1,len(dwell)):

        dwell_on = dwell[m_shape-i, m_shape-i-1]
        dwell_off = dwell[m_shape-i-1, m_shape-i]
        ## EM
        EM_p_on, f_tau_on, tau_i_on, s_tau_on, converged_p_on = get_params(dwell_on)
        EM_p_off, f_tau_off, tau_i_off, s_tau_off, converged_p_off = get_params(dwell_off)
        ## store results
        EM_on,EM_off,f_on,f_off,tau_on,tau_off = collect_params([EM_on,EM_p_on],[EM_off,EM_p_off],[f_on,f_tau_on],[f_off,f_tau_off],[tau_on,tau_i_on],[tau_off,tau_i_off])

        # EM_p.plot_fit_exp(xlim=[0, 10])

    f_on, tau_on, f_off, tau_off = reshape_results(f_on, tau_on, f_off, tau_off)

    df_f_on = pd.DataFrame(f_on, columns=get_col('f',f_on.shape[1]))
    df_tau_on = pd.DataFrame(tau_on, columns=get_col('tau',tau_on.shape[1]))
    df_on = pd.concat([df_f_on, df_tau_on], axis=1)

    df_f_off = pd.DataFrame(f_off, columns=get_col('f',f_off.shape[1]))
    df_tau_off = pd.DataFrame(tau_off, columns=get_col('tau',tau_off.shape[1]))
    df_off = pd.concat([df_f_off, df_tau_off], axis=1)

    df = [df_on, df_off]
    sheet_names = ['on', 'off']

    writer = pd.ExcelWriter( f'{gen_random_code(3)}_{name}_EM_results.xlsx')
    for i in range(2):
        df[i].to_excel(writer, sheet_name=sheet_names[i], index=True)
    writer.save()