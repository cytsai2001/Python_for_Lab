
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

from basic.select import get_files, select_folder
import scipy.io as sio
import numpy as np
from EM_Algorithm.EM import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    all_gauss = []
    all_survival = []
    all_results = []
    
    n_sample = []
    # conc = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]  ## S5S1
    # conc = [1.0, 1.2, 1.5, 1.8, 2.0, 3.0, 4.0] ## m51 only
    conc = ['0.10', '0.20', '0.25', '0.50', '0.70', '0.80', '1.10', '1.20', '2.00']  ## EcRecA
    # conc = ['2.00']
    # path_folder = select_folder()
    path_folder = '/home/hwligroup/Desktop/20210330/YYH_m51_data/step-dwell time/EcRecA'

    for i in range(1):

        tolerance = 1e-3

        for c in conc:
            path_data = get_files(f'*{c}*.mat', dialog=False, path_folder=path_folder)
            step = []
            dwell = []
            for path in path_data:
                data = sio.loadmat(path)
                step = np.append(step, data['step'])
                dwell = np.append(dwell, [data['dwell']])
            n_sample += [len(step)]
    
            ## get Gaussian EM results
            EM_g = EM(step)
            n_components_g = EM_g.opt_components(tolerance=1e-2, mode='GMM', criteria='BIC', figure=False)
            f, m, s, converged_g = EM_g.GMM(n_components_g, rand_init=True, tolerance=1e-2)

            all_gauss += [np.array([f, m, s, converged_g]).T]
    
            ## get poisson EM results
            EM_p = EM(dwell)
            n_components_p = EM_p.opt_components(tolerance=1e-2, mode='PEM', criteria='BIC', figure=False)
            f_tau, tau, s_tau, converged_p = EM_p.PEM(n_components_p)

            all_survival += [np.array([f_tau, tau, converged_p]).T]

            ##  2D clustering
            step_dwell = np.array([step, dwell]).T
            EM_gp = EM(step_dwell, dim=2)
            opt_components = EM_gp.opt_components(tolerance=1e-2, mode='GPEM', criteria='BIC')
            f1, m1, s1, tau1, converged_gp = EM_gp.GPEM(n_components=opt_components, tolerance=1e-2, rand_init=True)
            # para = [f1[-1].ravel(), m1[-1].ravel(), s1[-1].ravel(), f2[-1].ravel(), tau1[-1].ravel()]
            # labels, data_cluster = EM_gp.predict(step_dwell, function=ln_gau_exp_pdf, paras=para)

            ##  plot figure
            # EM_g.plot_fit_gauss(scatter=True, xlim=[0, 20], save=True, path=f'{c}_gauss.png')
            EM_p.plot_fit_exp(xlim=[0, 15], save=True, path=f'{c}_survival.png')
            # EM_gp.plot_gp_contour(xlim=[0, 20], ylim=[0, 20], save=True, path=f'{c}_2D.png')
            
            all_results += [np.array([f1, m1, s1, tau1, converged_gp]).T]

