# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:48:42 2021
"""

import numpy as np
from EM_Algorithm.EM import EM

if __name__ == '__main__':
    ##  import data
    # data = np.array([179, 165, 175, 185, 158, 190])
    data = np.array([4.35025209568627,6.02028403216761,8.74013549119374,8.79569063382595,6.30754174712644,6.17940619409282,2.56770170740741,4.09325603111111,4.05575733490197,4.98339941176471,3.34095459770115,5.00863758919531,7.38938795612111,7.39868092243186,4.40006851851852,6.27209499579082,5.35435634795918,4.18512907894737,3.28353755685509,9.16097426042396,5.63977660377359,7.12999459684424,6.53892494674741,8.72490313997478,5.12361121794871,9.96075507575757,4.63959454545455,4.05804113082041,2.32118111701031,2.96073818298969,5.73656437500000,5.53746076388889,2.92860277777778,2.94117303921568,2.96609860999415,4.43907202918790,9.15801579831469,6.47145942681818,5.82521718750000,11.5910488541667,2.35736168382353,4.33874518676471,2.17904692307692,2.42293252136752,2.27889488888889,3.40683852380952,3.38829304029305,4.11920620782726,9.00506035314685,3.59073716186253,4.36922064562410,4.23581264705882,3.24299982758621,8.83586320583333,8.48158618461538,5.01307128205129,13.6414206250000,3.53869546764706,7.39194575000000,5.67960400000000,2.59289492307693,5.95084875259875,10.1461682668816,8.17695539376624,6.04041297208539,4.79999806687566,5.97587668478261,6.90407480331263,5.22437378151260,11.3061579696395,8.34094021020408,9.80657928571429,4.26540867346939,4.72541359925788,9.35106106060606,6.61003634751770,9.15166902061106,10.5560535551465,7.02705234811166,7.53111469449486,2.77092562252404,5.70248472820513,4.30679525274726,3.54356904433497,3.43371724137931,8.15848085106383,6.76741071840000,3.71760370833333,4.26958538690476,2.34370449735451,5.34461455026455,3.56330604597701,3.48403185185185,2.59677200779727,3.66513947368421,3.30535000000000,9.52206230491804,6.18010657101449,5.63024459308807,5.41033917901669,4.67241280233528,6.24400914966036,9.78777438342209,7.39591117084155,10.1914972469325,9.85872378424658,8.78486689757160,9.91476024751675,8.97093417312407,5.61877513027866,5.56929660188998,5.96719300884956,16.2050459773942,7.68322756344598,4.18777229980620,5.36938336309524,4.11505008597883,6.76934169346979,5.79203198738926,4.00763740289414,5.21532623076923,7.63250354545455,4.65773116883117,9.33270112781954,5.66747811551783,5.13917333165323,5.32319821082746,4.34257265662943,3.42493850074964,5.29084105590061,6.85445320197043,4.05557226386807,6.31409484855378,7.27299129950629,3.72318544674325,5.73032427951518,7.15471772951794,4.40682487276951,7.49431772463308,6.80203304935065,3.37185547878789,7.39921446334311,6.22458818181819,4.66357500000000,6.62360071428572,7.74363246753247,2.66032070707072,8.15780895927173,5.02469860657895,16.0765605451128,6.29293597402597,4.81025136363638,3.09694347130682,4.87115098181818,6.53759592727273]
                    )
    ##  fit GMM
    EM = EM(data)
    n_components = 2
    f, m, s, converged = EM.GMM(n_components)
    EM.plot_EM_results()
    EM.plot_fit_gauss()
