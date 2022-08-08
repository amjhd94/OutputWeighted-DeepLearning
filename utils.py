import numpy as np
import scipy
from KDEpy import FFTKDE

def custom_KDE(data, weights=None, bw=None):
    data = data.flatten()
    if bw is None:
        try:
            sc = scipy.stats.gaussian_kde(data, weights=weights)
            bw = np.sqrt(sc.covariance).flatten()
            # Ensure that bw is a scalar value
            if np.size(bw) == 1:
                bw = np.asscalar(bw)
            else:
                raise ValueError("The bw must be a number.")
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0           
    return FFTKDE(bw=bw).fit(data, weights)