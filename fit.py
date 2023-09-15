import time
import numba as nb
import numpy as np
import scipy.integrate as int
from colossus.cosmology import cosmology as cm
import matplotlib.pyplot as plt
import numdifftools as nd
"""
all the quantities below are taken from the Planck collaboration https://arxiv.org/pdf/1807.06209.pdf and are given in the following units:
---mass         : SOLAR MASSES
---length       : MEGAPARSECS
---time         : SECONDS
"""
start_time = time.time()
accuracy = 3

h = 0.673
Hubble = 100*h
rhocrit = 1260e8
Omegam = 0.315
OmegaLambda = 0.685
Omegab = 0.049
Omegac = 0.265
As = 2.1e-9
ns = 0.964
zeq = 3402
aeq = 1/(1+zeq)
keq = 0.015*h
G = (4301e-12)
light = 2.997e5
deltac = 1.686
fb = Omegab/Omegam
rhobar = rhocrit*Omegam


cosmo = cm.setCosmology('planck18')


def radius(M):
    return (M/(6*np.pi**2*rhobar))**(1/3)*h

def SigmaAdquadroCosmo(M):
    """Choose here which filter we are considering the fit for
    """
    return cosmo.sigma(radius(M),filt='sharp-k')**2

masses = np.logspace(2,17,num=1000)
coeffs = np.polyfit(np.log(masses),np.log(SigmaAdquadroCosmo(masses)),16)
 


