import sympy as sym
import time
import numba as nb
import numpy as np
import scipy.integrate as int
from colossus.cosmology import cosmology as cm
import matplotlib.pyplot as plt
import numdifftools as nd
"""
all the quantities below are taken from the Planck collaboration et al. https://arxiv.org/pdf/1807.06209.pdf and are given in the following units:
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


#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def radius(M):
    """We define the radius in terms of the mass contained R(M)
    """
    return (M/(6*np.pi**2*rhobar))**(1/3)*h




#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def OmegaLam(a):
    return OmegaLambda/(OmegaLambda + Omegam/(a**3))
#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def OmegaM(a):
    return Omegam/(OmegaLambda*(a**3) + Omegam)
#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def GFF(a):
    """This is the growth factor of the linear density field. It was checked against 
    cosmo.growthFactor(z).
    """
    return 5/2*OmegaM(a)*a/(OmegaM(a)**(4/7) - OmegaLam(a) + (1 + OmegaM(a)/2)*(1 + OmegaLam(a)/70))




#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def nbarPBH(f,m):
    """Average number density of PBHs
    """
    return f*rhocrit*(Omegam - Omegab)/m

#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def SigmaAdquadro(M):
    """This is the adiabatic part of the variance. To speed up computation we fit it with:
    cosmo.sigma(radius(M),z=0,filt='sharp-k')**2 in the range of masses 2<M/M_solar<17. Notice that
    there is no time dependence here, since we put it in the barrier omega(z). The coefficients are taken
    from the file fit.py. 
    """
    coeffs = [ 3.06152467e-19, -9.43888003e-17,  1.32336493e-14, -1.11813970e-12,
    6.36171659e-11, -2.58002785e-09,  7.70124805e-08, -1.72233268e-06,
    2.90909739e-05, -3.71008415e-04,  3.54202830e-03, -2.48405668e-02,
    1.23679874e-01, -4.13148793e-01, 7.45437780e-01,  4.90617585e+00] 
    a  = 0
    for i in range(len(coeffs)):
        a += coeffs[-(i+1)]*np.log(M)**i
    return np.exp(a)

#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def makeExpr():
    X = sym.symbols('X')
    a = 0
    coeffs = [ 2.96455120e-11, -5.04257782e-09,  3.53297147e-07, -1.34074136e-05,
    2.98930915e-04, -4.03491116e-03,  3.03083641e-02, -2.03206994e-01,
    5.85216969e+00] 
    for i in range(len(coeffs)):
        a += coeffs[-(i+1)]*sym.log(X)**i
    a = sym.exp(a)
    deriv = sym.diff(a, X)
    return deriv
derivv = makeExpr()
#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def der(x):
    """Derivative of the adiabatic variance
    """
    X = sym.symbols('X')
    deriv = sym.lambdify(X, derivv)
    return deriv(x)

#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def d(z):
    return 1/(1+z)
#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def omega(z):
    """This is the time dependent barrier. The new growth factor is "a" since for redshifts z>10 we have this limit for 
    both growth factors. See the overleaf document for details 
    """
    return deltac/d(z) 

#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def sigmatotQuadro(M,f,m):
    """Total variance, sum of the adiabatic part and PBH part.
    """
    gammma = (Omegam - Omegab)/Omegam
    aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
    return (SigmaAdquadro(M)*(1/GFF(a=1))**2 +
            (f**2*2*rhobar/(nbarPBH(f,m)*9*np.pi*M))*(0.56*3*gammma/(2*aminus*aeq))**2)


#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def sigmaLogDerivClean(M,f,m):
    """returns dlog\simga/dlogM
    """
    gammma = (Omegam - Omegab)/Omegam
    aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
    return np.abs( 0.5*M/sigmatotQuadro(M,f,m)*
            ( der(M)*(1/GFF(a=1))**2 - 
             f**2*2*rhobar/(nbarPBH(f,m)*9*M**2*np.pi)*(0.56*3*gammma/(2*aminus*aeq))**2) 
             )


#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def massFunctionPBHST(M,zf,f,m):
    """This is the halo mass function multiplied by M (since we need to find the star mass density later). This was 
    calculated using the Sheth-Tormen halo multiplicity.
    The mass function is calculated at z = zf+Deltaz, since the idea is to use it later for the mass function at (Mi,zi) 
    """
    a = 0.75
    A = 0.32
    p = 0.3
    return (M*
            ( np.sqrt(a)*A )*
            ( np.sqrt(2/np.pi) )*
            ( omega(zf)/sigmatotQuadro(M,f,m)**0.5 )*
            ( np.exp(-0.5*a*omega(zf)**2/sigmatotQuadro(M,f,m)) )*
            ( 1+(a*omega(zf)**2/sigmatotQuadro(M,f,m))**(-p) )*
            ( rhobar/M**2 )*
            (sigmaLogDerivClean(M,f,m) )
            )

#@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def rhostarNoTransition(z,Mstarf,f,m,ep):
    """Returns the stellar mass density, but only the first term, with no transition probability.
    """
    Mf = np.logspace(np.log10(Mstarf/(ep*fb)),17,num=100000)
    return ep*fb*np.trapz(massFunctionPBHST(Mf,z,f,m),Mf)


#print('{:.5E}'.format(rhostarNoTransition(z=9,Mstarf=1e10,f=1,m=1e4,ep=0.1)))
print("--- %s seconds ---" % (time.time() - start_time))
