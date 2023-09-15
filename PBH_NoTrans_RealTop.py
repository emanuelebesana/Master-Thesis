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
    return (3*M/(4*np.pi*rhobar))**(1/3)



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




"""In the following block we calculate the PBH part of the variance, for the real tophat filter
"""
ss = 1/aeq
gammma = (Omegam - Omegab)/Omegam
aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
def Dzero(z):
    return (1 + (3*gammma)/(2*aminus)*(ss*(1/(1+z))))**aminus-1
def Piso(k,z,f,m):
    return f**2*(Dzero(z)-1)**2/nbarPBH(f, m)
def RealTopFilt(k,R):
    return 3/(k*R)**3*(np.sin(k*R)-k*R*np.cos(k*R))
def isoIntegrand(k,M,rs,f,m):
    return k**2/(2*np.pi**2)*Piso(k,rs,f,m)*RealTopFilt(k,radius(M))**2
def sigmaIsoQuadro(M,rs,f,m):
    ks = np.linspace(1e-5,1e16,num=10000)
    ls = []
    for x in M:
        ls.append(np.trapz(isoIntegrand(ks,M,rs,f,m),ks))
    return np.array(ls)
def sigmaIsoQuadroDer(M,rs,f,m):
    return np.gradient(sigmaIsoQuadro(M,rs,f,m),M)





def SigmaAdquadro(M):
    """Adiabatic part of the variance. The fit is done with fit.py against cosmo.sigma(radius(M),filt='tophat')
    """
    coeffs = [ 5.24901547e-19, -1.42650812e-16,  1.62231969e-14, -9.45280883e-13,
    2.37569493e-11,  3.63613948e-10, -4.15953006e-08,  7.30857969e-07,
    3.27820923e-05, -2.27512649e-03,  6.69778269e-02, -1.20722593e+00,
    1.41737655e+01, -1.06473891e+02,  4.67141979e+02, -9.08603778e+02] 
    a  = 0
    for i in range(len(coeffs)):
        a += coeffs[-(i+1)]*np.log(M)**i
    return np.exp(a)

def makeExpr():
    X = sym.symbols('X')
    a = 0
    coeffs = [ 1.62996174e-10, -3.19576654e-08,  2.69132275e-06, -1.27407724e-04,
    3.70809460e-03, -6.80289961e-02,  7.66357099e-01, -4.95067444e+00,
    1.88991357e+01] 
    for i in range(len(coeffs)):
        a += coeffs[-(i+1)]*sym.log(X)**i
    a = sym.exp(a)
    deriv = sym.diff(a, X)
    return deriv
derivv = makeExpr()
def der(x):
    """Derivative of the adiabatic variance
    """
    X = sym.symbols('X')
    deriv = sym.lambdify(X, derivv)
    return deriv(x)



def sigmatotQuadro(M,rs,f,m):
    """Total variance, sum of the adiabatic part and PBH part.
    """
    return (SigmaAdquadro(M)*(cosmo.growthFactor(rs)**2) + sigmaIsoQuadro(M,rs,f,m))
def sigmaLogDerivClean(M,rs,f,m):
    """returns dlog\simga/dlogM
    """
    return np.abs( 0.5*M/sigmatotQuadro(M,rs,f,m)*
            ( der(M)*(cosmo.growthFactor(rs)**2) + sigmaIsoQuadroDer(M,rs,f,m)) )


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
            ( deltac/sigmatotQuadro(M,zf,f,m)**0.5 )*
            ( np.exp(-0.5*a*deltac**2/sigmatotQuadro(M,zf,f,m)) )*
            ( 1+(a*deltac**2/sigmatotQuadro(M,zf,f,m))**(-p) )*
            ( rhobar/M**2 )*
            (sigmaLogDerivClean(M,zf,f,m) )
            )
def rhostarNoTransition(z,Mstarf,f,m,ep):
    """Returns the stellar mass density, but only the first term, with no transition probability.
    """
    Mf = np.logspace(np.log10(Mstarf/(ep*fb)),17,num=10000)
    return (ep*fb*np.trapz(massFunctionPBHST(Mf,z,f,m),Mf))


#print('{:.5E}'.format(rhostarNoTransitionCosmo(z=9,Mstarf=8e9,f=1,m=2e7,ep=0.1)))
print("m= ",2e4,", rhostar= ",'{:.5E}'.format(rhostarNoTransition(z=10,Mstarf=10**(10.5),f=1,m=3.5e4,ep=0.1)))
print("--- %s seconds ---" % (time.time() - start_time))
























