from colossus.cosmology import cosmology as cm
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
import numdifftools as nd
import sympy as sym
from autograd import grad
from colossus.lss import mass_function
"""
All the quantities below are taken from the Planck collaboration et al. https://arxiv.org/pdf/1807.06209.pdf and are given in the following units:
---mass         : SOLAR MASSES
---length       : MEGAPARSECS
---time         : SECONDS

Uncomment moreFilters01 and moreFilters1 to obtain Figure 17

"""

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
rhobar = rhocrit*Omegam
fb=Omegab/Omegam

cosmo = cm.setCosmology('planck18')



"""R(M) for the 3 different filters
"""
def radiussharpk(M):
    return (M/(6*np.pi**2*rhobar))**(1/3)*h
def radiusRealToP(M):
    return (3*M/(4*np.pi*rhobar))**(1/3)*h
def radiusGaussian(M):
    return (M/rhobar)**(1/3)*1/(2*np.pi)**0.5*h



"""Variances for the 3 different filters
"""
def sigmaQuadroAdSharpK(M,rs):
    return cosmo.sigma(radiussharpk(M),z=rs,filt='sharp-k')**2

def sigmaQuadroAdRealTop(M,rs):
    return cosmo.sigma(radiusRealToP(M),z=rs,filt='tophat')**2

def sigmaQuadroAdGauss(M,rs):
    return cosmo.sigma(radiusGaussian(M),z=rs,filt='gaussian')**2



def massFunctionSTSharpK(M,rs):
    """Halo mass function with the sharp-k filter with corrections for ellipsoidal dynamics (Sheth-Tormen halo
    multiplicity)
    """
    a = 0.75
    A = 0.32
    p = 0.3
    return -(M*
            ( np.sqrt(a)*A )*
            ( np.sqrt(2/np.pi) )*
            ( (deltac)/sigmaQuadroAdSharpK(M,rs)**0.5 )*
            ( np.exp(-0.5*a*(deltac)**2/sigmaQuadroAdSharpK(M,rs)) )*
            ( 1+(a*(deltac)**2/sigmaQuadroAdSharpK(M,rs))**(-p) )*
            ( rhobar/M**2 )*
            ( cosmo.sigma(radiusRealToP(M),z=0,filt='sharp-k',derivative=True)/3 )
            )


def massFunctionSTRealTop(M,rs):
    """Halo mass function with the real space tophat filter with corrections for ellipsoidal dynamics (Sheth-Tormen halo
    multiplicity)
    """
    a = 0.75
    A = 0.32
    p = 0.3
    return -(M*
            ( np.sqrt(a)*A )*
            ( np.sqrt(2/np.pi) )*
            ( deltac/sigmaQuadroAdRealTop(M,rs)**0.5 )*
            ( np.exp(-0.5*a*deltac**2/sigmaQuadroAdRealTop(M,rs)) )*
            ( 1+(a*deltac**2/sigmaQuadroAdRealTop(M,rs))**(-p) )*
            ( rhobar/M**2 )*
            ( cosmo.sigma(radiusRealToP(M),z=rs,filt='tophat',derivative=True)/3 )
            )

def massFunctionSTGauss(M,rs):
    """Halo mass function with the Gaussian filter with corrections for ellipsoidal dynamics (Sheth-Tormen halo
    multiplicity)
    """
    a = 0.75
    A = 0.32
    p = 0.3
    return -(M*
            ( np.sqrt(a)*A )*
            ( np.sqrt(2/np.pi) )*
            ( deltac/sigmaQuadroAdGauss(M,rs)**0.5 )*
            ( np.exp(-0.5*a*deltac**2/sigmaQuadroAdGauss(M,rs)) )*
            ( 1+(a*deltac**2/sigmaQuadroAdGauss(M,rs))**(-p) )*
            ( rhobar/M**2 )*
            ( cosmo.sigma(radiusGaussian(M),z=rs,filt='gaussian',derivative=True)/3 )
            )



"""Cumulative comoving stellar mass densities for the 3 different filters
"""
def rhostarSTTrapzSharpK(Mstar,z,ep):
    Mf = np.logspace(np.log10(Mstar/(ep*fb)),17,num=1000)
    return (ep*fb*np.trapz(massFunctionSTSharpK(Mf,z),Mf))
def rhostarSTTrapzRealTop(Mstar,z,ep):
    Mf = np.logspace(np.log10(Mstar/(ep*fb)),17,num=1000)
    return (ep*fb*np.trapz(massFunctionSTRealTop(Mf,z),Mf))
def rhostarSTTrapzGauss(Mstar,z,ep):
    Mf = np.logspace(np.log10(Mstar/(ep*fb)),17,num=1000)
    return (ep*fb*np.trapz(massFunctionSTGauss(Mf,z),Mf))





masses = np.logspace(8,12,num=1000)

def valuesSharpK(x,z,ep):
    ls = []
    for i in x:
        ls.append(rhostarSTTrapzSharpK(i,z,ep))
    return np.array(ls)

def valuesRealTop(x,z,ep):
    ls = []
    for i in x:
        ls.append(rhostarSTTrapzRealTop(i,z,ep))
    return np.array(ls)
def valuesGaussian(x,z,ep):
    ls = []
    for i in x:
        ls.append(rhostarSTTrapzGauss(i,z,ep))
    return np.array(ls)




"""#moreFilters01
plt.figure(figsize=((10,8)))
plt.loglog(masses,valuesSharpK(masses,9,0.1),label="Sharp-K")
plt.loglog(masses,valuesRealTop(masses,9,0.1),label="Real Tophat")
plt.loglog(masses,valuesGaussian(masses,9,0.1),label="Gaussian")
plt.ylim(1e-3,1e6)
plt.ylabel(r'$\rho_\star(>M_\star,z) [M_\odot \ Mpc^{-3}]$',fontsize=20 )
plt.xlabel(r'$M_\star [M_\odot]$',fontsize=20)
plt.xlim(min(masses),max(masses))
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.legend(fontsize=18,frameon=False)
plt.text(2e8,1e-1,r"$\epsilon=0.1, \ z \sim 9$",fontsize = 18)
plt.grid()         
plt.grid(which='minor', alpha=0.3)             
plt.show()

"""

"""#moreFilters1
plt.figure(figsize=((10,8)))
plt.loglog(masses,valuesSharpK(masses,9,1),label="Sharp-K")
plt.loglog(masses,valuesRealTop(masses,9,1),label="Real Tophat")
plt.loglog(masses,valuesGaussian(masses,9,1),label="Gaussian")
plt.ylim(1e-3,1e6)
plt.ylabel(r'$\rho_\star(>M_\star,z) [M_\odot \ Mpc^{-3}]$',fontsize=20 )
plt.xlabel(r'$M_\star [M_\odot]$',fontsize=20)
plt.xlim(min(masses),max(masses))
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.legend(fontsize=18,frameon=False)
plt.text(2e8,1e-1,r"$\epsilon=1, \ z \sim 9$",fontsize = 18)
plt.grid()         
plt.grid(which='minor', alpha=0.3)             
plt.show()

"""



