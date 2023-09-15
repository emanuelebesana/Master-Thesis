from colossus.cosmology import cosmology as cm
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
"""
All the quantities below are taken from the Planck collaboration https://arxiv.org/pdf/1807.06209.pdf and are given in the following units:
---mass         : SOLAR MASSES
---length       : MEGAPARSECS
---time         : SECONDS

Uncomment conditional_MF_z1.pdf, conditional_MF_z025.pdf and conditional_MF_z3.pdf to obtain Figure 7


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


cosmo = cm.setCosmology('planck18')


def radius(M):
    """We define the radius in terms of the mass contained R(M)
    """
    return (3*M/(4*np.pi*rhobar))**(1/3)*h  

def omega(rs):
    """This is omega(a) as defined by Zentner 2006. It is the moving barrier
    """
    return deltac/cosmo.growthFactor(rs)

def SigmaQuadro(M):
    """Variance obtained with COLOSSUS package
    """
    return cosmo.sigma(radius(M),z=0,filt='sharp-k')**2

def f(M1,M2,z1,z2):
    """Equation (52) of Zentner 2006
    """
    return 1/np.sqrt(2*np.pi)*(omega(z2)-omega(z1))/(SigmaQuadro(M2)-SigmaQuadro(M1))**(3/2)*np.exp( - ((omega(z2)-omega(z1))**2) / (2*(SigmaQuadro(M2)-SigmaQuadro(M1))) )

def conditional_massFunction(M1,fractions,z1,z2):
    """The conditional halo mass function.
    Important: M1 is NOT an array, but M2 is. From a numpy standpoint the operation M2/M1 is therefore well defined as
    a divition by a scalar.
    """
    return M1*f(M1,fractions,z1,z2)*np.abs(2/3*cosmo.sigma(radius(fractions),z=0,filt='sharp-k', derivative=True)*SigmaQuadro(fractions)/(fractions))


M1 =1e12
M11=1e13
M111 = 1e14
M1111 = 1e15
z1 = 0
z2 = 1
z3 = 1/4
z4 = 3 
fractions = np.logspace(-3,0,num=500)
M2 = np.logspace(11,14,num=1000)

"""conditional_MF_z1.pdf
plt.figure(figsize=((10,8)))
plt.loglog(fractions,conditional_massFunction(M1,fractions*M1,z1,z2),"g",label=r'$M_1 =10^{12}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M1,fractions*M1,z1,z2),"g",label=r'$M_1 =10^{12}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M111,fractions*M111,z1,z2),"b",label=r'$M_1 =10^{14}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M1111,fractions*M1111,z1,z2),"grey",label=r'$M_1 =10^{15}M_\odot$')
plt.xlim(1e-3,1)
plt.ylim(0.001,100)
plt.xlabel(r'$M_2/M_1$',fontsize=20)
plt.ylabel(r'$dn(M_2/M_1|M_1)/d\log(M_2/M_1)$',fontsize=20)
plt.legend(fontsize=20,frameon=False)
plt.text(0.01,0.3,r'$z_1=0$',fontsize=20)
plt.text(0.01,0.1,r'$z_2=1$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()
"""

"""conditional_MF_z025.pdf
plt.figure(figsize=((10,8)))
plt.loglog(fractions,conditional_massFunction(M1,fractions*M1,z1,z3),"g",label=r'$M_1 =10^{12}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M11,fractions*M11,z1,z3),"r",label=r'$M_1 =10^{13}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M111,fractions*M111,z1,z3),"b",label=r'$M_1 =10^{14}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M1111,fractions*M1111,z1,z3),"grey",label=r'$M_1 =10^{15}M_\odot$')
plt.xlim(1e-3,1)
plt.ylim(0.001,100)
plt.xlabel(r'$M_2/M_1$',fontsize=20)
plt.ylabel(r'$dn(M_2/M_1|M_1)/d\log(M_2/M_1)$',fontsize=20)
#plt.legend(fontsize=20,frameon=False)
plt.text(0.01,0.3,r'$z_1=0$',fontsize=20)
plt.text(0.01,0.1,r'$z_2=1/4$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()
"""

"""conditional_MF_z3.pdf
plt.figure(figsize=((10,8)))
plt.loglog(fractions,conditional_massFunction(M1,fractions*M1,z1,z4),"g",label=r'$M_1 =10^{12}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M11,fractions*M11,z1,z4),"r",label=r'$M_1 =10^{13}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M111,fractions*M111,z1,z4),"b",label=r'$M_1 =10^{14}M_\odot$')
plt.loglog(fractions,conditional_massFunction(M1111,fractions*M1111,z1,z4),"grey",label=r'$M_1 =10^{15}M_\odot$')
plt.xlim(1e-3,1)
plt.ylim(0.001,100)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel(r'$M_2/M_1$',fontsize=20)
plt.ylabel(r'$dn(M_2/M_1|M_1)/d\log(M_2/M_1)$',fontsize=20)
#plt.legend(fontsize=20,frameon=False)
plt.text(0.002,0.3,r'$z_1=0$',fontsize=20)
plt.text(0.002,0.1,r'$z_2=3$',fontsize=20)
plt.show()



"""

