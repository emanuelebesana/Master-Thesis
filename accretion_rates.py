from colossus.cosmology import cosmology as cm
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
import scipy.integrate as int
"""
All the quantities below are taken from the Planck collaboration https://arxiv.org/pdf/1807.06209.pdf and are given in the following units:
---mass         : SOLAR MASSES
---length       : MEGAPARSECS
---time         : SECONDS

Uncomment accretionRate1.pdf and accretionRate2.pdf to obtain Figure 8
Uncomment totalAccRate.pdf to obtain Figure 9
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
def derivOmega(rs):
    """Derivative d\omega(a)/da
    """
    return deltac*cosmo.growthFactor(rs,derivative=1)/(cosmo.growthFactor(rs)**2)

def SigmaQuadro(M):
    """Variance obtained with COLOSSUS package
    """
    return cosmo.sigma(radius(M),z=0,filt='sharp-k')**2
def Sigma(M):
    return cosmo.sigma(radius(M),z=0,filt='sharp-k')



def differential_accretionRate(DeltaM,z,M):
    """Note: everything is a function of the DeltaM, in the sense that we are calculating the accretion rate of an initial
             halo of mass M, as a function of accreted matter DeltaM. Therefore M1 = DeltaM + M. where M is fixed. 
             We are calculating the rate at which a halo with mass M transits to a halo with mass between M and M+DeltaM. 
    """
    return ( np.sqrt(2/np.pi)*
            DeltaM/(M+DeltaM)*
            (omega(z)/(Sigma(M+DeltaM)*(1-SigmaQuadro(M+DeltaM)/SigmaQuadro(M))**(3/2)))*
            np.exp(-(omega(z)**2*(SigmaQuadro(M)-SigmaQuadro(M+DeltaM)))/(2*SigmaQuadro(M+DeltaM)*SigmaQuadro(M)))*
            np.abs(derivOmega(z)*(1+z)/omega(z))*
            np.abs(cosmo.sigma(radius(M+DeltaM),derivative=True,filt='sharp-k')/3) )               

def totalAccretionRateIntegrand(DeltaM,z,M):
    """This function is the same as the one above. But I leave it here so that I can make some trials.
    """
    return ( np.sqrt(2/np.pi)*DeltaM/(M+DeltaM)*
            (omega(z)/(Sigma(M+DeltaM)*(1-SigmaQuadro(M+DeltaM)/SigmaQuadro(M))**(3/2)))*
            np.exp(-((omega(z)**2)*(SigmaQuadro(M)-SigmaQuadro(M+DeltaM)))/(2*SigmaQuadro(M+DeltaM)*SigmaQuadro(M)))*
            np.abs(derivOmega(z)*(1+z)/omega(z))*
            np.abs(cosmo.sigma(radius(M+DeltaM),derivative=True,filt='sharp-k')/3) )  

def totalAccretionRate(z,M):
    """Integral of the differential accretion rate
    """
    if M <= 1e9:
        return int.quad(totalAccretionRateIntegrand,M,M*300,args=(z,M))[0]
    if (M>1e9 and M<=1e10):
        return int.quad(totalAccretionRateIntegrand,M,M*100,args=(z,M))[0]
    if (M>1e10 and M<=1e11):
        return int.quad(totalAccretionRateIntegrand,M,M*13,args=(z,M))[0]    
    if (M>1e11 and M<=1e12):
        return int.quad(totalAccretionRateIntegrand,M,M*10,args=(z,M))[0]


"""accretionRate1.pdf
Minitial = 1e11
Minitiall = 1e12
Minitialll = 1e13
Minitiallll = 1e10
Minitialllll = 1e9

x = np.logspace(-3,6,num=1000)

xnew1 = x*Minitial
xnew2 = x*Minitiall
xnew3 = x*Minitialll
xnew4 = x*Minitiallll
xnew5 = x*Minitialllll

z=0
plt.figure(figsize=((10,8)))
plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{13}M_\odot$')
plt.xlim(1e-3,1e6)
plt.ylim(1e-2,150)
plt.legend(fontsize=20,frameon=False)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.text(10,10,r'$z=0$',fontsize=20)
plt.show()
"""

"""accretionRate2.pdf
Minitial = 1e11
Minitiall = 1e12
Minitialll = 1e13
Minitiallll = 1e10
Minitialllll = 1e9

x = np.logspace(-3,6,num=1000)

xnew1 = x*Minitial
xnew2 = x*Minitiall
xnew3 = x*Minitialll
xnew4 = x*Minitiallll
xnew5 = x*Minitialllll

z=0
plt.figure(figsize=((10,8)))
plt.loglog(x,x*differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,x*differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,x*differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,x*differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
plt.loglog(x,x*differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{13}M_\odot$')
plt.xlim(1e-3,1e6)
plt.ylim(1e-2,2*1e3)
plt.ylabel(r'$(\Delta M/M_2)d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.text(1e-2,3,r'$z=0$',fontsize=20)
plt.show()
"""

"""totalAccRate.pdf
z1 = 0
z2 = 0.5
z3 = 1


initialMasses = np.logspace(9,13,num=1000)
def vals(redshift,x):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate(redshift,i))
    return np.array(ls)


plt.figure(figsize=((10,8)))
plt.loglog(initialMasses,vals(z1,initialMasses)/initialMasses,"g",label=r"$z = 0$")
plt.loglog(initialMasses,vals(z2,initialMasses)/initialMasses,"r",label=r'$z = 0.5$')
plt.loglog(initialMasses,vals(z3,initialMasses)/initialMasses,"b",label=r"$z=1$")
plt.xlim(min(initialMasses),max(initialMasses))
plt.ylim(1,4000)
plt.legend(frameon = False, fontsize=20)
plt.xlabel(r'$M_2 \ [M_\odot]$',fontsize=20)
plt.ylabel(r'$dR/d\log a \ [\Delta M/M_2]$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()
"""

