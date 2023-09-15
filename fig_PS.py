from colossus.cosmology import cosmology as cm
import numpy as np
import matplotlib.pyplot as plt


"""
All the quantities below are taken from the Planck collaboration https://arxiv.org/pdf/1807.06209.pdf and are given in the following units:
---mass         : SOLAR MASSES
---length       : MEGAPARSECS
---time         : SECONDS


Uncomment PowerSpectrumLinearZ0.pdf to obtain Figure 15
Uncomment PBHM1_KLQL.pdf,PBHM2_KLQL.pdf and PBHM3_KLQL.pdf to obtain Figure 16 

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


cosmo = cm.setCosmology('planck18')


"""This block of code defines the white noise induced by PBHs in the power spectrum
"""
ss = 1/aeq
gammma = (Omegam - Omegab)/Omegam
aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
def Dzero(a):
    return (1 + (3*gammma)/(2*aminus)*(ss*a))**aminus
def nbarPBH(f,m):
    return f*rhocrit*(Omegam - Omegab)/m
def kPBH(f,m):
    return ((2*np.pi**2*nbarPBH(f, m))/f)**(1/3)
def kstar(f,m):
    return ((2*np.pi**2*nbarPBH(f, m)))**(1/3)
def OmegaLam(a):
    return OmegaLambda/(OmegaLambda + Omegam/(a**3))
def OmegaM(a):
    return Omegam/(OmegaLambda*(a**3) + Omegam)
def GFF(a):
    return 5/2*OmegaM(a)*a/(OmegaM(a)**(4/7) - OmegaLam(a) + (1 + OmegaM(a)/2)*(1 + OmegaLam(a)/70))



def Psi(k,a,f,m):
    """power spectrum with a cut at kPBH
    """
    ls = []
    for i,wn in enumerate(k):
        if wn<=kPBH(f,m):
            ls.append((cosmo.matterPowerSpectrum(wn,z=1/a-1)/h**3+(f*Dzero(a))**2/nbarPBH(f, m)))
        else:
            ls.append(cosmo.matterPowerSpectrum(wn,z=1/a-1)/h**3)
    return ls

def Pno(k,a,f,m):
    """power spectrum without a cut at kPBH
    """
    return cosmo.matterPowerSpectrum(k,z=1/a-1)+f**2*(Dzero(a)-1)**2/nbarPBH(f, m)


def KLQL(f,m,z):
    return 4/f**(1/3)*(20/(h*m))**(1/3)*(1+26*f*(100/(1+z)))**(-2/3)*h*1e3

rs = np.linspace(0,20,num=100)
ass = np.linspace(0.05,1,num=100)

"""#PSPresentation.pdf
k = np.logspace(-4,2,num=1000)/h
PS = cosmo.matterPowerSpectrum(k,z=0)/h**3

plt.figure(figsize=((10,8)))
plt.loglog(k,PS,'blue',label=r'$\Lambda$CDM',linewidth=3)
plt.hlines(y=10,xmin=min(k),xmax=max(k),colors='red', linestyles='-', lw=3)
plt.xlim(min(k),max(k))
plt.xlabel(r'$k \ [Mpc^{-1}]$',fontsize=22)
plt.ylabel(r'$P(k) \ [Mpc^{3}]$',fontsize=22)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.text(10,30,r"PBHs",color="red",fontsize=24)
plt.text(0.2,20502,r"$\Lambda$CDM",color="blue",fontsize=24)
plt.ylim(10**-3,9*10**4)
plt.show()
"""
"""
plt.figure(figsize=((10,8)))
plt.loglog(k,PS+10,'blue',label=r'$\Lambda$CDM',linewidth=3)
plt.xlim(min(k),max(k))
plt.xlabel(r'$k \ [Mpc^{-1}]$',fontsize=22)
plt.ylabel(r'$P(k) \ [Mpc^{3}]$',fontsize=22)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.ylim(10**-3,9*10**4)
plt.text(0.2,20502,r"$\Lambda$CDM+PBH",color="blue",fontsize=24)
plt.show()
"""



"""#PowerSpectrumLinearZ0.pdf
k = np.logspace(-3,3,num=1000)/h
PS = cosmo.matterPowerSpectrum(k,z=0)/h**3
psiM3 = np.array(Psi(k,1,1e-4,1e10))
psiM1 = np.array(Psi(k,1,0.0003,3*1e5))
psiM2 = np.array(Psi(k,1,1e-5,1e9))
plt.figure(figsize=((10,8)))
plt.loglog(k,PS,'black',label=r'$\Lambda$CDM')
plt.loglog(k,psiM3,'g--',label=r'$M_{PBH}=10^{10} \ M_\odot, \ f_{PBH}=10^{-4}$')
plt.loglog(k,psiM2,'--',color='orange',label=r'$M_{PBH}=10^{9} \ M_\odot, \ f_{PBH}=10^{-5}$')
plt.loglog(k,psiM1,'b--',label=r'$M_{PBH}=3\times 10^{5} \ M_\odot, \ f_{PBH}=3\times 10^{-4}$')
plt.xlim(min(k),max(k))
plt.xlabel(r'$k \ [Mpc^{-1}]$',fontsize=20)
plt.ylabel(r'$P(k) \ [Mpc^{3}]$',fontsize=20)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(frameon=False,fontsize=18, loc='lower left')
plt.savefig("PowerSpectrumLinearZ0.pdf",bbox_inches='tight')
plt.show()
"""

"""#PBHM1_KLQL.pdf
k = np.logspace(-3,4,num=1000)*h
PS = cosmo.matterPowerSpectrum(k,z=10)/h**3
psiM1 = np.array(Psi(k,1/11,0.0003,3*1e5))
plt.figure(figsize=((9,7.5)))
plt.loglog(k,psiM1,'b--',label=r'$M_{PBH}=3\times 10^{5} \ M_\odot, \ f_{PBH}=3\times 10^{-4}$')
plt.xlim(1,5*1e3*h)
plt.xlabel(r'$k \ [Mpc^{-1}]$',fontsize=20)
plt.ylabel(r'$P(k) \ [Mpc^{3}]$',fontsize=20)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.axvline(KLQL(0.0003,3*1e5,10),color="slategrey",linewidth=4)
plt.text(2.3*1e2,10^2,r'$k_{L-QL}(z)$',color="grey",fontsize=26)
plt.legend(frameon=False,fontsize=20, loc='lower left')
plt.savefig("PBHM1_KLQL.pdf",bbox_inches='tight')
plt.show()
"""

"""#PBHM2_KLQL.pdf
k = np.logspace(-3,4,num=1000)*h
PS = cosmo.matterPowerSpectrum(k,z=10)/h**3
psiM2 = np.array(Psi(k,1/11,1e-5,1e9))
plt.figure(figsize=((9,7.5)))
plt.loglog(k,psiM2,'--',color='orange',label=r'$M_{PBH}=10^{9} \ M_\odot, \ f_{PBH}=10^{-5}$')
plt.xlim(1,1e3*h)
plt.xlabel(r'$k \ [Mpc^{-1}]$',fontsize=20)
plt.ylabel(r'$P(k) \ [Mpc^{3}]$',fontsize=20)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.axvline(KLQL(1e-5,1e9,10),color="slategrey",linewidth=4)
plt.text(50,1e2,r'$k_{L-QL}(z)$',color="grey",fontsize=26)
plt.legend(frameon=False,fontsize=20, loc='lower left')
plt.savefig("PBHM2_KLQL.pdf",bbox_inches='tight')
plt.show()

"""

"""#PBHM3_KLQL.pdf
k = np.logspace(-3,4,num=1000)*h
PS = cosmo.matterPowerSpectrum(k,z=10)/h**3
psiM3 = np.array(Psi(k,1/11,1e-4,1e10))
plt.figure(figsize=((9,7.5)))
plt.loglog(k,psiM3,'g--',label=r'$M_{PBH}=10^{10} \ M_\odot, \ f_{PBH}=10^{-4}$')
plt.xlim(0.1,1e3*h)
plt.xlabel(r'$k \ [Mpc^{-1}]$',fontsize=20)
plt.ylabel(r'$P(k) \ [Mpc^{3}]$',fontsize=20)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.axvline(KLQL(1e-4,1e10,10),color="slategrey",linewidth=4)
plt.text(10,1e2,r'$k_{L-QL}(z)$',color="grey",fontsize=26)
plt.legend(frameon=False,fontsize=20, loc='lower left')
plt.savefig("PBHM3_KLQL.pdf",bbox_inches='tight')
plt.show()
"""

