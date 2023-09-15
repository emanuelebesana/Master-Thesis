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

Uncomment totalratesPBH to obtain Figure 20
Uncomment rates_PBH_4.pdf,....,rates_PBH_10.pdf to obtain Figure 19


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



"""This block of code defines the isocurvature part of the Power Spectrum induced by PBHs
"""
gammma = (Omegam - Omegab)/Omegam
aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
def Dzero(z):
    return (1 + (3*gammma)/(2*aminus)*(1+zeq)/(1+z))**aminus
def DzeroQuadro(z):
    return Dzero(z)**2
def nbarPBH(f,m):
    return f*rhocrit*(Omegam - Omegab)/m
def Piso(z,f,m):
    return f**2*Dzero(z)**2/nbarPBH(f,m)



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



def my_SigmaQuadro(M,rs):
    return cosmo.sigma(radius(M),z=rs,filt='sharp-k')**2
def my_Sigma(M,rs):
    return cosmo.sigma(radius(M),z=rs,filt='sharp-k')
def gfQuadro(z):
    return cosmo.growthFactor(z)**2



def sigmaQuadro_PBH(M,rs,f,m):
    """Total variance: adiabatic+PBHs
    """
    return my_SigmaQuadro(M,rs)+Piso(rs,f,m)*2/(9*np.pi)*rhobar/M
def sigma_PBH(M,rs,f,m):
    return sigmaQuadro_PBH(M,rs,f,m)**0.5
def function(M):
    return 1/M


def mine_PBH(DeltaM,z,M,f,m):
    """the differential halo accretion rate with PBHs in it
    """
    return (np.sqrt(2/np.pi)*
            DeltaM/(M+DeltaM)*
            (deltac/(sigma_PBH(M+DeltaM,z,f,m)*(1-sigmaQuadro_PBH(M+DeltaM,z,f,m)/sigmaQuadro_PBH(M,z,f,m))**(3/2)))*
            np.exp(-(deltac**2*(sigmaQuadro_PBH(M,z,f,m)-sigmaQuadro_PBH(M+DeltaM,z,f,m)))/(2*sigmaQuadro_PBH(M+DeltaM,z,f,m)*sigmaQuadro_PBH(M,z,f,m)))*
            np.abs((1+z)*(M+DeltaM)/sigmaQuadro_PBH(M+DeltaM,z,f,m)*0.25*
                   (nd.Derivative(gfQuadro)(z)*nd.Derivative(SigmaQuadro)(M+DeltaM)+2*rhobar/(9*np.pi)*f**2/nbarPBH(f,m)*nd.Derivative(DzeroQuadro)(z)*nd.Derivative(function)(M+DeltaM)))
            )


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

Minitial = 1e11
Minitiall = 1e12
Minitialll = 1e8
Minitiallll = 1e10
Minitialllll = 1e9

x = np.logspace(-3,6,num=1000)

xnew1 = x*Minitial
xnew2 = x*Minitiall
xnew3 = x*Minitialll
xnew4 = x*Minitiallll
xnew5 = x*Minitialllll


 

"""rates_PBH_4.pdf
M4PBH = 1e11
f4PBH = 1e-7
z=8
plt.figure(figsize=((10,8)))
plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{8}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
plt.loglog(x,mine_PBH(xnew5,z,Minitialllll,f4PBH,M4PBH),"g--")
plt.loglog(x,mine_PBH(xnew3,z,Minitialll,f4PBH,M4PBH),'--',color="purple")
plt.loglog(x,mine_PBH(xnew4,z,Minitiallll,f4PBH,M4PBH),"r--")
plt.loglog(x,mine_PBH(xnew1,z,Minitial,f4PBH,M4PBH),"b--")
plt.loglog(x,mine_PBH(xnew2,z,Minitiall,f4PBH,M4PBH),'--',color="grey")
plt.xlim(1e0,1e3)
plt.ylim(1e-2,10)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(frameon=False,fontsize=20)
plt.text(3,4,r'$z=8, \ f_{PBH} m_{PBH}=10^{4}$',fontsize=20)
plt.text(1e2,0.3,r'$Solid: \ \Lambda CDM$',fontsize=20)
plt.text(1e2,0.1,r'$Dashed: \ PBH$',fontsize=20)
plt.savefig("rates_PBH_4.pdf",bbox_inches='tight')
plt.show()
"""

"""rates_PBH_6.pdf
M6PBH = 1e10
f6PBH = 1e-4
z=8
plt.figure(figsize=((10,8)))
plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{8}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
plt.loglog(x,mine_PBH(xnew5,z,Minitialllll,f6PBH,M6PBH),"g--")
plt.loglog(x,mine_PBH(xnew3,z,Minitialll,f6PBH,M6PBH),'--',color="purple")
plt.loglog(x,mine_PBH(xnew4,z,Minitiallll,f6PBH,M6PBH),"r--")
plt.loglog(x,mine_PBH(xnew1,z,Minitial,f6PBH,M6PBH),"b--")
plt.loglog(x,mine_PBH(xnew2,z,Minitiall,f6PBH,M6PBH),'--',color="grey")
plt.xlim(1e0,1e3)
plt.ylim(1e-2,10)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(frameon=False,fontsize=20)
plt.text(2,2,r'$z=8, \ f_{PBH}m_{PBH}=10^{6}$',fontsize=20)
#plt.text(1e2,0.3,r'$Solid: \ \Lambda CDM$',fontsize=20)
#plt.text(1e2,0.1,r'$Dashed: \ PBH$',fontsize=20)
plt.savefig("rates_PBH_6.pdf",bbox_inches='tight')
plt.show()
"""

"""rates_PBH_5.pdf
f5PBH = 1e-6
M5PBH = 1e11
z=8
plt.figure(figsize=((10,8)))
plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{8}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
plt.loglog(x,mine_PBH(xnew5,z,Minitialllll,f5PBH,M5PBH),"g--")
plt.loglog(x,mine_PBH(xnew3,z,Minitialll,f5PBH,M5PBH),'--',color="purple")
plt.loglog(x,mine_PBH(xnew4,z,Minitiallll,f5PBH,M5PBH),"r--")
plt.loglog(x,mine_PBH(xnew1,z,Minitial,f5PBH,M5PBH),"b--")
plt.loglog(x,mine_PBH(xnew2,z,Minitiall,f5PBH,M5PBH),'--',color="grey")
plt.xlim(1e0,1e3)
plt.ylim(1e-2,10)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(frameon=False,fontsize=20)
plt.text(2,2,r'$z=8, \ f_{PBH}m_{PBH}=10^{5}$',fontsize=20)
#plt.text(1e2,0.3,r'$Solid: \ \Lambda CDM$',fontsize=20)
#plt.text(1e2,0.1,r'$Dashed: \ PBH$',fontsize=20)
plt.savefig("rates_PBH_5.pdf",bbox_inches='tight')
plt.show()
"""

"""rates_PBH_7.pdf
f7PBH = 1
M7PBH = 1e7
z=8
plt.figure(figsize=((10,8)))
plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{8}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
plt.loglog(x,mine_PBH(xnew5,z,Minitialllll,f7PBH,M7PBH),"g--")
plt.loglog(x,mine_PBH(xnew3,z,Minitialll,f7PBH,M7PBH),'--',color="purple")
plt.loglog(x,mine_PBH(xnew4,z,Minitiallll,f7PBH,M7PBH),"r--")
plt.loglog(x,mine_PBH(xnew1,z,Minitial,f7PBH,M7PBH),"b--")
plt.loglog(x,mine_PBH(xnew2,z,Minitiall,f7PBH,M7PBH),'--',color="grey")
plt.xlim(1e0,1e4)
plt.ylim(1e-2,10)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(frameon=False,fontsize=20)
plt.text(2,2,r'$z=8, \ f_{PBH}m_{PBH}=10^{7}$',fontsize=20)
#plt.text(1e2,0.3,r'$Solid: \ \Lambda CDM$',fontsize=20)
#plt.text(1e2,0.1,r'$Dashed: \ PBH$',fontsize=20)
plt.savefig("rates_PBH_7.pdf",bbox_inches='tight')
plt.show()
"""

"""rates_PBH_8.pdf
f8PBH = 1
M8PBH = 1e8
z=8
plt.figure(figsize=((10,8)))
#plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{8}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
plt.loglog(x,mine_PBH(xnew5,z,Minitialllll,f8PBH,M8PBH),"g--")
#plt.loglog(x,mine_PBH(xnew3,z,Minitialll,f8PBH,M8PBH),'--',color="purple")
plt.loglog(x,mine_PBH(xnew4,z,Minitiallll,f8PBH,M8PBH),"r--")
plt.loglog(x,mine_PBH(xnew1,z,Minitial,f8PBH,M8PBH),"b--")
plt.loglog(x,mine_PBH(xnew2,z,Minitiall,f8PBH,M8PBH),'--',color="grey")
plt.xlim(1e0,1e5)
plt.ylim(1e-2,10)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(frameon=False,fontsize=20)
plt.text(2,2,r'$z=8, \ f_{PBH}m_{PBH}=10^{8}$',fontsize=20)
#plt.text(1e2,0.3,r'$Solid: \ \Lambda CDM$',fontsize=20)
#plt.text(1e2,0.1,r'$Dashed: \ PBH$',fontsize=20)
plt.savefig("rates_PBH_8.pdf",bbox_inches='tight')
plt.show()
"""

"""rates_PBH_9.pdf
f9PBH = 1e-3
M9PBH = 1e12
z=8
plt.figure(figsize=((10,8)))
#plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{8}M_\odot$')
#plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
#plt.loglog(x,mine_PBH(xnew5,z,Minitialllll,f9PBH,M9PBH),"g--")
#plt.loglog(x,mine_PBH(xnew3,z,Minitialll,f9PBH,M9PBH),'--',color="purple")
plt.loglog(x,mine_PBH(xnew4,z,Minitiallll,f9PBH,M9PBH),"r--")
plt.loglog(x,mine_PBH(xnew1,z,Minitial,f9PBH,M9PBH),"b--")
plt.loglog(x,mine_PBH(xnew2,z,Minitiall,f9PBH,M9PBH),'--',color="grey")
plt.xlim(1e0,1e5)
plt.ylim(1e-2,10)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(frameon=False,fontsize=20)
plt.text(2,2,r'$z=8, \ f_{PBH}m_{PBH}=10^{9}$',fontsize=20)
#plt.text(1e2,0.3,r'$Solid: \ \Lambda CDM$',fontsize=20)
#plt.text(1e2,0.1,r'$Dashed: \ PBH$',fontsize=20)
plt.savefig("rates_PBH_9.pdf",bbox_inches='tight')
plt.show()
"""

"""rates_PBH_10.pdf
f10PBH = 1e-3
M10PBH = 1e12
z=8
plt.figure(figsize=((10,8)))
#plt.loglog(x,differential_accretionRate(xnew3,z,Minitialll),"purple",label=r'$M_2=10^{8}M_\odot$')
#plt.loglog(x,differential_accretionRate(xnew5,z,Minitialllll),"g",label=r'$M_2=10^{9} M_\odot$')
#plt.loglog(x,differential_accretionRate(xnew4,z,Minitiallll),"r",label=r'$M_2=10^{10}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew1,z,Minitial),"b",label=r'$M_2=10^{11}M_\odot$')
plt.loglog(x,differential_accretionRate(xnew2,z,Minitiall),"grey",label=r'$M_2=10^{12}M_\odot$')
#plt.loglog(x,mine_PBH(xnew5,z,Minitialllll,f10PBH,M10PBH),"g--")
#plt.loglog(x,mine_PBH(xnew3,z,Minitialll,f10PBH,M10PBH),'--',color="purple")
#plt.loglog(x,mine_PBH(xnew4,z,Minitiallll,f10PBH,M10PBH),"r--")
plt.loglog(x,mine_PBH(xnew1,z,Minitial,f10PBH,M10PBH),"b--")
plt.loglog(x,mine_PBH(xnew2,z,Minitiall,f10PBH,M10PBH),'--',color="grey")
plt.xlim(1e0,1e3)
plt.ylim(1e-2,10)
plt.ylabel(r'$d^2R/d\log\Delta M \ d\log a$',fontsize=20)
plt.xlabel(r'$\Delta M/M_2$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(frameon=False,fontsize=20)
plt.text(2,2,r'$z=8, \ f_{PBH}m_{PBH}=10^{10}$',fontsize=20)
#plt.text(1e2,0.3,r'$Solid: \ \Lambda CDM$',fontsize=20)
#plt.text(1e2,0.1,r'$Dashed: \ PBH$',fontsize=20)
plt.savefig("rates_PBH_10.pdf",bbox_inches='tight')
plt.show()
"""

"""totalratesPBH
def totalAccretionRate_PBH_9(z,M,f,m):
    if (M>=1e10 and M<=5e10):
        return int.quad(mine_PBH,M,M*2500,args=(z,M,f,m))[0]    
    if (M>5e10 and M<=1e11):
        return int.quad(mine_PBH,M,M*800,args=(z,M,f,m))[0]
    if (M>1e11 and M<=5e11):
        return int.quad(mine_PBH,M,M*350,args=(z,M,f,m))[0]
    if (M>5e11 and M<=1e12):
        return int.quad(mine_PBH,M,M*60,args=(z,M,f,m))[0]
def totalAccretionRate_PBH_8(z,M,f,m):
    if M <= 5e9:
        return int.quad(mine_PBH,M,M*4000,args=(z,M,f,m))[0]
    if (M>5e9 and M<=1e10):
        return int.quad(mine_PBH,M,M*500,args=(z,M,f,m))[0]
    if (M>1e10 and M<=5e10):
        return int.quad(mine_PBH,M,M*250,args=(z,M,f,m))[0]    
    if (M>5e10 and M<=1e11):
        return int.quad(mine_PBH,M,M*70,args=(z,M,f,m))[0]
    if (M>1e11 and M<=5e11):
        return int.quad(mine_PBH,M,M*35,args=(z,M,f,m))[0]
    if (M>5e11 and M<=1e12):
        return int.quad(mine_PBH,M,M*8,args=(z,M,f,m))[0]
def totalAccretionRate_PBH_7(z,M,f,m):
    if M <= 5e8:
        return int.quad(mine_PBH,M,M*4000,args=(z,M,f,m))[0]
    if (M>5e8 and M<=1e9):
        return int.quad(mine_PBH,M,M*700,args=(z,M,f,m))[0]
    if (M>1e9 and M<=5e9):
        return int.quad(mine_PBH,M,M*250,args=(z,M,f,m))[0]    
    if (M>5e9 and M<=1e10):
        return int.quad(mine_PBH,M,M*70,args=(z,M,f,m))[0]
    if (M>1e10 and M<=5e10):
        return int.quad(mine_PBH,M,M*40,args=(z,M,f,m))[0]
    if (M>5e10 and M<=1e11):
        return int.quad(mine_PBH,M,M*10,args=(z,M,f,m))[0]
    if (M>1e11 and M<=1e12):
        return int.quad(mine_PBH,M,M*4,args=(z,M,f,m))[0]
def totalAccretionRate_PBH_6(z,M,f,m):
    if M <= 5e8:
        return int.quad(mine_PBH,M,M*500,args=(z,M,f,m))[0]
    if (M>5e8 and M<=1e9):
        return int.quad(mine_PBH,M,M*100,args=(z,M,f,m))[0]
    if (M>1e9 and M<=5e9):
        return int.quad(mine_PBH,M,M*50,args=(z,M,f,m))[0]    
    if (M>5e9 and M<=1e10):
        return int.quad(mine_PBH,M,M*15,args=(z,M,f,m))[0]
    if (M>1e10 and M<=5e10):
        return int.quad(mine_PBH,M,M*10,args=(z,M,f,m))[0]
    if (M>5e10 and M<=1e11):
        return int.quad(mine_PBH,M,M*4,args=(z,M,f,m))[0]
    if (M>1e11 and M<=1e12):
        return int.quad(mine_PBH,M,M*4,args=(z,M,f,m))[0]
def totalAccretionRate_PBH_5(z,M,f,m):
    if M <= 5e8:
        return int.quad(mine_PBH,M,M*100,args=(z,M,f,m))[0]
    if (M>5e8 and M<=1e9):
        return int.quad(mine_PBH,M,M*40,args=(z,M,f,m))[0]
    if (M>1e9 and M<=5e9):
        return int.quad(mine_PBH,M,M*20,args=(z,M,f,m))[0]    
    if (M>5e9 and M<=1e10):
        return int.quad(mine_PBH,M,M*8,args=(z,M,f,m))[0]
    if (M>1e10 and M<=5e10):
        return int.quad(mine_PBH,M,M*8,args=(z,M,f,m))[0]
    if (M>5e10 and M<=1e11):
        return int.quad(mine_PBH,M,M*8,args=(z,M,f,m))[0]
    if (M>1e11 and M<=1e12):
        return int.quad(mine_PBH,M,M*3,args=(z,M,f,m))[0]
def totalAccretionRate_PBH_4(z,M,f,m):
    if M <= 5e8:
        return int.quad(mine_PBH,M,M*60,args=(z,M,f,m))[0]
    if (M>5e8 and M<=1e9):
        return int.quad(mine_PBH,M,M*60,args=(z,M,f,m))[0]
    if (M>1e9 and M<=5e9):
        return int.quad(mine_PBH,M,M*40,args=(z,M,f,m))[0]    
    if (M>5e9 and M<=1e10):
        return int.quad(mine_PBH,M,M*20,args=(z,M,f,m))[0]
    if (M>1e10 and M<=5e10):
        return int.quad(mine_PBH,M,M*10,args=(z,M,f,m))[0]
    if (M>5e10 and M<=1e11):
        return int.quad(mine_PBH,M,M*4,args=(z,M,f,m))[0]
    if (M>1e11 and M<=1e12):
        return int.quad(mine_PBH,M,M*2,args=(z,M,f,m))[0]
def totalAccretionRate(z,M):
    if M <= 1e9:
        return int.quad(differential_accretionRate,M,M*300,args=(z,M))[0]
    if (M>1e9 and M<=1e10):
        return int.quad(differential_accretionRate,M,M*100,args=(z,M))[0]
    if (M>1e10 and M<=1e11):
        return int.quad(differential_accretionRate,M,M*13,args=(z,M))[0]    
    if (M>1e11 and M<=1e12):
        return int.quad(differential_accretionRate,M,M*10,args=(z,M))[0]
    

initialMasses8 = np.logspace(9,12,num=100)
initialMasses9 = np.logspace(10,12,num=100)
initialMasses = np.logspace(8,12,num=100)

def vals_PBH_9(redshift,x,f,m):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate_PBH_9(redshift,i,f,m))
    return np.array(ls)
def vals_PBH_8(redshift,x,f,m):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate_PBH_8(redshift,i,f,m))
    return np.array(ls)
def vals_PBH_7(redshift,x,f,m):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate_PBH_7(redshift,i,f,m))
    return np.array(ls)
def vals_PBH_6(redshift,x,f,m):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate_PBH_6(redshift,i,f,m))
    return np.array(ls)
def vals_PBH_5(redshift,x,f,m):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate_PBH_5(redshift,i,f,m))
    return np.array(ls)
def vals_PBH_4(redshift,x,f,m):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate_PBH_4(redshift,i,f,m))
    return np.array(ls)
def vals_lcdm(redshift,x):
    ls =[]
    for i in x:
        ls.append(totalAccretionRate(redshift,i))
    return np.array(ls)


f9PBH = 1
M9PBH = 1e9
f8PBH = 1
M8PBH = 1e8
f7PBH = 1
M7PBH = 1e7
f6PBH = 1
M6PBH = 1e6
f5PBH = 1
M5PBH = 1e5
f4PBH = 1
M4PBH = 1e4

plt.figure(figsize=((10,8)))
plt.loglog(initialMasses9,vals_PBH_9(8,initialMasses9,f9PBH,M9PBH)/initialMasses9,"b--",label=r"$PBH, f_{PBH}m_{PBH}=10^{9}$")
plt.loglog(initialMasses8,vals_PBH_8(8,initialMasses8,f8PBH,M8PBH)/initialMasses8,"r--",label=r"$f_{PBH}m_{PBH}=10^{8}$")
plt.loglog(initialMasses,vals_PBH_7(8,initialMasses,f7PBH,M7PBH)/initialMasses,"g--",label=r"$f_{PBH}m_{PBH}=10^{7}$")
plt.loglog(initialMasses,vals_PBH_6(8,initialMasses,f6PBH,M6PBH)/initialMasses,"m--",label=r"$f_{PBH}m_{PBH}=10^{6}$")
plt.loglog(initialMasses,vals_PBH_5(8,initialMasses,f5PBH,M5PBH)/initialMasses,'--',color="orange",label=r"$f_{PBH}m_{PBH}=10^{5}$")
plt.loglog(initialMasses,vals_PBH_4(8,initialMasses,f4PBH,M4PBH)/initialMasses,'--',color="cyan",label=r"$f_{PBH}m_{PBH}=10^{4}$")
plt.loglog(initialMasses,vals_lcdm(8,initialMasses)/initialMasses,"black",label=r'$\Lambda CDM$')
plt.xlim(min(initialMasses),max(initialMasses))
plt.ylim(0.01,1e3)
plt.legend(frameon = False, fontsize=12)
plt.xlabel(r'$M_2 \ [M_\odot]$',fontsize=20)
plt.ylabel(r'$dR/d\log a \ [\Delta M/M_2]$',fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.savefig("totalAccRate_Results_PBH.pdf",bbox_inches='tight')
plt.show()
"""



