import numpy as np
import matplotlib.pyplot as plt


"""
Uncomment PBHtoHALOmassfunctions to obtain Figure 24
"""

x = np.linspace(0.1,1e13,num=1000000)

"""Parameters of the PBH initial mass function (see thesis)
"""
B = 10**(-1.9)
mB = 4e8

def fb(x):
    """PBH initial mass function
    """
    return 1/x**2 * np.exp(-(1-x/mB)**2/B)


"""Parameters of the halo mass function at z=10
"""
C = 10**(-2)
mH = (300)*mB


def fh(x):
    """Halo mass function at z=10
    """
    return 1/x**2 * np.exp(-(1-x/mH)**2/B)


"""#ResultsAccretionPBHs
def integration(f):
    int1 = np.linspace(1.6e9,1.89e9,num=10000000,dtype=np.float64)
    int2 = np.linspace(1.9e9,1.998e9,num=10000000,dtype=np.float64)
    int3 = np.linspace(1.999e9,2.001e9,num=10000000,dtype=np.float64)
    int4 = np.linspace(2.002e9,2.3e9,num=10000000,dtype=np.float64)
    int5 = np.linspace(2.301e9,3e9,num=10000000,dtype=np.float64)
    return np.trapz(f(int1),int1)+np.trapz(f(int2),int2)+np.trapz(f(int3),int3)+np.trapz(f(int4),int4)+np.trapz(f(int5),int5)

def integration(f):
    int1 = np.linspace(4e11,5.89e11,num=10000000,dtype=np.float64)
    int2 = np.linspace(5.9e11,5.998e11,num=10000000,dtype=np.float64)
    int3 = np.linspace(5.999e11,6.001e11,num=10000000,dtype=np.float64)
    int4 = np.linspace(6.002e11,6.299e11,num=10000000,dtype=np.float64)
    int5 = np.linspace(6.301e11,8e11,num=10000000,dtype=np.float64)
    return np.trapz(f(int1),int1)+np.trapz(f(int2),int2)+np.trapz(f(int3),int3)+np.trapz(f(int4),int4)+np.trapz(f(int5),int5)

"""

"""#PBHtoHALOmassfunctions
plt.figure()
plt.loglog(x,fb(x),color="dodgerblue",label="PBHs")
plt.loglog(x,fh(x),color="indianred",label="Halos")
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
plt.ylabel(r"$dn/dM \ [M_\odot \ Mpc^{-3}]$",fontsize = 20)
plt.xlabel(r"$M  \ [M_\odot]$",fontsize = 20)
plt.xlim(1e5,1e12)
plt.ylim(1e-50,1e-10)
plt.vlines(mB,ymax=fb(mB),ymin=0,color="black",ls="--")
plt.legend(frameon=False,fontsize=20)
plt.text(1e6,1e-14,r"$B = 10^{-1.9}$",fontsize=19)
plt.text(1e6,1e-16,r"$m_B = 4 \times 10^9 M_\odot$", fontsize = 19)
plt.text(2e8,1e-49,r"$m_B$",fontsize = 18)
#plt.axvline(10**(10.5)/(0.1*0.15),color="grey")
plt.show()
"""





