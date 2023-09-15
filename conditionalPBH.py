import time
import numba as nb
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

"""
All the quantities below are taken from the Planck collaboration https://arxiv.org/pdf/1807.06209.pdf and are given in the following units:
---mass         : SOLAR MASSES
---length       : MEGAPARSECS
---time         : SECONDS


With this code we generate Figure 21 and 22 by uncommenting "plotting integrand"
With this code we generate Figure 23 by uncommenting "plottingfinalResults"




"""



start_time = time.time()

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


 

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def radius(M):
    """We define the radius in terms of the mass contained R(M)
    """
    return (3*M/(4*np.pi*rhobar))**(1/3)


 

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def OmegaLam(a):
    return OmegaLambda/(OmegaLambda + Omegam/(a**3))
@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def OmegaM(a):
    return Omegam/(OmegaLambda*(a**3) + Omegam)
@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def GFF(a):
    """This is the growth factor of the linear density field. It was checked against 
    cosmo.growthFactor(z).
    """
    return 5/2*OmegaM(a)*a/(OmegaM(a)**(4/7) - OmegaLam(a) + (1 + OmegaM(a)/2)*(1 + OmegaLam(a)/70))


 


@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def nbarPBH(f,m):
    """Average number density of PBHs
    """
    return f*rhocrit*(Omegam - Omegab)/m


 

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def SigmaAdquadro(M):
    """This is the adiabatic part of the variance. To speed up computation we fit it with:
    cosmo.sigma(radius(M),z=0,filt='sharp-k')**2 in the range of masses 2<M/M_solar<17. Notice that
    there is no time dependence here, since we put it in the barrier omega(z). The coefficients are taken
    from the file fit.py. 
    """
    coeffs = [ 7.35081958e-21, -1.20700528e-18,  6.43660305e-17, -3.34071931e-16,
    -7.52265860e-14,  9.54679057e-13,  9.59419962e-11, -2.30551601e-09,
    -9.48644938e-08,  6.06523482e-06, -1.48279765e-04,  2.06515169e-03,
    -1.76285785e-02,  8.92360517e-02, -3.36119485e-01,  5.74018129e+00] 
    a  = 0
    for i in range(len(coeffs)):
        a += coeffs[-(i+1)]*np.log(M)**i
    return np.exp(a)

def makeExpr():
    X = sym.symbols('X')
    a = 0
    coeffs =[ 3.39385035e-11, -5.22947850e-09,  3.30157119e-07, -1.12720576e-05,
    2.25188781e-04, -2.73781770e-03,  1.77429047e-02, -1.49978288e-01,
    5.53924792e+00]
    for i in range(len(coeffs)):
        a += coeffs[-(i+1)]*sym.log(X)**i
    a = sym.exp(a)
    deriv = sym.diff(a, X)
    return deriv
derivv = makeExpr()
@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def der(x):
    """This is the derivative of the variance. We use a fit with 7 coefficients instead of 15 because
    the results are less sensitive to it. This way the computation is sped up. The fit can be 
    generated with fit.py.
    """
    X = sym.symbols('X')
    deriv = sym.lambdify(X, derivv)
    return deriv(x)


 

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def d(z):
    return 1/(1+z)
@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def omega(z):
    """This is the time dependent barrier. The new growth factor is "a" since for redshifts z>10 we have this limit for 
    both growth factors. See the thesis for details 
    """
    return deltac/d(z) 


 

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def sigmatotQuadro(M,f,m):
    """This function returns the total variance. This is the sum of the adibatic variance and the PBH induced one, 
    where everything is calculated with a sharp-k filter. See the thesis for the calculations.
    """
    gammma = (Omegam - Omegab)/Omegam
    aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
    return (SigmaAdquadro(M)*(1/GFF(a=1))**2 +
            (f**2*2*rhobar/(nbarPBH(f,m)*9*np.pi*M))*(0.56*3*gammma/(2*aminus*aeq))**2)




@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def sigmaLogDerivClean(M,f,m):
    """This returns dlog\sigma/dlogM
    """
    gammma = (Omegam - Omegab)/Omegam
    aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
    return np.abs( 0.5*M/sigmatotQuadro(M,f,m)*
            ( np.abs(der(M)*(1/GFF(a=1))**2) + 
             np.abs(f**2*2*rhobar/(nbarPBH(f,m)*9*M**2*np.pi)*(0.56*3*gammma/(2*aminus*aeq))**2)) 
             )




@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def sigmaQuadroDeriv(M,f,m):
    """This returns dsigma^2/dM
    """
    gammma = (Omegam - Omegab)/Omegam
    aminus = 1/4*((1 + 24*gammma)**(1/2) - 1)
    return np.abs(
              np.abs(der(M)*(1/GFF(a=1))**2) + 
             np.abs(f**2*2*rhobar/(nbarPBH(f,m)*9*M**2*np.pi)*(0.56*3*gammma/(2*aminus*aeq))**2)
    )


 

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def massFunctionPBHST(M,zf,Deltaz,f,m):
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
            ( omega(zf+Deltaz)/sigmatotQuadro(M,f,m)**0.5 )*
            ( np.exp(-0.5*a*omega(zf+Deltaz)**2/sigmatotQuadro(M,f,m)) )*
            ( 1+(a*omega(zf+Deltaz)**2/sigmatotQuadro(M,f,m))**(-p) )*
            ( rhobar/M**2 )*
            (sigmaLogDerivClean(M,f,m) )
            )

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def check(Mi,Mf):
    return Mi<Mf/10


@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def transitionProbPBH(Mi,Mf,zf,Deltaz,f,m):
    """This is the transition probability of a halo to transition from (Mi,zi=zf+Deltaz) to (Mf,zf)
    """
    return (1/np.sqrt(2*np.pi)*
            omega(zf)*(omega(zf+Deltaz)-omega(zf))/omega(zf+Deltaz)*
            (sigmatotQuadro(Mi,f,m)/(sigmatotQuadro(Mf,f,m)*(sigmatotQuadro(Mi,f,m)-sigmatotQuadro(Mf,f,m))))**(3/2)*
            np.exp(-(omega(zf)*sigmatotQuadro(Mi,f,m)-omega(zf+Deltaz)*sigmatotQuadro(Mf,f,m))**2/(2*sigmatotQuadro(Mf,f,m)*sigmatotQuadro(Mi,f,m)*(sigmatotQuadro(Mi,f,m)-sigmatotQuadro(Mf,f,m))))
            )


 
@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def FirstIntegrand(Mi,Mf,zf,Deltaz,f,m):
    """This is a part of the integrand of the second term for the stellar mass density. It is the integrand
    of the inner integral (see thesis). It is the mass function times the transition probabilities.
    """
    Check = check(Mi,Mf)

    transition_prob = transitionProbPBH(Mi,Mf,zf,Deltaz,f,m)

    if (Check[np.isnan(transition_prob)]==0).all():
        transition_prob[np.isnan(transition_prob)] = 0
        transition_prob *= Check

        return (massFunctionPBHST(Mi,zf,Deltaz,f,m)*transition_prob)
    
    else:
        raise(ValueError("One or more NAN is not suppressed by the Mi,Mf condition."))

 

@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def Integrand(Mi,Mf,zf,Deltaz,f,m,ep):
    """This is the complete integrand for the integration in both Mi and Mf.
    """
    return ep*fb*FirstIntegrand(Mi,Mf,zf,Deltaz,f,m)*sigmaQuadroDeriv(Mf,f,m)



@nb.jit(fastmath=True,error_model="numpy",parallel=True)
def rhostar(zf,Mstarf,Deltaz,f,m,ep,acc11,acc12,acc13,acc2,lb):
    """This is the calculation of the stellar mass density. To speed up computation we look at the 
        integrand and see where the most important integration points are, and then we devide the 
        integration domain. This needs to be done for every set of parameters. 
    """
    Mf = np.geomspace(Mstarf/(ep*fb),1e14,num=int(10**acc2))
    
    Mi1 = np.geomspace(lb,1e7,num=int(10**acc11),dtype=np.float64)
    X1,Y = np.meshgrid(Mi1,Mf)
    inn1 = np.trapz(Integrand(X1,Y,zf,Deltaz,f,m,ep),Mi1,axis=1)

    Mi2 = np.geomspace(1e7,1e10,num=int(10**acc12),dtype=np.float64)
    X2,Y = np.meshgrid(Mi2,Mf)
    inn2 = np.trapz(Integrand(X2,Y,zf,Deltaz,f,m,ep),Mi2,axis=1) 

    Mi3 = np.geomspace(1e10,1e13,num=int(10**acc13),dtype=np.float64)
    X3,Y = np.meshgrid(Mi3,Mf)
    inn3 = np.trapz(Integrand(X3,Y,zf,Deltaz,f,m,ep),Mi3,axis=1) 

    return np.trapz(inn1,Mf,axis=0)+ np.trapz(inn2,Mf,axis=0)+ np.trapz(inn3,Mf,axis=0)


"""#finalResults

Deltazs = np.linspace(1,20,num=20)

ls11 = [rhostar(zf=9,Mstarf=1e10,Deltaz=x,f=1,m=4e5,ep=1,acc11=4,acc12=3,acc13=2,acc2=3,lb=1e2)/5e5 for x in Deltazs]
ls12 = [rhostar(zf=9,Mstarf=1e10,Deltaz=x,f=1,m=4e5,ep=1,acc11=4,acc12=3,acc13=2,acc2=3,lb=50)/5e5 for x in Deltazs]
ls13 = [rhostar(zf=9,Mstarf=1e10,Deltaz=x,f=1,m=4e5,ep=1,acc11=4,acc12=3,acc13=2,acc2=3,lb=25)/5e5 for x in Deltazs]

print("-----------------------------------done with 1-----------------------------------")

ls21 = [rhostar(zf=9,Mstarf=1e10,Deltaz=x,f=1,m=2e7,ep=0.1,acc11=4,acc12=3,acc13=2,acc2=3,lb=1e2)/5e5 for x in Deltazs]
ls22 = [rhostar(zf=9,Mstarf=1e10,Deltaz=x,f=1,m=2e7,ep=0.1,acc11=4,acc12=3,acc13=2,acc2=3,lb=50)/5e5 for x in Deltazs]
ls23 = [rhostar(zf=9,Mstarf=1e10,Deltaz=x,f=1,m=2e7,ep=0.1,acc11=4,acc12=3,acc13=2,acc2=3,lb=25)/5e5 for x in Deltazs]


print("-----------------------------------done with 2-----------------------------------")


ls31 = [rhostar(zf=8,Mstarf=10**(10.5),Deltaz=x,f=1,m=1e6,ep=1,acc11=4,acc12=3,acc13=2,acc2=3,lb=1e2)/5e5 for x in Deltazs]
ls32 = [rhostar(zf=8,Mstarf=10**(10.5),Deltaz=x,f=1,m=1e6,ep=1,acc11=4,acc12=3,acc13=2,acc2=3,lb=50)/5e5 for x in Deltazs]
ls33 = [rhostar(zf=8,Mstarf=10**(10.5),Deltaz=x,f=1,m=1e6,ep=1,acc11=4,acc12=3,acc13=2,acc2=3,lb=25)/5e5 for x in Deltazs]


print("-----------------------------------done with 3-----------------------------------")

ls41 = [rhostar(zf=8,Mstarf=10**(10.5),Deltaz=x,f=1,m=4.5e7,ep=0.1,acc11=4,acc12=3,acc13=2,acc2=3,lb=1e2)/5e5 for x in Deltazs]
ls42 = [rhostar(zf=8,Mstarf=10**(10.5),Deltaz=x,f=1,m=4.5e7,ep=0.1,acc11=4,acc12=3,acc13=2,acc2=3,lb=50)/5e5 for x in Deltazs]
ls43 = [rhostar(zf=8,Mstarf=10**(10.5),Deltaz=x,f=1,m=4.5e7,ep=0.1,acc11=4,acc12=3,acc13=2,acc2=3,lb=25)/5e5 for x in Deltazs]

print("printing seconds")
print(ls21)
print(ls22)
print(ls23)


print("printing thirds")
print(ls31)
print(ls32)
print(ls33)
"""

"""#plotting integrand

Deltazs = np.linspace(5,600,num=1000)
inMass1 = np.geomspace(10,10**(11)/10,num=1000)
inMass2 = np.geomspace(10,10**(12)/10,num=1000)
inMass3 = np.geomspace(10,10**(18)/2,num=1000)
plt.figure()

plt.subplot(1,3,1)
plt.title(r"$M_f = 10^{11}, z_f=9$",fontsize=20)
plt.loglog(inMass1,Integrand(inMass1,1e11,9,3,1,4e7,1)/fb,"r",label=r"$\Delta z = 3$")
plt.loglog(inMass1,Integrand(inMass1,1e11,9,1,1,4e7,1)/fb,"b",label=r"$\Delta z = 1$")
plt.loglog(inMass1,Integrand(inMass1,1e11,9,0.5,1,4e7,1)/fb,"g",label=r"$ \Delta z = 0.5$")
plt.loglog(inMass1,Integrand(inMass1,1e11,9,0.1,1,4e7,1)/fb,"m",label=r"$\Delta z = 0.1$")
plt.legend(frameon=False,fontsize=17,loc='lower left')
plt.xlabel(r"$M_i \ (<M_f/2)$",fontsize=20)
plt.ylabel(r'$Integrand \ [Mpc^{-3}]$',fontsize=20)
plt.xlim(min(inMass1),max(inMass1))
plt.ylim(1e-22,1e-8)
plt.text(1e3,1e-9,r"$f_{PBH}m_{PBH} = 5\times 10^{5} \ M_\odot$",fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.subplot(1,3,2)
plt.title(r"$M_f = 10^{12}, z_f=9$",fontsize=20)
plt.loglog(inMass2,Integrand(inMass2,1e12,9,3,1,4e7,1)/fb,"r")
plt.loglog(inMass2,Integrand(inMass2,1e12,9,8,1,4e7,1)/fb,"b")
plt.loglog(inMass2,Integrand(inMass2,1e12,9,15,1,4e7,1)/fb,"g")
plt.loglog(inMass2,Integrand(inMass2,1e12,9,20,1,4e7,1)/fb,"m")
plt.xlabel(r"$M_i \ (<M_f/2)$",fontsize=20)
plt.xlim(min(inMass2),max(inMass2))
plt.ylim(1e-33,1e-16)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.subplot(1,3,3)
plt.title(r"$M_f = 10^{18}, z_f=9$",fontsize=20)
plt.loglog(inMass3,Integrand(inMass3,1e18,9,3,1,4e7,1)/fb,"r")
plt.loglog(inMass3,Integrand(inMass3,1e18,9,8,1,4e7,1)/fb,"b")
plt.loglog(inMass3,Integrand(inMass3,1e18,9,15,1,4e7,1)/fb,"g")
plt.loglog(inMass3,Integrand(inMass3,1e18,9,20,1,4e7,1)/fb,"m")
plt.xlabel(r"$M_i \ (<M_f/2)$",fontsize=20)
plt.xlim(min(inMass3),max(inMass3))
plt.ylim(1e-50,1e-36)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.show()
"""

"""plottingfinalResults
Deltazs = np.linspace(1,20,num=20)
a1 = [0.02032722657859564, 0.04046259585461847, 0.059252705781555025, 0.07561126742749993, 0.08877356360817934, 0.09843987937708136, 0.10477443709043763, 0.10828702299660245, 0.10966092509452735, 0.10959138273143543, 0.1086744706275911, 0.10735636633371737, 0.10593138518708459, 0.10456850911492763, 0.10334767200727506, 0.10229346614087452, 0.10140070909908425, 0.10065112788148144, 0.10002284782988698, 0.09949501590813756]
a2 = [0.0720706832350005, 0.14004684732480352, 0.20401873186073127, 0.2639120062517766, 0.31957278580567455, 0.37082987567897535, 0.41753993191549427, 0.45961847514730725, 0.4970585969625333, 0.5299389342471204, 0.5584225926825491, 0.58274890585702, 0.6032200645909702, 0.6201846647839134, 0.6340200813440409, 0.6451152999497992, 0.6538554700481954, 0.6606090330496593, 0.6657178782256503, 0.6694906245786797]
a3 = [0.008489060835460542, 0.01684629837522803, 0.024487483912709374, 0.03089045408815726, 0.035737573072282584, 0.038977690505201674, 0.04079541566468638, 0.041518162419189236, 0.04150973574066775, 0.04108955489344774, 0.040493057430484036, 0.039868010911078625, 0.03929169757929573, 0.03879437752659286, 0.03837989137224303, 0.038039981533052304, 0.03776259007665946, 0.0375358603640858, 0.03734962911206471, 0.03719569416085167]
a4 = [0.03186389507032694, 0.06178846597411513, 0.08974591067210946, 0.11562595666820259, 0.1392925559000754, 0.1606245684132546, 0.17954350860816218, 0.196029908163588, 0.21012960202098785, 0.22195152319096606, 0.23165897641169161, 0.23945661338511873, 0.24557534748544585, 0.2502572140144539, 0.25374175852745345, 0.256255008639004, 0.2580015468176257, 0.2591597295510701, 0.25987974087815846, 0.260283944679803]


plt.subplot(2,2,1)
plt.plot(Deltazs,a1,"blue",label = r"$f_{PBH}m_{PBH} = 4\times 10^{5}, \epsilon=1$")
plt.text(1.5,1.3,r"$f_{PBH}m_{PBH} = 4\times 10^{5} M_\odot, \epsilon=1$",fontsize=12)
plt.ylabel(r'$\rho_\star (M_\star>10^{10}M_\odot,z_f=9)/\tilde{\rho}_\star$',fontsize=12)
plt.xlabel(r'$\Delta z$',fontsize=12)
plt.axhline(y=1, color='grey', linewidth=2.5)
plt.ylim(1e-3,1.5)
plt.xlim(1,5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.text(2.5,0.8,r'$\rho_\star = \tilde{\rho}_\star$',color="grey",fontsize=12)



plt.subplot(2,2,2)
plt.plot(Deltazs,a2,"dodgerblue",label = r"$f_{PBH}m_{PBH} = 2\times 10^{7}M_\odot, \epsilon=0.1$")
plt.text(1.5,1.3,r"$f_{PBH}m_{PBH} = 2\times 10^{7}M_\odot, \epsilon=0.1$",fontsize=12)
#plt.ylabel(r'$\rho_\star (M_\star>10^{10},z_f=9)/\tilde{\rho}_\star$',fontsize=12)
plt.axhline(y=1, color='grey', linewidth=2.5)
plt.xlabel(r'$\Delta z$',fontsize=14)
plt.xlim(1,5)
plt.ylim(1e-3,1.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(2,2,3)
plt.plot(Deltazs,a3,"blue",label = r"$f_{PBH}m_{PBH} = 10^{6}M_\odot, \epsilon=1$")
plt.text(1.5,1.3,r"$f_{PBH}m_{PBH} = 10^{6}M_\odot, \epsilon=1$",fontsize=12)

plt.xlabel(r'$\Delta z$',fontsize=12)
plt.ylabel(r'$\rho_\star (>10^{10.5}M_\odot,z_f=8)/\tilde{\rho}_\star$',fontsize=12)
plt.axhline(y=1, color='grey', linewidth=2.5)

plt.xlim(1,5)
plt.ylim(1e-3,1.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)



plt.subplot(2,2,4)
plt.plot(Deltazs,a4,"blue",label = r"$f_{PBH}m_{PBH} = 4.5\times 10^{7}M_\odot, \epsilon=0.1$")
plt.text(1.5,1.3,r"$f_{PBH}m_{PBH} = 4.5\times 10^{7}M_\odot, \epsilon=0.1$",fontsize=12)
plt.xlabel(r'$\Delta z$',fontsize=12)
#plt.ylabel(r'$\rho_\star (>10^{10.5},z_f=9)/\tilde{\rho}_\star$',fontsize=12)
plt.axhline(y=1, color='grey', linewidth=2.5)
plt.xlim(1,5)
plt.ylim(1e-3,1.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


"""




