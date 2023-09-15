# Master-Thesis
I used these Python programs in order to make plots and find results form my master's thesis. A subset of these.

The following two programs implement Excursion Set Theory as taken from [Zentner 2006](https://arxiv.org/abs/astro-ph/0611454):

- **conditional_MF.py**: Implements the halo conditional mass function
- **accretion_rates.py**: Implements the differential and total accretion rates for halos

The following programs are mostly used to produce the results found in Chapter 5 of my thesis. At the start of each file there are instructions on how to draw the corresponding Figures.

- **PBH_NoTrans_Gauss.py**: This code implements the calculation of the comoving cumulative stellar mass density without taking into account the transition probability, like in [Liu & Bromm 2022](https://arxiv.org/abs/2208.13178), for a Gaussian smoothing filter
- **PBH_NoTrans_RealTop.py**: Like above, but for a real space tophat window
- **PBH_NoTrans_SharpK.py**: Like above, but for a Fourier space tophat window
- **rates_PBH.py**: Calculates the differential and total halo accretion rates including the Poissonian effect of primordial black holes in the power spectrum
- **conditional_PBH.py**: Calculates the contribution of the accreting halos to the stellar mass density in the presence of primordial black holes, basically the main results of Chapter 5
- **fit.py**: Script used to fit the variances with PBHs, in order to speed up computation
- **PBH_peak_Acc.py**: It implements the code to find the results for the Seed effect
