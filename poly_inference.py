import numpy as np
import scipy.stats
import emcee
import faulthandler
faulthandler.enable()

import os
import sys
nseospy_tk = os.getenv('NSEOSPY_TK')
sys.path.insert(0, nseospy_tk)

from multiprocessing import Pool
from nseospy import setup_den_tot
from nseospy import setup_den_tov
from nseospy import tovsolver
from nseospy import build_eos
from interface import generate_polycore_EOS

np.random.seed(10)

# Array of central densities (baryons fm^-3) for EOS (logarithmic spacing)
# Solves for pressure at 200 density points and interpolates
Den = setup_den_tot(model="n-d-log", var2=0.0, n_min=2.e-7, n_max=1.7, npt=200)

# List of EoS params passed to TOV solver
# The following are the fixed symmetric parameters used in Will's model
# E0:      -15.926198132940234
# nast:      0.1562
# K0:      239.6642289121857
# Q0:      -362.46919971374865
# Z0:      1465.6538274573425
list_param_sly4 = np.array([-15.93, 0.1562, 239.6, -362.5, 1466, 32.01,  46.00, -120.0,  350.0,  -690.0, 1.0000, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 6.90, 0.00], dtype=np.float64)
list_param_qyc = np.array([250, 0.3])
list_param_pair = np.array([1.573e-03, 3.105, 8.551e-02, 1.386, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
list_ns_massRef =  np.array([8.67, 68.0, 30.0], dtype=np.float64) # SLy5
list_aFsFlag = np.array([4, 1, 0], dtype=np.int32)
list_aInit = np.array([60.0, 0.167, 0.4, 0.0], dtype=np.float64)
list_flags_ns = np.array([0, 1], dtype=np.int32)
# list_fs_param = [0.967, 1.039, 1.257, 1.116, 0.980, 1.000, 1.000, 1.000]
list_fs_param = 'SLy5'
list_ns_outFileFormat = np.array([0, 0, 0, 0], dtype=np.int32)

# Define physically valid ranges of parameters to be varied
n = 20
e_sym_range = np.linspace(26.83, 38.71, n)
l_sym_range = np.linspace(9.9, 64, n)
k_sym_range = np.linspace(-235, 213, n)
q_sym_range = np.linspace(-86, 846, n)
z_sym_range = np.linspace(-1450, -5, n)

param_ranges = np.array([np.max(e_sym_range)-np.min(e_sym_range), np.max(l_sym_range)-np.min(l_sym_range),
                         np.max(k_sym_range)-np.min(k_sym_range), np.max(q_sym_range)-np.min(q_sym_range),
                         np.max(z_sym_range)-np.min(z_sym_range)])

# Generate the radius at 1.4 solar masses for given polytropic parameters using Will's model

e_sym= 32.0
l_sym = 50.0
k_sym = 0.0
n1 = 0.5
n2 = 0.5

# Generate the EOS by calling the interface for Will Newton's code
Eos = generate_polycore_EOS(e_sym, l_sym, k_sym, n1, n2) #Parameters are: J, L, Ksym, n1, n2

# Use the TOV solver from nseospy in exactly the same way as for nseospy's EOSs
NSnbc = setup_den_tov( model="n-lin", den_step = 0.01, NPOINT = 400 )
atov = tovsolver(NSnbc, Eos, domi = 'y', dotd = 'y')

difference_array = 1.4 - atov.m
nearest_above = np.where(difference_array < 0, difference_array, -np.inf).argmax()
nearest_below = np.where(difference_array > 0, difference_array, np.inf).argmin()
rad_at_1_point_4_msun_poly = atov.rad[nearest_below] + (1.4-atov.m[nearest_below])*((atov.rad[nearest_above]-atov.rad[nearest_below])/(atov.m[nearest_above]-atov.m[nearest_below]))


def Sk_params(J,L,Ksym):
    hbar=197.3269804
    m=939.56542052
    h2over2m=(hbar*hbar)/(2*m)
    two_thirds=2/3
    five_thirds=5/3
    five_ninths=5/9

    n0=0.1562
    E0=-15.926198132940234
    K0=239.6642289121857
    t1=301.8208
    t2=-273.2827
    x1=-0.3622
    x2=-0.4105
    a3=1/3
    a4=1

    n23 = n0**two_thirds
    n53 = n0**five_thirds
    A3 = n0**a3
    A4 = n0**a4

    ck = 0.6*((1.5*np.pi*np.pi)**two_thirds)
    CKE = h2over2m*ck
    DKE = five_ninths*CKE
    C12 = ck*0.125*0.5*(3.0*t1 + 4.0*t2*x2 + 5.0*t2)
    D12 = five_ninths*ck*0.125*(-3.0*t1*x1 + 5.0*t2*x2 + 4.0*t2)

    E0p = (E0 - CKE*n23 - C12*n53)/n0
    Jp  = (J  - DKE*n23 - D12*n53)/n0
    L0p = (  - 2.0*CKE*n23 - 5.0*C12*n53)/n0
    Lp  = (L - 2.0*DKE*n23 - 5.0*D12*n53)/n0
    K0p   = (K0   + 2.0*CKE*n23 - 10.0*C12*n53)/n0
    Ksymp = (Ksym + 2.0*DKE*n23 - 10.0*D12*n53)/n0

    C0 = 9.0*E0p*(a3+1.0)*(a4+1.0) - 3.0*L0p*(a3+a4+1.0) + K0p
    C0 = C0/(9.0*a3*a4)
    C3 = 9.0*E0p*(a4+1.0) - 3.0*L0p*(a4+1.0) + K0p
    C3 = C3/(9.0*A3*(a3*a3-a3*a4))
    C4 = 9.0*E0p*(a3+1.0) - 3.0*L0p*(a3+1.0) + K0p
    C4 = -C4/(9.0*A4*(a3*a4-a4*a4))
    D0 = 9.0*Jp*(a3+1.0)*(a4+1.0) - 3.0*Lp*(a3+a4+1.0) + Ksymp
    D0 = D0/(9.0*a3*a4)
    D3 = 9.0*Jp*(a4+1.0) - 3.0*Lp*(a4+1.0) + Ksymp
    D3 = D3/(9.0*A3*(a3*a3-a3*a4))
    D4 = 9.0*Jp*(a3+1.0) - 3.0*Lp*(a3+1.0) + Ksymp
    D4 = -D4/(9.0*A4*(a3*a4-a4*a4))

    t0 = (8.0/3.0)*C0
    t3 = 16.0*C3
    t4 = 16.0*C4
    x0 = -0.5*((3.0*D0/C0) + 1.0)
    x3 = -0.5*((3.0*D3/C3) + 1.0)
    x4 = -0.5*((3.0*D4/C4) + 1.0)

    # print(t0,t1,t2,t3,t4)
    # print(x0,x1,x2,x3,x4)
    # #results from Will's code for my default injected parameters
    return t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4


#Note that Qsym is not independant in this version of the skyrme model
def Qsym(nsat,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4):
    n=nsat  #definately could get nsat from the other parameters, but this will do for now
    hbar=197.3269804
    m=939.56542052
    h2over2m=(hbar*hbar)/(2*m)
    kremains = 3/5 * (3*np.pi*np.pi)**(2/3)
    p0 = h2over2m * kremains * (1-2**(-2/3)) * n**(-7/3) * 8/27
    p1 = 0
    p2 = -t3/48 * n**(a3-2) * (2*x3+1) * (a3+1) * a3 * (a3-1)
    p3 = -t4/48 * n**(a4-2) * (2*x4+1) * (a4+1) * a4 * (a4-1)
    p4 = 1/8 * kremains * (1-2**(-2/3)) * (t1*(2+x1)+t2*(2+x2)) * n**(-4/3) * -10/27
    p5 = -1/8 * kremains * (1-2**(-5/3)) * (t1*(2*x1+1)-t2*(2*x2+1)) * n**(-4/3) * -10/27
    return 27*(nsat**3)*(p0+p1+p2+p3+p4+p5)


#Note that Qsym is not independant in this version of the skyrme model
def Zsym(nsat,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4):
    n=nsat  #definately could get nsat from the other parameters, but this will do for now
    hbar=197.3269804
    m=939.56542052
    h2over2m=(hbar*hbar)/(2*m)
    kremains = 3/5 * (3*np.pi*np.pi)**(2/3)
    p0 = h2over2m * kremains * (1-2**(-2/3)) * n**(-10/3) * -56/81
    p1 = 0
    p2 = -t3/48 * n**(a3-3) * (2*x3+1) * (a3+1) * a3 * (a3-1) * (a3-2)
    p3 = -t4/48 * n**(a4-3) * (2*x4+1) * (a4+1) * a4 * (a4-1) * (a4-2)
    p4 = 1/8 * kremains * (1-2**(-2/3)) * (t1*(2+x1)+t2*(2+x2)) * n**(-7/3) * 40/81
    p5 = -1/8 * kremains * (1-2**(-5/3)) * (t1*(2*x1+1)-t2*(2*x2+1)) * n**(-7/3) * 40/81
    return 81*(nsat**4)*(p0+p1+p2+p3+p4+p5)


# Define a likelihood function that takes nuc params generated by MCMC, uses them to solve the TOV
# eqns, produces the radius of a 1.4 solar mass star and calculates the likelihood of generating the
# observed data generated using Will's polytropic model
def ln_likelihood(params):

    # Append the params not being varied
    params = np.append(list_param_sly4[0:5], params)
    params = np.append(params, list_param_sly4[10:])

    # Build the EoS
    Eos = build_eos(
        Den,
        form="tov",
        mf_model="mm_", # mm_ is nucleonic, qyc is quarkyonic
        pair_model="no_",
        ffg_type="nr",
        mm_type="mmnr",
        # mm_param="SLy5opt2",
        mm_param=params,
        qyc_type="qycis__",
        qyc_param="250-3",
        # qyc_param=list_param_qyc,
        pair_type="pair1",
        pair_bcs="bcs",
        pair_param=list_param_pair,
        fmuon="y",
        fneutrino="n",
        fbnuc="y",
        fblep="y",
        ns_massRef=list_ns_massRef,
        aFsFlag=list_aFsFlag,
        force="new",
        aInit=list_aInit,
        flags_ns=list_flags_ns,
        fs_param=list_fs_param,
        ns_outFileFormat=list_ns_outFileFormat
        )

    # Central density array for NS
    NSnbc = setup_den_tov(model="n-lin", den_step=0.05, NPOINT=80)

    # Solves TOV eqns for each central density and solves to give masses and radii
    atov = tovsolver(NSnbc, Eos)

    print("Radii:", atov.rad, "in", atov.rad_unit)
    print("Densities:", atov.aNSnbc)
    print("Masses:", atov.m, "in", atov.m_unit)

    # Use linear interpolation to determine the radius of a 1.4 solar mass star
    difference_array = 1.4 - atov.m
    nearest_above = np.where(difference_array < 0, difference_array, -np.inf).argmax()
    nearest_below = np.where(difference_array > 0, difference_array, np.inf).argmin()

    if len(np.unique(atov.m)) > 1 and atov.failed[nearest_above] == 0 and atov.failed[nearest_below] == 0:
        rad_at_1_point_4_msun_nuc = atov.rad[nearest_below] + (1.4-atov.m[nearest_below])*((atov.rad[nearest_above]-atov.rad[nearest_below])/(atov.m[nearest_above]-atov.m[nearest_below]))
        return [np.log(scipy.stats.norm(rad_at_1_point_4_msun_poly, 1).pdf(rad_at_1_point_4_msun_nuc)), np.max(atov.m)]

    return [-np.inf, -np.inf]


# Prior distribution, just uniform for now
def ln_prior(params):
    if np.min(e_sym_range) < params[0] < np.max(e_sym_range) and np.min(l_sym_range) < params[1] < np.max(l_sym_range) and np.min(k_sym_range) < params[2] < np.max(k_sym_range) and np.min(q_sym_range) < params[3] < np.max(q_sym_range) and np.min(z_sym_range) < params[4] < np.max(z_sym_range):
        return 0.5
    return -np.inf


# Calculate the posterior distribution using the priors and likelihoods
def ln_posterior(params):
    ln_prior_val = ln_prior(params)
    if not np.isfinite(ln_prior_val):
        return -np.inf, -np.inf
    return ln_prior_val + ln_likelihood(params)[0], ln_likelihood(params)[1]


ndim = 5
nwalkers = 32
nsteps = 1000

t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4=Sk_params(e_sym, l_sym, k_sym) #J,L,Ksym
q_sym = Qsym(0.15625851,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4)
z_sym = Zsym(0.15625851,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4)

initial_params = [e_sym, l_sym, k_sym, 350.0, -690.0]

initial_positions = [initial_params + 1e-2 * np.random.randn(ndim) * param_ranges for i in range(nwalkers)]

# Setup the backend to ensure data is consistently logged
# Good backup to have in case the code crashes halfway through for example
filename = "poly_inference_32_50_0_0.5_0.5.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool(16) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, backend=backend, pool=pool)
    sampler.run_mcmc(initial_positions, nsteps, **{'skip_initial_state_check':True})
