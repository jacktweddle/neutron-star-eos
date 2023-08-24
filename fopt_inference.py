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

np.random.seed(10)

# Array of central densities (baryons fm^-3) for EOS (logarithmic spacing)
# Solves for pressure at 200 density points and interpolates
# What should I use for n_min and n_max?
Den = setup_den_tot(model="n-d-log", var2=0.0, n_min=2.e-7, n_max=1.7, npt=200)

list_param_sly4 = np.array([-15.97, 0.1595, 230.0, -225.0, -443.0, 32.01,  46.00, -120.0,  350.0,  -690.0, 1.0000, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 6.90, 0.00], dtype=np.float64)
list_param_qyc = np.array([250, 3])
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

# Define the parameters of the FOPT model

n_sat = list_param_sly4[1]  # Nuclear saturation density - should this be a parameter to the FOPT model?
n_trans = 1.5 * n_sat  # Transition density
delta_n_trans = 0.1 * n_sat  # Width of density transition
a = 0.5  # Sound speed after transition

# Calculate the values of other parameters at the transition
# Do I need to do / is it possible to do this or do I take these values from the nuc array?
rho_trans = 0
mu_trans = 0
p_trans = 0
cs_trans = 0

# Useful constants
c_squared = (3E8)**2

# Generate NSEOS according to nuc params
# and replace values in NSEOS with FOPT values following transition

# Should the EoS use any specific parameters and should the walkers start around these values?
# How does it work since the initial model is also nucleonic?

Eos = build_eos(
    Den,
    form="tov",
    mf_model="mm_",
    pair_model="no_",
    ffg_type="nr",
    mm_type="mmnr",
    # mm_param="SLy5opt2",
    mm_param=list_param_sly4,
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

for count, value in enumerate(Eos.n_b):
    if value > n_trans:
        Eos.rho_b[count] = rho_trans + (mu_trans/c_squared)*(value-n_trans)
        if value < n_trans+delta_n_trans:
            Eos.pre_b[count] = p_trans
            Eos.cs2_b[count] = 0
        elif value > n_trans+delta_n_trans:
            Eos.pre_b[count] = p_trans + a*(mu_trans/c_squared)*(value-n_trans)
            Eos.cs2_b[count] = a

# Generate radius at 1.4 solar masses based on FOPT EoS

NSnbc = setup_den_tov( model="n-lin", den_step = 0.01, NPOINT = 400 )
atov = tovsolver(NSnbc, Eos, domi = 'y', dotd = 'y')

difference_array = 1.4 - atov.m
nearest_above = np.where(difference_array < 0, difference_array, -np.inf).argmax()
nearest_below = np.where(difference_array > 0, difference_array, np.inf).argmin()
rad_at_1_point_4_msun_poly = atov.rad[nearest_below] + (1.4-atov.m[nearest_below])*((atov.rad[nearest_above]-atov.rad[nearest_below])/(atov.m[nearest_above]-atov.m[nearest_below]))

# Define a likelihood function that takes nuc params generated by MCMC, uses them to solve the TOV
# eqns, produces the radius of a 1.4 solar mass star and calculates the likelihood of generating the
# observed data generated using the simple FOPT model
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
    if np.min(e_sym_range) < params[0] < np.max(e_sym_range) and np.min(l_sym_range) < params[1] < np.max(l_sym_range) and \
    np.min(k_sym_range) < params[2] < np.max(k_sym_range) and np.min(q_sym_range) < params[3] < np.max(q_sym_range) and \
    np.min(z_sym_range) < params[4] < np.max(z_sym_range):
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

initial_positions = [list_param_sly4[5:10] + 1e-2 * np.random.randn(ndim) * param_ranges for i in range(nwalkers)]

# Setup the backend to ensure data is consistently logged
# Good backup to have in case the code crashes halfway through for example
filename = "fopt_inference.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool(16) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, backend=backend, pool=pool)
    sampler.run_mcmc(initial_positions, nsteps, **{'skip_initial_state_check':True})