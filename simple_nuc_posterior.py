import numpy as np
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
Den = setup_den_tot(model="n-d-log", var2=0.0, n_min=2.e-7, n_max=1.7, npt=200)

list_param_sly4 = np.array([-15.97, 0.1595, 230.0, -225.0, -443.0, 32.01,  46.00, -120.0,  350.0,  -690.0, 1.0000, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 6.90, 0.00], dtype=np.float64)
list_param_qyc = np.array([250, 0.3])
list_param_pair = np.array([1.573e-03, 3.105, 8.551e-02, 1.386, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
list_ns_massRef =  np.array([8.67, 68.0, 30.0], dtype=np.float64) # SLy5
list_aFsFlag = np.array([4, 1, 0], dtype=np.int32)
list_aInit = np.array([60.0, 0.167, 0.4, 0.0], dtype=np.float64)
list_flags_ns = np.array([0, 1], dtype=np.int32)
# list_fs_param = [0.967, 1.039, 1.257, 1.116, 0.980, 1.000, 1.000, 1.000]
list_fs_param = 'SLy5'
list_ns_outFileFormat = np.array([0, 0, 0, 0], dtype=np.int32)

n = 20
sly4_0 = np.linspace(-16.35, -15.31, n)  # E_sat
sly4_1 = np.linspace(0.145, 0.1746, n)  # n_sat
sly4_2 = np.linspace(201, 338, n)  # K_sat
sly4_3 = np.linspace(-570, 950, n)  # Q_sat
sly4_4 = np.linspace(-903, 9997, n)  # Z_sat
sly4_5 = np.linspace(26.83, 38.71, n)  # E_sym
sly4_6 = np.linspace(9.9, 64, n)  # L_sym
sly4_7 = np.linspace(-235, 213, n)  # K_sym
sly4_8 = np.linspace(-86, 846, n)  # Q_sym
sly4_9 = np.linspace(-1450, -5, n)  # Z_sym

param_ranges = np.array([np.max(sly4_0)-np.min(sly4_0), np.max(sly4_1)-np.min(sly4_1)]) #,
                       # np.max(sly4_2)-np.min(sly4_2), np.max(sly4_3)-np.min(sly4_3),
                       # np.max(sly4_4)-np.min(sly4_4), np.max(sly4_5)-np.min(sly4_5),
                       # np.max(sly4_6)-np.min(sly4_6), np.max(sly4_7)-np.min(sly4_7),
                       # np.max(sly4_8)-np.min(sly4_8), np.max(sly4_9)-np.min(sly4_9)])

def ln_likelihood(params):
    # Append the params not being varied
    params = np.append(params, list_param_sly4[2:])

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
    NSnbc = setup_den_tov(model="test")

    # Solves TOV eqns for each central density and solves to give masses and radii
    atov = tovsolver(NSnbc, Eos)

    print("Radii:", atov.rad, "in", atov.rad_unit)
    print("Densities:", atov.aNSnbc)
    print("Masses:", atov.m, "in", atov.m_unit)

    if np.all(atov.rad):
        return [0., np.max(atov.m)]

    return -np.inf


def ln_prior(params):
    if np.min(sly4_0) < params[0] < np.max(sly4_0) and np.min(sly4_1) < params[1] < np.max(sly4_1): # and \
   # np.min(sly4_2) < params[2] < np.max(sly4_2) and np.min(sly4_3) < params[3] < np.max(sly4_3) and \
   # np.min(sly4_4) < params[4] < np.max(sly4_4) and np.min(sly4_5) < params[5] < np.max(sly4_5) and \
   # np.min(sly4_6) < params[6] < np.max(sly4_6) and np.min(sly4_7) < params[7] < np.max(sly4_7) and \
   # np.min(sly4_8) < params[8] < np.max(sly4_8) and np.min(sly4_9) < params[9] < np.max(sly4_9):
        return 0.5
    return -np.inf


def ln_posterior(params):
    ln_prior_val = ln_prior(params)
    if not np.isfinite(ln_prior_val):
        return -np.inf, ln_likelihood(params)[1]
    return ln_prior_val + ln_likelihood(params)[0], ln_likelihood(params)[1]


ndim = 2
nwalkers = 32
nsteps = 1000

initial_positions = [list_param_sly4[0:2] + 1e-2 * np.random.randn(ndim) * param_ranges for i in range(nwalkers)]

# Setup the backend to ensure data is consistently logged
# Good backup to have in case the code crashes halfway through for example
filename = "simple_nuc_posterior.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, backend=backend, pool=pool)
    sampler.run_mcmc(initial_positions, nsteps, **{'skip_initial_state_check':True})





















