# Builds EOS from input params and solves TOV eqns to return mass-radius relationship
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
nseospy_tk = os.getenv('NSEOSPY_TK')
sys.path.insert(0, nseospy_tk)

from nseospy import setup_den_tot
from nseospy import setup_den_tov
from nseospy import tovsolver
from nseospy import build_eos

def main():
    fig, ax = plt.subplots()

    # Define the equation of state
    print('-'*20)
    print("Build EOS")
    print('-'*20)

    # Create the input array:
    print("Create input array:")

    # Array of central densities (baryons fm^-3) for EOS (logarithmic spacing)
    # Solves for pressure at 200 density points and interpolates
    Den = setup_den_tot(model="n-d-log", var2=0.0, n_min=2.e-7, n_max=1.7, npt=200)

    # Main nucleonic parameters
    # SLy4 - popular nucleonic model
    # First 5 are params of symmetric matter and second 5 are the params of the difference between SNM and PNM (symmetry energy)
    # First couple are well constrained whereas later params are more uncertain
    # Using Margueron 2018 to determine reasonable parameter ranges based on nuclear data

    n = 20  # Number of data points in the parameter range to explore

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

# =============================================================================
#     params = np.array([[ 31.47557709,  48.52601495, -84.3140322],
# [  31.84414266,   47.66744407, -128.23152032],
# [  34.56709615,   40.82086218, -189.1195815 ],
# [30.55492445, 47.07537119, 16.74177759], # FAILS ON THESE PARAMS, SHOULD K_SYM NOT BE POSITIVE?
# [  32.58990553,   38.58091106, -157.56153909],
# [  34.63594482,   30.99440832, -110.36273107],
# [ 33.49661247,  41.2749154,  -93.70987502],
# [  31.65163227,   48.37985501, -114.03509775],
# [ 35.03829891,  49.86532134, -79.5006231 ],
# [ 31.88489102,  40.99766262, -95.06729553],
# [  32.05936852,   51.71919517, -172.58539036],
# [ 30.37950703,  41.26929417, -96.16228568],
# [  31.56520959,   55.04753024, -139.53475464],
# [  36.45467475,   48.14146179, -199.70352159],
# [ 30.81766793,  46.02282455, -57.26042727],
# [  32.28990408,   44.8208114,  -131.45008346]])
# I THINK WE NEED MORE STRINGENT CHECKS ON WHETHER THE EOS SOLVER CONVERGES / PRODUCES A VALID SOLUTION
# =============================================================================

    params = np.array([[27.64567143, 60.75944073, -7.82177855],
[ 35.55791214,  31.9232412,  -50.001861  ],
[  27.7298963,    44.51735009, -169.60861346],
[ 33.95666557,  32.84620864, -92.5426332 ],
[  31.20376687,   31.57080596, -163.11193449],
[  31.54636659,   25.91586118, -109.45549538],
[  32.63370326,   49.52999429, -215.22208858],
[  30.84470229,   37.79642873, -162.55187495],
[ 34.23059095,  32.85675247, -90.98966031],
[24.84893775, 44.75030407, 28.91665762], # Fails at this one
[ 31.64518031,  76.6713388,  -68.64320741],
[  24.18763499,   21.64064866, -139.81388464],
[  31.69010975,   34.57609335, -112.9171583 ],
[  37.48470571,   41.41999124, -161.60601777],
[  27.31359163,   47.97393612, -187.43772642],
[ 36.66770347,  42.18697854, -23.35118327]])

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

    for i in range(0, len(params)):
        list_param_sly4 = np.array([-15.97, 0.1595, 230.0, -225.0, -443.0, params[i, 0],  params[i, 1], params[i, 2],  350.0,  -690.0, 1.0000, 0.000, 0.0, 0.0, 0.0, 0.0, 0.0, 6.90, 0.00], dtype=np.float64)

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

# =============================================================================
#         print("Pressure:", Eos.pre_b)
#         print("Baryon density:", Eos.n_b)
#         print("Energy density:", Eos.rho_b)
#         print("Sound speed:", Eos.cs2_b)
# =============================================================================

        # Create the input array: nbc

        print('-'*20)
        print("Create input array: nbc")
        print('-'*20)

        # Central density array for NS
        NSnbc = setup_den_tov(model="test")

        # Test tovSolver
        print('-'*20)
        print("Test tovSolver")
        print('-'*20)

        # Solves TOV eqns for each central density and solves to give masses and radii
        try:
            atov = tovsolver(NSnbc, Eos)
        except:
            print("This one failed")

        print("Print output:")
        # Central density values
        print("Params:", params[i])
        print(atov.failed)
        print(np.unique(atov.m))
        print("Densities:", atov.aNSnbc)
        # Radius
        print("Radii:", atov.rad, " in ", atov.rad_unit)
        # Mass
        print("Masses:", atov.m, " in ", atov.m_unit)
        # Tidal deformability
        print("Tidal deformabilities:", atov.tdlda)

        print("Write EoS in file:")
        print("End of test:")

if __name__ == "__main__":
    main()


