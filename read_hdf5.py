import emcee
import corner
import h5py
import numpy as np
import matplotlib.pyplot as plt

reader = emcee.backends.HDFBackend('poly_inference_32_20_0_0.5_0.5.h5')

samples = reader.get_chain(discard=100, flat=True)
print(np.shape(samples))
max_mass_samps = reader.get_blobs(flat=True)
plt.hist(samples[:100, 4], 100, color="red", weights=np.zeros(len(samples[:100, 0])) + 1/len(samples[:100, 0]), histtype="step", label='100 samples')
plt.hist(samples[:1000, 4], 100, color="green", weights=np.zeros(len(samples[:1000, 0])) + 1/len(samples[:1000, 0]), histtype="step", label='1000 samples')
plt.hist(samples[:10000, 4], 100, color="blue", weights=np.zeros(len(samples[:10000, 0])) + 1/len(samples[:10000, 0]), histtype="step", label='10000 samples')
plt.hist(samples[:, 4], 100, color="black", weights=np.zeros(len(samples[:, 0])) + 1/len(samples[:, 0]), histtype="step", label='All samples')

#plt.hist(max_mass_samps[:], 100, range=[2.35, 2.45], histtype="step", color='black')
#plt.hist(max_mass_samps[:1000], 100, color="blue", histtype="step", label='1000 samples')
#plt.hist(max_mass_samps[:], 100, color="black", histtype="step", label='All samples')
plt.xlabel(r"Z$_{sym}$")
plt.ylabel(r"p(Z$_{sym}$)")
plt.legend()
plt.gca().set_yticks([])

fig = corner.corner(samples, labels=(["E$_{sym}$", "L$_{sym}$", "K$_{sym}$", "Q$_{sym}$", "Z$_{sym}$"]))

