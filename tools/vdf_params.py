import matplotlib.pyplot as plt
import numpy as np
import math

# constants
mp = 1.67e-27     # kg
kB = 1.380649e-23 # J/K

# enter parameters in SI!!!
particle_mass = 1*mp       # kg
density = 1e6              # m^-3
temperature = 5e5          # K
vmean = 5e5                # m/s
vmin = -1e6                # m/s
vmax =  1e6                # m/s
n_blocks = 25
block_width = 4
sparsity_threshold = 1e-15 # s^3/m^6

v_th = math.sqrt(3. * kB * temperature / particle_mass)

fig = plt.figure()
ax = fig.gca()

dv = (vmax-vmin)/(n_blocks * block_width)
v = np.arange(vmin, vmax, dv)

vdf = np.ma.array(density * (particle_mass / (2.*math.pi*kB*temperature))**(3./2.) * np.exp(-particle_mass*(v-vmean)**2 / (2.*kB*temperature)))
vdf = np.ma.masked_where(vdf < 0.01*sparsity_threshold, vdf)

print("Parameters:")
print("mass: ", particle_mass, " kg or ", particle_mass/mp, "proton masses")
print("temperature: ", temperature, " K or umpteen eV")
print("density: ", density, " m^-3")
print("thermal speed: ", v_th, " m/s")
print("dv: ", dv, " m/s")
print("VDF min, max, sparsity threshold:", np.min(vdf), np.max(vdf), sparsity_threshold)


ax.scatter(v, vdf, s=10)
#ax.step(v, vdf)
ax.scatter(v[::block_width], vdf[::block_width], marker="|", s=300)
#ax.step(v[::block_width], vdf[::block_width], where="post")
ax.axvline(vmean)
ax.set_yscale("log")
ax.hlines(sparsity_threshold, vmin, vmax, label="sparsity threshold", color="C3", lw=3, alpha=0.5)
ax.axvspan(vmean - v_th, vmean + v_th, alpha=0.3, label="thermal velocity")
ax.axhspan(np.ma.min(vdf)*0.1, sparsity_threshold, xmin=vmin, xmax=vmax, color="C3", hatch="/", alpha=0.5)
ax.set_xlim((vmin, vmax))

ax.legend()

plt.draw()
plt.show()
