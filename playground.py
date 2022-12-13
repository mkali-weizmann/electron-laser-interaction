import numpy as np
from microscope import *


# %%
C = Cavity2FrequenciesNumericalPropagator(l_1=1064 * 1e-9,
                                          l_2=532 * 1e-9,
                                          E_1=1.725e9,
                                          E_2=-1,
                                          NA=0.1,
                                          n_z=1000,
                                          n_t=100,
                                          alpha_cavity=None,  # tilt angle of the lattice (of the cavity)
                                          theta_polarization=0,
                                          ignore_past_files=True,
                                          debug_mode=True,
                                          z_integral_interval=80e-6)
n_x = 150
n_y = 150
input_coordinate_system = CoordinateSystem(lengths=(150e-6, 30e-6),
                                           n_points=(n_x, n_y))
input_wave = WaveFunction(psi=np.ones((n_x, n_y)),
                          coordinates=input_coordinate_system,
                          E0=KeV2Joules(300))

output_wave = C.propagate(input_wave)
# %%
N = 10
Es = np.linspace(1.71e9, 1.73e9, N)
phases = np.zeros(N, dtype=np.complex128)

n_x = 5
n_y = 5
input_coordinate_system = CoordinateSystem(lengths=(1e-8, 1e-8),
                                           n_points=(n_x, n_y))
input_wave = WaveFunction(psi=np.ones((n_x, n_y)),
                          coordinates=input_coordinate_system,
                          E0=KeV2Joules(300))

for i, E in enumerate(Es):
    print(i)
    C = Cavity2FrequenciesNumericalPropagator(l_1=1064 * 1e-9,
                                              l_2=532 * 1e-9,
                                              E_1=E,
                                              E_2=-1,
                                              NA=0.1,
                                              n_z=1000,
                                              n_t=100,
                                              alpha_cavity=None,  # tilt angle of the lattice (of the cavity)
                                              theta_polarization=0,
                                              # ignore_past_files=True,
                                              # debug_mode=True,
                                              z_integral_interval=80e-6)
    phase_and_amplitude_mask = C.generate_phase_and_amplitude_mask(input_wave)
    phases[i] = np.angle(phase_and_amplitude_mask[n_x // 2, n_y // 2])
plt.plot(Es, phases)
plt.axhline(-np.pi/2, color='r')
plt.show()

