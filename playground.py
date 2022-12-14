import numpy as np
from microscope import *
# %%  Get amplitude for pi/2 phase shift: DO NOT ERASE
N = 10
Es = np.linspace(2.2e9, 2.4e9, N)
phases = np.zeros(N, dtype=np.complex128)

n_x = 13
n_y = 1
input_coordinate_system = CoordinateSystem(lengths=(1e-8, 0),
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
                                              # debug_mode=True)
                                              )
    phase_and_amplitude_mask = C.generate_phase_and_amplitude_mask(input_wave)
    phases[i] = np.angle(phase_and_amplitude_mask[n_x // 2, n_y // 2])
plt.plot(Es, np.real(phases))
plt.axhline(-np.pi/2, color='r')
plt.show()

