import numpy as np
from microscope import *

# %%
C = Cavity2FrequenciesNumericalPropagator(l_1=1064 * 1e-9,
                                          l_2=532 * 1e-9,
                                          E_1=1.37e9,  # 1.93e9
                                          E_2=-1,
                                          NA_1=0.1,
                                          n_t=3,
                                          alpha_cavity=None,  # tilt angle of the lattice (of the cavity)
                                          theta_polarization=np.pi / 2,
                                          # ignore_past_files=True,
                                          debug_mode=True)
n_x = 200
n_y = 1
input_coordinate_system = CoordinateSystem(lengths=(350e-6, 0),  # 50e-6
                                           n_points=(n_x, n_y))
dummy_input_wave = WaveFunction(psi=np.ones((n_x, n_y)),
                                coordinates=input_coordinate_system,
                                E0=KeV2Joules(300))

phase_and_amplitude_mask_ = C.generate_phase_and_amplitude_mask(dummy_input_wave)
print(np.angle(phase_and_amplitude_mask_[n_x // 2, n_y // 2]))
print(np.abs(phase_and_amplitude_mask_[n_x // 2, n_y // 2]))

# %%  Get amplitude for pi/2 phase shift: DO NOT ERASE
# N = 10
# Es = np.linspace(1e7, 2.4e9, N)
# phases = np.zeros(N, dtype=np.complex128)
#
# n_x = 5
# n_y = 1
# input_coordinate_system = CoordinateSystem(lengths=(1e-6, 0),
#                                            n_points=(n_x, n_y))
# input_wave = WaveFunction(psi=np.ones((n_x, n_y)),
#                           coordinates=input_coordinate_system,
#                           E0=KeV2Joules(300))
#
# for i, E in enumerate(Es):
#     print(i)
#     C = Cavity2FrequenciesNumericalPropagator(l_1=1064 * 1e-9,
#                                               l_2=532 * 1e-9,
#                                               E_1=E,
#                                               E_2=-1,
#                                               NA=0.1,
#                                               n_t=3,
#                                               alpha_cavity=None,  # tilt angle of the lattice (of the cavity)
#                                               theta_polarization=0,
#                                               # ignore_past_files=True,
#                                               # debug_mode=True)
#                                               )
#     phase_and_amplitude_mask = C.generate_phase_and_amplitude_mask(input_wave)
#     phases[i] = np.angle(phase_and_amplitude_mask[n_x // 2, n_y // 2])
# plt.plot(Es, np.real(phases))
# plt.axhline(-np.pi/2, color='r')
# plt.show()

# %%  Get derivative factor for pi/2 phase shift:
# N = 100
# derivative_factors = np.logspace(1, 2.4, N)
# phases = np.zeros(N)
#
# n_x = 5
# n_y = 1
# input_coordinate_system = CoordinateSystem(lengths=(1e-6, 0),
#                                            n_points=(n_x, n_y))
# input_wave = WaveFunction(psi=np.ones((n_x, n_y)),
#                           coordinates=input_coordinate_system,
#                           E0=KeV2Joules(300))
#
# for i, D in enumerate(derivative_factors):
#     C = Cavity2FrequenciesNumericalPropagator(l_1=1064 * 1e-9,
#                                               l_2=532 * 1e-9,
#                                               E_1=2.245e9,
#                                               E_2=-1,
#                                               NA=0.1,
#                                               n_z=1000,
#                                               n_t=3,
#                                               alpha_cavity=None,  # tilt angle of the lattice (of the cavity)
#                                               theta_polarization=0,
#                                               derivative_factor=derivative_factors[i],
#                                               # ignore_past_files=True,
#                                               # debug_mode=True)
#                                               )
#     phase_and_amplitude_mask = C.generate_phase_and_amplitude_mask(input_wave)
#     phases[i] = np.angle(phase_and_amplitude_mask[n_x // 2, n_y // 2])
# plt.axhline(np.real(phases[-1]), color='orange', linestyle='--')
# plt.plot(derivative_factors, np.real(phases))
# plt.axhline(-np.pi/2, color='r')
# plt.ylim(np.min(np.real(phases)), 0)
# plt.show()

# %%
# N = 100
# NZs = np.logspace(1.9, 3, N).astype(int)
# phases = np.zeros(N)
#
# n_x = 5
# n_y = 1
# input_coordinate_system = CoordinateSystem(lengths=(1e-6, 0),
#                                            n_points=(n_x, n_y))
# input_wave = WaveFunction(psi=np.ones((n_x, n_y)),
#                           coordinates=input_coordinate_system,
#                           E0=KeV2Joules(300))
#
# for i, D in enumerate(NZs):
#     C = Cavity2FrequenciesNumericalPropagator(l_1=1064 * 1e-9,
#                                               l_2=532 * 1e-9,
#                                               E_1=2.245e9,
#                                               E_2=-1,
#                                               NA=0.1,
#                                               n_t=3,
#                                               alpha_cavity=None,  # tilt angle of the lattice (of the cavity)
#                                               theta_polarization=0
#                                               # ignore_past_files=True,
#                                               # debug_mode=True)
#                                               )
#     phase_and_amplitude_mask = C.generate_phase_and_amplitude_mask(input_wave)
#     phases[i] = np.angle(phase_and_amplitude_mask[n_x // 2, n_y // 2])
# plt.axhline(np.real(phases[-1]), color='orange', linestyle='--')
# plt.plot(NZs, np.real(phases))
# plt.axhline(-np.pi/2, color='r')
# plt.ylim(np.min(np.real(phases)), 0)
# plt.show()

from microscope import *
# initiate input wave function:
N_POINTS = 1024
input_coordinate_system = CoordinateSystem(lengths=(52.4e-9, 52.4e-9), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)),
                          coordinates=input_coordinate_system,
                          E0=KeV2Joules(300))
# initiate dummy_sample, lenses, and a cavity:
dummy_sample = SamplePropagator(dummy_potential='letters',
                                axes=tuple([first_wave.coordinates.axes[0],
                                            first_wave.coordinates.axes[1],
                                            np.linspace(-5e-10, 5e-10, 2)]))
first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)

second_lens = LensPropagator(focal_length=3.3e-3, fft_shift=False)
cavity_2f = Cavity2FrequenciesAnalyticalPropagator(l_1=1064e-9, l_2=532e-9, E_1=3.42e-7, E_2=-1, NA=0.05)

# Initiate Microscope, which is a list of all the propagators:
M = Microscope([dummy_sample, first_lens, cavity_2f, second_lens])

# propagate the WaveFunction through all the propagators:
M.propagate(first_wave)
# %%



print(True or True)
print(True or False)
print(False or True)
print(False or False)





































