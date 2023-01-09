# %%
from microscope import *
# %%
# N_x = 11
# N_y = 11
# pixel_size = 1e-10
# input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_x, N_y))
# first_wave = WaveFunction(psi=np.ones((N_x, N_y)),
#                           coordinates=input_coordinate_system,
#                           E0=Joules_of_keV(300))
#
#
# first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)
#
#
# C = Cavity2FrequenciesNumericalPropagator(E_1=925177518, NA_1=0.05,
#                                                             print_progress=True, ignore_past_files=True)
#
# fourier_plane_wave = first_lens.propagate(first_wave)
# # phi = C.phi(fourier_plane_wave.coordinates.x_axis, fourier_plane_wave.coordinates.y_axis, fourier_plane_wave.beta)
# mask = C.phase_and_amplitude_mask(fourier_plane_wave)
# print("kaki")
# %%
N_POINTS = 128
pixel_size = 1e-10
input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)),
                          coordinates=input_coordinate_system,
                          E0=Joules_of_keV(300))

dummy_sample = SamplePropagator(dummy_potential='letters small',
                                axes=tuple([first_wave.coordinates.axes[0],
                                            first_wave.coordinates.axes[1],
                                            np.linspace(-5e-10, 5e-10, 2)]))

first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)

C = Cavity2FrequenciesNumericalPropagator(E_1=925177518.67943, NA_1=0.05, print_progress=True, ignore_past_files=True, ring_cavity=True)
fourier_plane_wave = first_lens.propagate(first_wave)
mask = C.phase_and_amplitude_mask(fourier_plane_wave)
plt.imshow(np.abs(mask)**2)
plt.show()
print("kaki")