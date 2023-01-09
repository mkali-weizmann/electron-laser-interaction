from microscope import *

# %% global parameters
l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
N_POINTS = 128  # Resolution of image
pixel_size = 1e-10


# %%
input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)),
                          coordinates=input_coordinate_system,
                          E0=Joules_of_keV(300))

dummy_sample = SamplePropagator(dummy_potential='letters small',
                                axes=tuple([first_wave.coordinates.axes[0],
                                            first_wave.coordinates.axes[1],
                                            np.linspace(-5e-10, 5e-10, 2)]))

first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)
second_lens = LensPropagator(focal_length=3.3e-3, fft_shift=False)

cavity_2f_analytical = Cavity2FrequenciesAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1)
cavity_2f_analytical_NA_8 = Cavity2FrequenciesAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=0.08)
cavity_2f_analytical_NA_10 = Cavity2FrequenciesAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=0.1)
cavity_2f_analytical_NA_12 = Cavity2FrequenciesAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=0.12)
cavity_2f_analytical_NA_20 = Cavity2FrequenciesAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=0.2)
cavity_2f_numerical = Cavity2FrequenciesNumericalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1,
                                                            print_progress=True, ignore_past_files=True)
cavity_2f_numerical_a = Cavity2FrequenciesNumericalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1,
                                                            print_progress=True, ignore_past_files=True)
cavity_1f = Cavity2FrequenciesAnalyticalPropagator(l_1=l_1, l_2=None, E_1=-1, E_2=None, NA_1=NA_1)
cavity_2f_analytical_ring = Cavity2FrequenciesAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=True)
cavity_2f_numerical_ring = Cavity2FrequenciesNumericalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1,
                                                                 ring_cavity=True, print_progress=True)

aberration_propagator = AberrationsPropagator(Cs=0e-7, defocus=0e-21, atigmatism_parameter=0, astigmatism_orientation=0)

M_2f_a = Microscope([dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator])
pic_2f_a = M_2f_a.take_a_picture(first_wave)

# M_2f_a_NA_8 = Microscope([dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator])
# pic_2f_a_NA_8 = M_2f_a_NA_8.take_a_picture(first_wave)
#
# M_2f_a_NA_10 = Microscope([dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator])
# pic_2f_a_NA_10 = M_2f_a_NA_10.take_a_picture(first_wave)
#
# M_2f_a = Microscope([dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator])
# pic_2f_a = M_2f_a.take_a_picture(first_wave)
#
# M_2f_a = Microscope([dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator])
# pic_2f_a = M_2f_a.take_a_picture(first_wave)

# M_2f_a_ring = Microscope([dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator])
# pic_2f_a_ring = M_2f_a_ring.take_a_picture(first_wave)
#
# M_1f = Microscope([dummy_sample, first_lens, cavity_1f, second_lens, aberration_propagator])
# pic_1f_a = M_1f.take_a_picture(first_wave)
#
M_2f_n = Microscope([dummy_sample, first_lens, cavity_2f_numerical, second_lens, aberration_propagator])
# pic_2f_n = M_2f_n.take_a_picture(first_wave)
#
# M_2f_n_ring = Microscope([dummy_sample, first_lens, cavity_2f_numerical, second_lens, aberration_propagator])
# # pic_2f_n_ring = M_2f_n_ring.take_a_picture(first_wave)
#
# M_no_cavity = Microscope([dummy_sample, first_lens, second_lens, aberration_propagator])
# no_f_pic = M_no_cavity.take_a_picture(first_wave)


# %%
# M.plot_step(0, title="specimen - input wave (upper) and output (lower) wave", file_name="specimen.png")
# M.plot_step(1, title="first lens - input wave (upper) and output (lower) wave", file_name="first_lens.png")
# M.plot_step(2, title="cavity_2f - input wave (upper) and output (lower) wave", file_name="cavity_2f.png")
# M.plot_step(3, title="second lens wave - input (upper) and output (lower) wave", file_name="second_lens.png")

# %% sample phase shift
# phase_map = np.angle(M_2f_a.step_of_propagator(dummy_sample).output_wave.psi)
# plt.title('total phase delay given by the sample')
# plt.imshow(phase_map - np.max(phase_map), extent=M_2f_a.propagation_steps[0].output_wave.coordinates.limits)
# plt.show()
# plt.savefig('Figures\phase_map_2f.png')
# %% analytical 2f cavity - final image
# plt.title('2f, standing, analytical')
# plt.imshow(pic_2f_a[0], extent=input_coordinate_system.limits)
# # plt.savefig('Figures\phase_map_2f.png')
# plt.show()

# %% numerical 2f cavity - final image
# plt.title('2f, standing, numerical')
# plt.imshow(pic_2f_n[0], extent=input_coordinate_system.limits)
# # plt.savefig('Figures\phase_map_2f.png')
# plt.show()

# %% analytical 1f cavity - final image
# plt.title('1f, standing, analytical')
# plt.imshow(pic_1f_a[0], extent=input_coordinate_system.limits)
# # plt.savefig('Figures\phase_map_2f.png')
# plt.show()

# %% analytical 2f cavity, ring cavity - final image
# plt.title('2f, ring, analytical')
# plt.imshow(pic_2f_a_ring[0], extent=input_coordinate_system.limits)
# # plt.savefig('Figures\phase_map_2f.png')
# plt.show()
# %%
fourier_plane_wave = first_lens.propagate(first_wave)
phase_and_amplitude_mask_n = cavity_2f_numerical.phase_and_amplitude_mask(fourier_plane_wave)
phase_and_amplitude_mask_a = cavity_2f_analytical.phase_and_amplitude_mask(fourier_plane_wave)
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(np.real(np.angle(phase_and_amplitude_mask_a)))
ax[0, 0].set_title('analytical - phase')
ax[0, 1].imshow(np.abs(phase_and_amplitude_mask_a)**2)
ax[0, 1].set_title('analytical - intensity')
ax[1, 0].imshow(np.real(np.angle(phase_and_amplitude_mask_n)))
ax[1, 0].set_title('numerical - phase')
ax[1, 1].imshow(np.abs(phase_and_amplitude_mask_n)**2)
ax[1, 1].set_title('numerical - intensity')
plt.show()

