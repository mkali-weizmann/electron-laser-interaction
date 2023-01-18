from microscope import *

l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
N_POINTS = 128  # Resolution of image
pixel_size = 1e-10

input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)),
                          coordinates=input_coordinate_system,
                          E0=Joules_of_keV(300))

dummy_sample = SamplePropagator(dummy_potential=f'letters_{N_POINTS}',
                                coordinates_for_dummy_potential=CoordinateSystem(axes=(input_coordinate_system.x_axis,
                                                                                       input_coordinate_system.y_axis,
                                                                                       np.linspace(-5e-10, 5e-10, 2)
                                                                                       )))
dummy_sample.sample.values *= 30
first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)
second_lens = LensPropagator(focal_length=3.3e-3, fft_shift=False)
cavity_2f_analytical = CavityAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=False,
                                                  starting_E_in_auto_E_search=1e6)
cavity_2f_numerical = CavityNumericalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=False,
                                                  starting_E_in_auto_E_search=1e6, debug_mode=True)
aberration_propagator = AberrationsPropagator(Cs=1e-8, defocus=1e-10, astigmatism_parameter=0,
                                              astigmatism_orientation=0)

# M_a = Microscope([dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator],
#                print_progress=True,
#                n_electrons_per_square_angstrom=50)

M_n = Microscope([dummy_sample, first_lens, cavity_2f_numerical, second_lens, aberration_propagator],
               print_progress=True,
               n_electrons_per_square_angstrom=50)


# pic_a = M_a.take_a_picture(first_wave)
pic_n = M_n.take_a_picture(first_wave)

# asd = np.real(np.angle(M_a.propagation_steps[0].output_wave.psi))
# M_a.plot_step(dummy_sample)

# plt.imshow(asd, cmap='gray')
# plt.colorbar()
# plt.show()
# # plt.imshow(pic_a.values)
# # plt.title('image_a')
# # plt.show()
# #
# # plt.imshow(pic_b.values)
# # plt.title('image_a')
# # plt.show()
