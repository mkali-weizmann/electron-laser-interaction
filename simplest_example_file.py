from microscope import *

l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
N_POINTS = 128  # Resolution of image
pixel_size = 1e-10

input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)), coordinates=input_coordinate_system, E0=Joules_of_keV(300))

dummy_sample = SamplePropagator(
    dummy_potential=f"letters_{N_POINTS}",
    coordinates_for_dummy_potential=CoordinateSystem(
        axes=(input_coordinate_system.x_axis, input_coordinate_system.y_axis, np.linspace(-5e-10, 5e-10, 2))
    ),
)

first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)
second_lens = LensPropagator(focal_length=3.3e-3, fft_shift=False)
cavity_2f_analytical = CavityAnalyticalPropagator(
    l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=False, starting_E_in_auto_E_search=1e6
)
aberration_propagator = AberrationsPropagator(
    Cs=1e-8, defocus=10e-9, astigmatism_parameter=0, astigmatism_orientation=0
)

M = Microscope(
    [dummy_sample, first_lens, cavity_2f_analytical, second_lens, aberration_propagator],
    print_progress=True,
    n_electrons_per_square_angstrom=150,
)


pic = M.take_a_picture(first_wave)

plt.imshow(pic.values, extent=pic.coordinates.limits)
plt.title("image")
plt.show()
