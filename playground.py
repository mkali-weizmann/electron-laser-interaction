from microscope import *
import pandas as pd
l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
N_POINTS = 256  # Resolution of image
pixel_size = 2e-10

input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_POINTS, N_POINTS))
input_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)), coordinates=input_coordinate_system,
                          E_0=Joules_of_keV(300))

first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)

fourier_plane_wave = first_lens.propagate(input_wave)

columns = ['NA_1', 'polarization_pies', 'ring_cavity', 'E_0', 'second_laser', 'alpha_cavity_deviation_degrees', 'n_electrons', 'resolution', 'pixel_size', 'focal_length_mm', 'Cs_mm', 'defocus_nm', 'n_z', 'power_1']
list_of_combinations = [[0.05, 1/2,      True,          300,    True,          0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.1,  1/2,      True,          300,    True,          0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.2,  1/2,      True,          300,    True,          0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.05, 0,        True,          300,    True,          0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.05, 1/2,      False,         300,    True,          0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.05, 1/2,      True,          100,    True,          0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.05, 1/2,      True,          300,    False,         0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.05, 1/2,      True,          300,    True,          3,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   -1],
                        [0.05, 1/2,      True,          300,    True,          0,                               20,             256,          2e-10,        3.3,               0.3,     0,           None,   0]
                        ]
df = pd.DataFrame(list_of_combinations, columns=columns)


def f(NA_1=0.05, second_laser=True,
      ring_cavity=True, polarization_pies=1 / 2, E_0=300, defocus_nm=0,
      Cs_mm=0.3, n_electrons=20, power_1=-1,
      focal_length_mm=3.3, alpha_cavity_deviation_degrees=0, resolution=256, pixel_size=2e-10, n_z=None):
    first_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=True)
    if second_laser:
        power_2 = -1
    else:
        power_2 = None

    input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=r'Data Arrays\Static Data\apof.mrc')
    input_wave = WaveFunction(E_0=input_wave_full.E_0,
                              psi=input_wave_full.psi[150:150 + resolution, 150:150 + resolution],
                              coordinates=CoordinateSystem(dxdydz=(pixel_size, pixel_size),
                                                           n_points=(resolution, resolution)))

    cavity = CavityNumericalPropagator(l_1=l_1, l_2=l_2, power_1=power_1, power_2=power_2, NA_1=NA_1,
                                       ring_cavity=ring_cavity,
                                       alpha_cavity_deviation=alpha_cavity_deviation_degrees / 360 * 2 * np.pi,
                                       theta_polarization=polarization_pies * np.pi,
                                       n_z=n_z, input_wave_energy_for_power_finding=Joules_of_keV(E_0))
    print(cavity.n_z)
    second_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=False)
    aberration_propagator = AberrationsPropagator(Cs=Cs_mm * 1e-3, defocus=defocus_nm * 1e-9, astigmatism_parameter=0,
                                                  astigmatism_orientation=0)
    M = Microscope([first_lens, cavity, second_lens, aberration_propagator],
                   n_electrons_per_square_angstrom=n_electrons)
    pic = M.take_a_picture(input_wave)

    fig, ax = plt.subplots(2, 3, figsize=(21, 14))
    mask = cavity.load_or_calculate_phase_and_amplitude_mask(M.step_of_propagator(cavity).output_wave)


    im_intensity = ax[0, 1].imshow(np.flip(pic.values), extent=input_wave.coordinates.limits)
    ax[0, 1].set_title(f"Image")
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_intensity, cax=cax, orientation="vertical")

    output_wave = M.step_of_propagator(aberration_propagator).output_wave
    image_fourier_plane = np.fft.fft2(output_wave.psi)
    fft_freq_x = np.fft.fftfreq(output_wave.psi.shape[0], output_wave.coordinates.dxdydz[0])
    fft_freq_x = np.fft.fftshift(fft_freq_x)
    image_fourier_plane = np.clip(np.abs(image_fourier_plane), a_min=0,
                                  a_max=np.percentile(np.abs(image_fourier_plane), 99))
    im_fourier = ax[1, 1].imshow(np.abs(image_fourier_plane),
                                 extent=(fft_freq_x[0], fft_freq_x[-1], fft_freq_x[0], fft_freq_x[-1]))
    ax[1, 1].set_title(f"Image - Fourier")
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_fourier, cax=cax, orientation="vertical")

    mask_phase = ax[0, 0].imshow(np.angle(input_wave.psi), extent=input_wave.coordinates.limits)
    ax[0, 0].set_title(f"Original wave - phase")
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_phase, cax=cax, orientation="vertical")

    mask_attenuation = ax[1, 0].imshow(np.abs(input_wave.psi) ** 2, extent=input_wave.coordinates.limits)
    ax[1, 0].set_title(f"Original image - Intensity")
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_attenuation, cax=cax, orientation="vertical")

    mask_phase = ax[0, 2].imshow(np.angle(mask), extent=M.step_of_propagator(cavity).input_wave.coordinates.limits)
    ax[0, 2].set_title(f"mask - phase")
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_phase, cax=cax, orientation="vertical")

    mask_attenuation = ax[1, 2].imshow(np.abs(mask) ** 2,
                                       extent=M.step_of_propagator(cavity).input_wave.coordinates.limits)
    ax[1, 2].set_title(f"mask - intensity transfer")
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_attenuation, cax=cax, orientation="vertical")

    title_a = f"NA={NA_1:.2f}, ring_cavity={ring_cavity}, second_laser={second_laser}, polarization_pies={polarization_pies:.2f}, E_0={E_0:.0f}, alpha_cavity_deviation_degrees={alpha_cavity_deviation_degrees:.0f}"
    title_b = f"power_1={cavity.power_1:.2e}"
    title_c = f"Cs_mm={Cs_mm:.2f}, n_electrons_per_ang2={n_electrons:.0f}, defocus_nm={defocus_nm:.0f}"
    plt.suptitle(f"{title_a}\n{title_b}, {title_c}")
    file_name = f"Figures\\examples\\{title_a}, {title_b}.png"
    plt.savefig(file_name)
    plt.close()

for index, r in df.iterrows():
    print(r.to_dict())
    f(**r.to_dict())


