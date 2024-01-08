from microscope import *
import pandas as pd
l_1 = 1064e-9
l_2 = 532e-9

columns = ['NA_1', 'polarization_pies', 'ring_cavity', 'E_0', 'second_laser', 'alpha_cavity_deviation_degrees', 'n_electrons', 'resolution', 'focal_length_mm', 'Cs_mm', 'defocus_nm', 'n_z', 'power_1', 'custom_title']
list_of_combinations = [
                        # [0.05, 1/2,      True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        # [0.1,  1/2,      True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        [0.2,  1/2,      True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1, 'Simulating a double-wavelength laser phase plate'],
                        # [0.05, 1/2,      False,         300,    False,         0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        # [0.1,  1/2,      False,         300,    False,         0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        [0.2,  1/2,      False,         300,    False,         0,                               20,             1024,          3.3,               3,       0,           None,   -1, 'Simulating a single-wavelength laser phase plate'],
                        # [0.05, 0,        True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        # [0.05, 1/2,      False,         300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        # [0.05, 1/2,      True,          100,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        # [0.05, 1/2,      True,          300,    False,         0,                               20,             1024,          3.3,               3,       0,           None,   -1, None],
                        # [0.05, 1/2,      True,          300,    True,          0.2,                             20,             1024,          3.3,               3,       0,           None,   -1, None],
                        # [0.05, 1/2,      True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   0, None]
                        ]
df = pd.DataFrame(list_of_combinations, columns=columns)
N_0_x = 100
N_0_y = 150


def f(NA_1=0.05, second_laser=True,
      ring_cavity=True, polarization_pies=1 / 2, E_0=300, defocus_nm=0,
      Cs_mm=0.3, n_electrons=20, power_1=-1,
      focal_length_mm=3.3, alpha_cavity_deviation_degrees=0, resolution=256, n_z=None, shot_noise=True,
      file_name="myoglobin", custom_title=''):
    first_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=True)
    if second_laser:
        power_2 = -1
    else:
        power_2 = None

    input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=f"data\\static data\\{file_name}.mrc")
    input_wave = WaveFunction(E_0=input_wave_full.E_0,
                              psi=input_wave_full.psi[N_0_x:N_0_x + resolution, N_0_y:N_0_y + resolution],
                              coordinates=CoordinateSystem(dxdydz=input_wave_full.coordinates.dxdydz,
                                                           n_points=(resolution, resolution)))

    cavity = CavityNumericalPropagator(l_1=l_1, l_2=l_2, power_1=power_1, power_2=power_2, NA_1=NA_1,
                                       ring_cavity=ring_cavity,
                                       alpha_cavity_deviation=alpha_cavity_deviation_degrees / 360 * 2 * np.pi,
                                       theta_polarization=polarization_pies * np.pi,
                                       n_z=n_z, input_wave_energy_for_power_finding=Joules_of_keV(E_0))
    second_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=False)
    aberration_propagator = AberrationsPropagator(Cs=Cs_mm * 1e-3, defocus=defocus_nm * 1e-9, astigmatism_parameter=0,
                                                  astigmatism_orientation=0)
    M = Microscope([first_lens, cavity, second_lens, aberration_propagator],
                   n_electrons_per_square_angstrom=n_electrons)
    pic = M.take_a_picture(input_wave, add_shot_noise=shot_noise)

    fig, ax = plt.subplots(2, 3, figsize=(21, 14))
    mask = cavity.load_or_calculate_phase_and_amplitude_mask(M.step_of_propagator(cavity).output_wave)

    vmin, vmax = np.percentile(pic.values, [10, 90])
    im_intensity = ax[0, 1].imshow(np.flip(pic.values), extent=input_wave.coordinates.limits, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0, 1].set_title(f"Image", fontsize=20)
    ax[0, 1].set_title(f"(b)", loc='left', x=-0.08, fontsize=16)
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
                                 extent=(fft_freq_x[0], fft_freq_x[-1], fft_freq_x[0], fft_freq_x[-1]), cmap='gray')
    ax[1, 1].set_title(f"Image - Fourier", fontsize=20)
    ax[1, 1].set_title(f"(e)", loc='left', x=-0.08, fontsize=16)
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_fourier, cax=cax, orientation="vertical")

    mask_phase = ax[0, 0].imshow(np.angle(input_wave.psi), extent=input_wave.coordinates.limits, cmap='gray')
    ax[0, 0].set_title(f"Original wave - phase", fontsize=20)
    ax[0, 0].set_title(f"(a)", loc='left', x=-0.08, fontsize=16)
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_phase, cax=cax, orientation="vertical")

    mask_attenuation = ax[1, 0].imshow(np.abs(input_wave.psi) ** 2, extent=input_wave.coordinates.limits, cmap='gray')
    ax[1, 0].set_title(f"Original wave - Intensity", fontsize=20)
    ax[1, 0].set_title(f"(d)", loc='left', x=-0.08, fontsize=16)
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_attenuation, cax=cax, orientation="vertical")

    mask_phase = ax[0, 2].imshow(np.angle(mask), extent=M.step_of_propagator(cavity).input_wave.coordinates.limits, cmap='gray')
    ax[0, 2].set_title(f"mask - phase", fontsize=20)
    ax[0, 2].set_title(f"(c)", loc='left', x=-0.08, fontsize=16)
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_phase, cax=cax, orientation="vertical")

    mask_attenuation = ax[1, 2].imshow(np.abs(mask) ** 2,
                                       extent=M.step_of_propagator(cavity).input_wave.coordinates.limits, cmap='gray')
    ax[1, 2].set_title(f"mask - intensity transfer", fontsize=20)
    ax[1, 2].set_title(f"(f)", loc='left', x=-0.08, fontsize=16)
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mask_attenuation, cax=cax, orientation="vertical")

    # set the ticks font size to 12:
    for i in range(2):
        for j in range(3):
            for tick in ax[i, j].xaxis.get_major_ticks():
                tick.label.set_fontsize(12)
            for tick in ax[i, j].yaxis.get_major_ticks():
                tick.label.set_fontsize(12)

    title_a = f"NA={NA_1:.2f}, ring_cavity={ring_cavity}, second_laser={second_laser}, polarization_pies={polarization_pies:.2f}, E_0={E_0:.0f}, alpha_cavity_deviation_degrees={alpha_cavity_deviation_degrees:.0f}"
    shorted_title_a = f"NA={NA_1:.2f}, ring={ring_cavity}, second_laser={second_laser}, polarization={polarization_pies:.2f}, E_0={E_0:.0f}, alpha_deviation_degrees={alpha_cavity_deviation_degrees:.1f}"
    title_b = f"power_1={cavity.power_1:.2e}, shot_noise={shot_noise}"
    title_b_shorted = f"power_1={cavity.power_1:.2e}"
    title_c = f"Cs_mm={Cs_mm:.2f}, n_electrons_per_ang2={n_electrons:.0f}, defocus_nm={defocus_nm:.0f}"


    # Use custom title and file name:
    if custom_title is not None:
        # Setting a title with enlarged font:
        plt.suptitle(custom_title, fontsize=24)
        print(f"{title_a}\n{title_b}, {title_c}")
        file_name = f"Figures\\examples\\{custom_title}.png"
    else:
        # Generate title and file name:
        plt.suptitle(f"{title_a}\n{title_b}, {title_c}", fontsize=16)
        file_name = f"Figures\\examples\\{shorted_title_a}, {title_b_shorted}.png"

    plt.savefig(file_name)
    plt.show()
    plt.close()


import time
start_time = time.time()
for file_name in ["apof_in_ice"]:  # ,"myoglobin"
    for shot_noise in [True]:
        for index, r in df.iterrows():
            settings_dict = r.to_dict()
            settings_dict['shot_noise'] = shot_noise
            settings_dict['file_name'] = file_name
            print(settings_dict)
            f(**settings_dict)
            print(time.time() - start_time)
print("Done")

