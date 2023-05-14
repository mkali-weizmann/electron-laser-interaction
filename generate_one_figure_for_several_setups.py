import matplotlib.pyplot as plt

from microscope import *
import pandas as pd
import time
l_1 = 1064e-9
l_2 = 532e-9

columns = ['NA_1', 'polarization_pies', 'ring_cavity', 'E_0', 'second_laser', 'alpha_cavity_deviation_degrees', 'n_electrons', 'resolution', 'focal_length_mm', 'Cs_mm', 'defocus_nm', 'n_z', 'power_1']
list_of_combinations = [[0.05, 1/2,      True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1],
                        [0.1,  1/2,      True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1],
                        [0.2,  1/2,      True,          300,    True,          0,                               20,             1024,          3.3,               3,       0,           None,   -1],
                        [0.05, 1/2,      False,         300,    False,         0,                               20,             1024,          3.3,               3,       0,           None,   -1],
                        [0.1,  1/2,      False,         300,    False,         0,                               20,             1024,          3.3,               3,       0,           None,   -1],
                        [0.2,  1/2,      False,         300,    False,         0,                               20,             1024,          3.3,               3,       0,           None,   -1],
                        ]
df = pd.DataFrame(list_of_combinations, columns=columns)
N_0_x = 100
N_0_y = 150




def f(fig_1, fig_2, fig_3, NA_1=0.05, second_laser=True,
      ring_cavity=True, polarization_pies=1 / 2, E_0=300, defocus_nm=0,
      Cs_mm=0.3, n_electrons=20, power_1=-1,
      focal_length_mm=3.3, alpha_cavity_deviation_degrees=0, resolution=256, n_z=None, shot_noise=True,
      file_name="myoglobin", i=0, image_resolution_reduction_factor=2):
    first_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=True)

    if second_laser:
        power_2 = -1
        number_of_lasers = "two lasers"
    else:
        power_2 = None
        number_of_lasers = "one laser"

    input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=f"d\\Static Data\\{file_name}.mrc")
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

    ax = fig_1.add_subplot(2, 3, i + 1)
    image = lower_image_resolution(np.flip(pic.values), image_resolution_reduction_factor)
    vmin, vmax = np.percentile(image, [10, 90])
    im_intensity = ax.imshow(image, extent=input_wave.coordinates.limits, cmap='gray', vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig_1.colorbar(im_intensity, cax=cax, orientation="vertical")
    ax.set_title(f"{number_of_lasers}, NA_1={settings_dict['NA_1']:.2f}")

    ax = fig_2.add_subplot(2, 3, i + 1)
    image_fourier = np.abs(np.fft.fft2(np.flip(pic.values)))
    image_fourier = np.fft.fftshift(image_fourier)
    image_fourier = lower_image_resolution(image_fourier, image_resolution_reduction_factor)


    output_wave = M.step_of_propagator(aberration_propagator).output_wave

    fft_freq_x = np.fft.fftfreq(image.shape[0], output_wave.coordinates.dxdydz[0] * image_resolution_reduction_factor)
    fft_freq_x = np.fft.fftshift(fft_freq_x)

    im_fourier = ax.imshow(np.abs(image_fourier),
                           extent=(fft_freq_x[0], fft_freq_x[-1], fft_freq_x[0], fft_freq_x[-1]), cmap='gray', vmin=0, vmax=np.percentile(np.abs(image_fourier), 99))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig_2.colorbar(im_fourier, cax=cax, orientation="vertical")
    ax.set_title(f"{number_of_lasers}, NA={settings_dict['NA_1']:.2f}")

    plt.figure(fig_3.number)
    output_wave = M.step_of_propagator(aberration_propagator).output_wave
    wave_fourier_plane = np.fft.fft2(output_wave.psi)
    fft_freq_x = np.fft.fftfreq(output_wave.psi.shape[0], output_wave.coordinates.dxdydz[0])
    fft_freq_x = np.fft.fftshift(fft_freq_x)
    wave_fourier_plane = np.abs(wave_fourier_plane)
    # wave_fourier_plane = np.clip(np.abs(wave_fourier_plane), a_min=0,
    #                               a_max=np.percentile(np.abs(wave_fourier_plane), 99))
    freq_X, freq_Y = np.meshgrid(fft_freq_x, fft_freq_x)
    freq_R = np.sqrt(freq_X ** 2 + freq_Y ** 2)
    freq_R = np.round(freq_R, -5)
    freq_R_reshaped = freq_R.reshape(-1)
    sorting_permutation = np.argsort(freq_R_reshaped)
    freq_R_reshaped = freq_R_reshaped[sorting_permutation]
    wave_fourier_plane_reshaped = wave_fourier_plane.reshape(-1)
    wave_fourier_plane_reshaped = wave_fourier_plane_reshaped[sorting_permutation]
    freq_R_unique, slicing_indices = np.unique(freq_R_reshaped, return_index=True)
    binned_image_fourier_plane = np.split(wave_fourier_plane_reshaped, slicing_indices)[1:]
    mean = np.array([np.mean(x) for x in binned_image_fourier_plane])
    mean = np.clip(mean, a_min=0, a_max=np.percentile(mean, 99))
    mean = np.convolve(mean, np.ones(30) / 30, mode='same')
    plt.plot(freq_R_unique[15:-15], mean[15:-15], label=f"{number_of_lasers}, NA={settings_dict['NA_1']:.2f}",
             linewidth=0.3)
    plt.legend()




start_time = time.time()
for file_name in ["Apof_in_ice", "myoglobin"]:
    for shot_noise in [False, True]:
        i = 0
        fig_1 = plt.figure(figsize=(15, 10))
        fig_2 = plt.figure(figsize=(15, 10))
        fig_3 = plt.figure(figsize=(15, 10))
        for index, r in df.iterrows():
            settings_dict = r.to_dict()
            settings_dict['shot_noise'] = shot_noise
            settings_dict['file_name'] = file_name
            settings_dict['fig_1'] = fig_1
            settings_dict['fig_2'] = fig_2
            settings_dict['fig_3'] = fig_3
            settings_dict['i'] = i
            settings_dict['image_resolution_reduction_factor'] = 4
            print(settings_dict)
            figure_file_name = f(**settings_dict)
            print(time.time() - start_time)
            i += 1

        if file_name == "Apof_in_ice":
            shortened_name = "a"
        else:
            shortened_name = "m"
        title_a = f"{shortened_name} shot_noise={shot_noise} polarization_pies={settings_dict['polarization_pies']:.2f}, E_0={settings_dict['E_0']:.0f}, alpha_cavity_deviation_degrees={settings_dict['alpha_cavity_deviation_degrees']:.0f}"
        title_c = f"Cs_mm={settings_dict['Cs_mm']:.2f}, n_electrons_per_ang2={settings_dict['n_electrons']:.0f}, defocus_nm={settings_dict['defocus_nm']:.0f}"
        title_d = f"shot_noise={shot_noise}"
        plt.suptitle(f"{title_a}, {title_c}\n{title_d}")

        figure_file_name_1 = f"Figures\\examples\\{title_a}_real.png"
        figure_file_name_2 = f"Figures\\examples\\{title_a}_fourier.png"
        figure_file_name_3 = f"Figures\\examples\\{title_a}_histogram.png"

        # Set the fig_1 to be the active figure:
        plt.figure(fig_1.number)
        plt.suptitle(f"{title_a}, {title_c}\n{title_d}")
        plt.savefig(figure_file_name_1)

        plt.figure(fig_2.number)
        plt.suptitle(f"{title_a}, {title_c}\n{title_d}")
        plt.savefig(figure_file_name_2)

        plt.figure(fig_3.number)
        plt.suptitle(f"{title_a}, {title_c}\n{title_d}")
        plt.savefig(figure_file_name_3)
        print("\n\n\nkakikaki\n\n\n")

print("Done")

