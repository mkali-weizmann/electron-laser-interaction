from microscope import *
from microscope import k_of_beta
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*DataFrame concatenation with empty or all-NA entries is deprecated.*",
        category=FutureWarning,
    )
a = np.linspace(0, 1, 5, endpoint=True)
b = np.linspace(0, 1, 5, endpoint=False)
import scipy.integrate as integrate

print_parameters = False
NA_1 = 15.0000000000e-02
second_laser = True
ring_cavity = False
polarization_pies = 0.5
E_0 = 3.0000000000e+02
defocus_nm = 0.0000000000e+00
Cs_mm = 0.0000000000e+00
n_electrons = 20
auto_set_power = True
power_1 = 6.7000000000e+04
focal_length_mm = 3.0000000000e+00
alpha_cavity_deviation_degrees = 0.0000000000e+00
resolution = 1064
n_z = 1000

from tqdm import tqdm
for NA_1 in tqdm([0.05, 0.1, 0.15], desc='NA', leave=True):  #
    # for polarization_pies in tqdm([0], desc='Polarization', leave=False, position=1):  #  , 0.5
    for second_laser in tqdm([False, True], desc='Second laser', leave=False, position=2):  # True,
#             if not second_laser and polarization_pies == 0:
#                 continue
        first_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=True)

        if second_laser:
            power_2 = -1
        else:
            power_2 = None
        l_1 = 1064e-9
        l_2 = 532e-9

        if auto_set_power:
            power_1 = find_power_for_phase(starting_power=1e4, power_2=power_2, cavity_type='numerical', print_progress=False,
                                           NA_1=NA_1, n_z=n_z, theta_polarization=polarization_pies * np.pi,
                                           ring_cavity=ring_cavity, alpha_cavity_deviation=alpha_cavity_deviation_degrees / 360 * 2 * np.pi)


        input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=r'data\static data\apof_in_ice.mrc')
        input_wave = WaveFunction(E_0=input_wave_full.E_0,
                                  psi=input_wave_full.psi[280:280 + resolution, 30:30 + resolution],
                                  coordinates=CoordinateSystem(dxdydz=input_wave_full.coordinates.dxdydz,
                                                               n_points=(resolution, resolution)))

        cavity = CavityNumericalPropagator(l_1=l_1, l_2=l_2, power_1=power_1, power_2=power_2, NA_1=NA_1,
                                           ring_cavity=ring_cavity,
                                           alpha_cavity_deviation=alpha_cavity_deviation_degrees / 360 * 2 * np.pi,
                                           theta_polarization=polarization_pies * np.pi,
                                           n_z=n_z, ignore_past_files=False, print_progress=False)
        second_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=False)
        aberration_propagator = AberrationsPropagator(Cs=Cs_mm * 1e-3, defocus=defocus_nm * 1e-9, astigmatism_parameter=0,
                                                      astigmatism_orientation=0)
        M = Microscope([first_lens, cavity, second_lens, aberration_propagator], n_electrons_per_square_angstrom=n_electrons)
        pic = M.take_a_picture(input_wave)


        mask = cavity.load_or_calculate_phase_and_amplitude_mask(M.step_of_propagator(cavity).output_wave)
        middle_phase_mask_value = mask[mask.shape[0] // 2, mask.shape[1] // 2]
        attenuation_factor = np.abs(middle_phase_mask_value)
        phase_factor = np.real(np.angle(middle_phase_mask_value))

        lambda_electron = 2 * np.pi / k_of_beta(M.step_of_propagator(cavity).input_wave.beta)
        focal_plane_fourier_limits = 2 * np.pi * np.array(M.step_of_propagator(cavity).input_wave.coordinates.limits) / (lambda_electron * focal_length_mm * 1e-3)
        repetitive_title = rf"Cavity NA = {NA_1}, attenuating = {second_laser}"  # , $\theta_{{\text{{polarization}}}} = {polarization_pies * 180:.0f}^{{\circ}}$
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(10, 10))
        mask_phase = ax_1.imshow(np.angle(mask),
                                 extent=focal_plane_fourier_limits,
                                 cmap='grey')
        ax_1.set_title(f"Contrast Transfer Function\n{repetitive_title}")
        ax_1.set_xlabel(r"Fourier plane $k_{x}$  $[\frac{1}{m}]$")
        ax_1.set_ylabel(r"Fourier plane $k_{y}$  $[\frac{1}{m}]$")
        fig_1.colorbar(mask_phase, ax=ax_1, fraction=0.046, pad=0.04)
        plt.savefig(f"Figures\\examples\\CTF-{NA_1*100:.0f}-{polarization_pies}-{second_laser}-{n_z}.png")
        plt.show()

        fig_2, ax_2 = plt.subplots(1, 1, figsize=(10, 10))
        im_intensity = ax_2.imshow(np.flip(pic.values), extent=input_wave.coordinates.limits, cmap='grey', vmax=9)
        plt.colorbar(im_intensity, ax=ax_2, fraction=0.046, pad=0.04)
        ax_2.set_title(f"Final image\n{repetitive_title}")
        plt.savefig(f"Figures\\examples\\final_image-{NA_1*100:.0f}-{polarization_pies}-{second_laser}-{n_z}.png")
        plt.show()

        fig_3, ax_3 = plt.subplots(1, 1, figsize=(10, 10))
        mask_attenuation = ax_3.imshow(np.abs(mask) ** 2, extent=focal_plane_fourier_limits,
                                           cmap='grey')
        ax_3.set_title(f"mask - intensity transfer\n{repetitive_title}")
        plt.colorbar(mask_attenuation, ax=ax_3, fraction=0.046, pad=0.04)
        plt.savefig(f"Figures\\examples\\attenuation_mask-{NA_1*100:.0f}-{polarization_pies}-{second_laser}-{n_z}.png")
        plt.show()

# plt.savefig(f"Figures\\examples\\kaki.png")
# print(f"{cavity.power_1=:.2e}, {cavity.power_2=:.2e}")


# from tqdm import tqdm
# print_parameters = True
# defocus_nm = 0.0000000000e+00
# Cs_mm = 0.0000000000e+00
# n_electrons = 20
# power_1 = 1.8648123635e+04
# focal_length_mm = 3.0000000000e+00
# n_z = None
# for resolution in tqdm([1024, 512], desc='Resolution'):
#     for E_0 in tqdm([300, 80, 100, 200, 250], desc='E_0'):
#         for NA_1 in tqdm([0.1, 0.2, 0.02, 0.05, 0.08, 0.15], desc='NA_1'):
#             for second_laser in tqdm([True, False], desc='Second laser'):
#                 for ring_cavity in tqdm([True, False], desc='Ring cavity'):
#                     for polarization_pies in tqdm(list(np.linspace(0, 1/2, 5, endpoint=True)), desc='Polarization'):
#
#                         first_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=True)
#                         if second_laser:
#                             power_2 = -1
#                         else:
#                             power_2 = None
#                         l_1 = 1064e-9
#                         l_2 = 532e-9
#
#                         # DELETE THIS LINE, IT'S HERE ONLY FOR THE PROPOSAL DEFENSE:
#                         if second_laser:
#                             ring_cavity = True
#                         else:
#                             ring_cavity = False
#
#                         input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=r'data\static data\apof_in_ice.mrc')
#                         input_wave = WaveFunction(E_0=input_wave_full.E_0,
#                                                   psi=input_wave_full.psi[100:100 + resolution, 100:100 + resolution],
#                                                   coordinates=CoordinateSystem(dxdydz=input_wave_full.coordinates.dxdydz,
#                                                                                n_points=(resolution, resolution)))
#
#                         # dummy_sample = SamplePropagator(dummy_potential=f'letters_{N_POINTS}',
#                         #                                 coordinates_for_dummy_potential=CoordinateSystem(axes=(input_coordinate_system.x_axis,
#                         #                                                                                        input_coordinate_system.y_axis,
#                         #                                                                                        np.linspace(-5e-10, 5e-10, 2)
#                         #                                                                                        )))
#
#                         cavity = CavityNumericalPropagator(l_1=l_1, l_2=l_2, power_1=power_1, power_2=power_2, NA_1=NA_1,
#                                                            ring_cavity=ring_cavity,
#                                                            alpha_cavity_deviation=alpha_cavity_deviation_degrees / 360 * 2 * np.pi,
#                                                            theta_polarization=polarization_pies * np.pi,
#                                                            n_z=n_z, ignore_past_files=False, print_progress=False)
#                         second_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=False)
#                         aberration_propagator = AberrationsPropagator(Cs=Cs_mm * 1e-3, defocus=defocus_nm * 1e-9, astigmatism_parameter=0,
#                                                                       astigmatism_orientation=0)
#                         M = Microscope([first_lens, cavity, second_lens, aberration_propagator], n_electrons_per_square_angstrom=n_electrons)
#                         pic = M.take_a_picture(input_wave)