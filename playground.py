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
second_laser = False
ring_cavity = False
polarization_pies = 0.5
E_0 = 3.0000000000e+02
defocus_nm = 0.0000000000e+00
Cs_mm = 3.2e-3
n_electrons = 20
auto_set_power = True
power_1 = 6.7000000000e+04
focal_length_mm = 6.8
alpha_cavity_deviation_degrees = 0.0000000000e+00
resolution = 2048
n_z = 1000

# Alternative wave function generation:
size = 8e-8
coorinates = CoordinateSystem(axes=(np.linspace(-size, size, resolution), np.linspace(-size, size, resolution)))
x_axis = coorinates.x_axis
y_axis = coorinates.y_axis
X, Y = np.meshgrid(x_axis, y_axis)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(X, Y)
theta_rounded = np.mod(theta, np.pi / 20)
fan = theta_rounded < (np.pi/40)
fan = np.where(r < size / 10, False, fan)
fan_int = fan.astype(int)
phase_object = np.exp(0.1j * np.pi * fan_int) + np.random.normal(0, 0.01) + np.random.normal(0, 0.01)
input_wave = WaveFunction(E_0=Joules_of_keV(E_0), psi=phase_object, coordinates=coorinates)
vmax = None
title_fs = 30
label_fs = 28

from tqdm import tqdm
fig_1, ax_1 = plt.subplots(1, 1, figsize=(10, 3.5))
for NA_1 in tqdm([0.05, 0.15], desc='NA', leave=True):  #
    # for polarization_pies in tqdm([0], desc='Polarization', leave=False, position=1):  #  , 0.5
    for second_laser in tqdm([False], desc='Second laser', leave=False, position=2):  # True,  # , True
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


        # input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=r'data\static data\apof_in_ice.mrc')
        # input_wave = WaveFunction(E_0=input_wave_full.E_0,
        #                           psi=input_wave_full.psi[280:280 + resolution, 30:30 + resolution],
        #                           coordinates=CoordinateSystem(dxdydz=input_wave_full.coordinates.dxdydz,
        #                                                        n_points=(resolution, resolution)))

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

        fft_freq_x = np.fft.fftfreq(
            input_wave.psi.shape[0], input_wave.coordinates.dx
        )  # this is f and not k
        fft_freq_y = np.fft.fftfreq(
            input_wave.psi.shape[1], input_wave.coordinates.dy
        )  # this is f and not k
        fft_freq_x, fft_freq_y = np.fft.fftshift(fft_freq_x), np.fft.fftshift(
            fft_freq_y
        )
        aberration_mask = M.propagators[-1].aberrations_mask(fft_freq_x, fft_freq_y, input_wave.E_0)

        mask = cavity.load_or_calculate_phase_and_amplitude_mask(M.step_of_propagator(cavity).output_wave)
        middle_phase_mask_value = mask[mask.shape[0] // 2, mask.shape[1] // 2]
        attenuation_factor = np.abs(middle_phase_mask_value)
        phase_factor = np.real(np.angle(middle_phase_mask_value))

        lambda_electron = 2 * np.pi / k_of_beta(M.step_of_propagator(cavity).input_wave.beta)
        focal_plane_fourier_limits = 2 * np.pi * np.array(M.step_of_propagator(cavity).input_wave.coordinates.limits) / (lambda_electron * focal_length_mm * 1e-3) / 1e10
        repetitive_title = rf"Cavity NA = {NA_1}"  # , $\theta_{{\text{{polarization}}}} = {polarization_pies * 180:.0f}^{{\circ}}$
        mask_phase_array = np.angle(mask) + np.angle(aberration_mask)
        CTF = np.cos(mask_phase_array)**2
        aberrations_phase = M.propagators[-1]

        # Angular (radial) average of the CTF as a function of the radial spatial frequency k.
        from scipy.interpolate import RegularGridInterpolator
        k_x_axis = np.linspace(focal_plane_fourier_limits[0], focal_plane_fourier_limits[1], CTF.shape[1])
        k_y_axis = np.linspace(focal_plane_fourier_limits[2], focal_plane_fourier_limits[3], CTF.shape[0])

        # Artificially increase the resolution of the CTF image by 4x on each axis (linear
        # interpolation), so the radial bins below are populated by many more samples.
        upscale = 4
        k_x_fine = np.linspace(focal_plane_fourier_limits[0], focal_plane_fourier_limits[1], CTF.shape[1] * upscale)
        k_y_fine = np.linspace(focal_plane_fourier_limits[2], focal_plane_fourier_limits[3], CTF.shape[0] * upscale)
        ctf_interp = RegularGridInterpolator((k_y_axis, k_x_axis), CTF, method='linear')
        K_Y, K_X = np.meshgrid(k_y_fine, k_x_fine, indexing='ij')
        CTF_fine = ctf_interp((K_Y, K_X))
        K_radial = np.sqrt(K_X ** 2 + K_Y ** 2)

        n_bins = min(CTF_fine.shape) // 2
        k_bins = np.linspace(0, K_radial.max(), n_bins + 1)
        ctf_sum, _ = np.histogram(K_radial.ravel(), bins=k_bins, weights=CTF_fine.ravel())
        ctf_count, _ = np.histogram(K_radial.ravel(), bins=k_bins)
        CTF_radial = ctf_sum / np.maximum(ctf_count, 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        # Apply a narrow Gaussian smoothing (sigma = 5 samples) only within 1e-3 < k < 1e-1.
        from scipy.ndimage import gaussian_filter1d
        smooth_range = (k_centers >= 1e-3) & (k_centers <= 5e-2)
        CTF_radial[smooth_range] = gaussian_filter1d(CTF_radial[smooth_range], sigma=1.1)

        ax_1.semilogx(k_centers, CTF_radial, label=rf"NA = {NA_1}")

ax_1.axvline(1 / 20, color='tab:blue', linestyle='--')
ax_1.axvline(1 / 60, color='tab:orange', linestyle='--')
ax_1.set_title("Angular-averaged Contrast Transfer Function", fontsize=title_fs)
ax_1.set_xlabel(r"$s\ \left[A^{-1}\right]$", fontsize=label_fs)
ax_1.set_ylabel("CTF (angular average)", fontsize=label_fs)
ax_1.grid(True, which='both', alpha=0.3)
ax_1.tick_params(axis='both', which='major', labelsize=label_fs)
ax_1.legend(fontsize=label_fs)
ax_1.set_xlim(1e-3, 1)
ax_1.set_ylim(0, 1.1)
# Pad the first x tick so 10^-3 doesn't collide with the 0.0 y tick.
ax_1.tick_params(axis='x', which='major', pad=10)
fig_1.tight_layout()
plt.savefig(f"Figures\\examples\\dummy sample\\CTF-radial-{polarization_pies}-{second_laser}-{n_z}.png", bbox_inches='tight')
plt.show()
# %%

from tqdm import tqdm
for NA_1 in tqdm([0.05, 0.15], desc='NA', leave=True):  #
    # for polarization_pies in tqdm([0], desc='Polarization', leave=False, position=1):  #  , 0.5
    for second_laser in tqdm([False], desc='Second laser', leave=False, position=2):  # True,  # , True
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


        # input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=r'data\static data\apof_in_ice.mrc')
        # input_wave = WaveFunction(E_0=input_wave_full.E_0,
        #                           psi=input_wave_full.psi[280:280 + resolution, 30:30 + resolution],
        #                           coordinates=CoordinateSystem(dxdydz=input_wave_full.coordinates.dxdydz,
        #                                                        n_points=(resolution, resolution)))

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

        fft_freq_x = np.fft.fftfreq(
            input_wave.psi.shape[0], input_wave.coordinates.dx
        )  # this is f and not k
        fft_freq_y = np.fft.fftfreq(
            input_wave.psi.shape[1], input_wave.coordinates.dy
        )  # this is f and not k
        fft_freq_x, fft_freq_y = np.fft.fftshift(fft_freq_x), np.fft.fftshift(
            fft_freq_y
        )
        aberration_mask = M.propagators[-1].aberrations_mask(fft_freq_x, fft_freq_y, input_wave.E_0)

        mask = cavity.load_or_calculate_phase_and_amplitude_mask(M.step_of_propagator(cavity).output_wave)
        middle_phase_mask_value = mask[mask.shape[0] // 2, mask.shape[1] // 2]
        attenuation_factor = np.abs(middle_phase_mask_value)
        phase_factor = np.real(np.angle(middle_phase_mask_value))

        lambda_electron = 2 * np.pi / k_of_beta(M.step_of_propagator(cavity).input_wave.beta)
        focal_plane_fourier_limits = 2 * np.pi * np.array(M.step_of_propagator(cavity).input_wave.coordinates.limits) / (lambda_electron * focal_length_mm * 1e-3) / 1e10
        repetitive_title = rf"Cavity NA = {NA_1}"  # , $\theta_{{\text{{polarization}}}} = {polarization_pies * 180:.0f}^{{\circ}}$
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(10, 10))
        mask_phase_array = np.angle(mask) + np.angle(aberration_mask)
        CTF = np.cos(mask_phase_array)**2
        # Show only the central eighth-to-each-direction of the image (a quarter-width window
        # centered on k=0), and scale the extent accordingly.
        n_y, n_x = CTF.shape
        y0, y1 = 3 * n_y // 8, 5 * n_y // 8
        x0, x1 = 3 * n_x // 8, 5 * n_x // 8
        CTF_center = CTF[y0:y1, x0:x1]
        # Use a symmetric extent so both axes share the same range (and hence the same ticks).
        half_range = np.max(np.abs(focal_plane_fourier_limits)) / 4
        center_extent = [-half_range, half_range, -half_range, half_range]
        mask_phase = ax_1.imshow(CTF_center,
                                 extent=center_extent,
                                 cmap='grey')
        aberrations_phase = M.propagators[-1]
        ax_1.set_title(f"Contrast Transfer Function\n{repetitive_title}", fontsize=title_fs)
        ax_1.set_xlabel(r"$s_{x}\ \left[A^{-1}\right]$", fontsize=label_fs)
        ax_1.set_ylabel(r"$s_{y}\ \left[A^{-1}\right]$", fontsize=label_fs)
        # Force identical, symmetric limits and ticks on both axes (imshow's equal aspect
        # otherwise expands one axis and gives the two axes different auto-ticks).
        symmetric_ticks = np.arange(-0.3, 0.31, 0.3)
        ax_1.set_xlim(-half_range, half_range)
        ax_1.set_ylim(-half_range, half_range)
        ax_1.set_xticks(symmetric_ticks)
        ax_1.set_yticks(symmetric_ticks)
        ax_1.tick_params(axis='both', which='major', labelsize=label_fs)
        fig_1.colorbar(mask_phase, ax=ax_1, fraction=0.046, pad=0.04)
        fig_1.tight_layout()
        plt.savefig(f"Figures\\examples\\dummy sample\\CTF-{NA_1*100:.0f}-{polarization_pies}-{second_laser}-{n_z}.png", bbox_inches='tight')
        plt.show()