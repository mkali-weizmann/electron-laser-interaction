from microscope import *
NA_1=0.10
second_laser=False
ring_cavity=False
polarization_pies=0.50
E_0=300.0
defocus_nm=0.00
Cs_mm=0.00
n_electrons=20
power_1=1.2723e+04
focal_length_mm=3.00
alpha_cavity_deviation_degrees=0.0
resolution=256
n_z=None

first_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=True)
if second_laser:
    power_2 = -1
else:
    power_2 = None
l_1 = 1064e-9
l_2 = 532e-9

# DELETE THIS LINE, IT'S HERE ONLY FOR THE PROPOSAL DEFENSE:
if second_laser:
    ring_cavity = True
else:
    ring_cavity = False

input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=r'data\static data\apof_in_ice.mrc')
input_wave = WaveFunction(E_0=input_wave_full.E_0,
                          psi=input_wave_full.psi[100:100 + resolution, 100:100 + resolution],
                          coordinates=CoordinateSystem(dxdydz=input_wave_full.coordinates.dxdydz,
                                                       n_points=(resolution, resolution)))

# dummy_sample = SamplePropagator(dummy_potential=f'letters_{N_POINTS}',
#                                 coordinates_for_dummy_potential=CoordinateSystem(axes=(input_coordinate_system.x_axis,
#                                                                                        input_coordinate_system.y_axis,
#                                                                                        np.linspace(-5e-10, 5e-10, 2)
#                                                                                        )))

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

fig, ax = plt.subplots(2, 3, figsize=(21, 14))
mask = cavity.load_or_calculate_phase_and_amplitude_mask(M.step_of_propagator(cavity).output_wave)
middle_phase_mask_value = mask[mask.shape[0] // 2, mask.shape[1] // 2]
attenuation_factor = np.abs(middle_phase_mask_value)
phase_factor = np.real(np.angle(middle_phase_mask_value))
#     plt.suptitle(f' phase_factor over pi{phase_factor / np.pi:.2f} {attenuation_factor=:.2f}')
#     plt.imshow(pic_2f_a.values, extent=pic_2f_a.coordinates.limits)
#     plt.colorbar()

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

mask_attenuation = ax[1, 2].imshow(np.abs(mask) ** 2, extent=M.step_of_propagator(cavity).input_wave.coordinates.limits)
ax[1, 2].set_title(f"mask - intensity transfer")
divider = make_axes_locatable(ax[1, 2])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(mask_attenuation, cax=cax, orientation="vertical")

plt.savefig(f"Figures\\examples\\kaki.png")
plt.show()