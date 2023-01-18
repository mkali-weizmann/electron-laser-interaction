from microscope import *

# %% global parameters
l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
N_POINTS = 256  # Resolution of image
pixel_size = 1e-10
# %%
input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)),
                          coordinates=input_coordinate_system,
                          E0=Joules_of_keV(300))

dummy_sample = SamplePropagator(dummy_potential=f'letters_{N_POINTS}',
                                coordinates_for_dummy_potential=CoordinateSystem(axes=(input_coordinate_system.x_axis,
                                                                                       input_coordinate_system.y_axis,
                                                                                       np.linspace(-5e-10, 5e-10, 2)
                                                                                       )))


first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)
second_lens = LensPropagator(focal_length=3.3e-3, fft_shift=False)
for ring_cavity in [True, False]:
    for cavity_propagator in [CavityNumericalPropagator]:   # CavityAnalyticalPropagator
        for NA_1 in [0.05, 0.08, 0.1, 0.15, 0.2]:  #
            for E_2 in [-1, None]:  # (with and without a second laser)
                for Cs in [0, 1e-8, 5e-8]:
                    for defocus in [-1e-7, -5e-8, -1e-8, 1e-8, 5e-8, 1e-8]:
                        if cavity_propagator.__name__ == 'CavityAnalyticalPropagator':
                            cavity_name = 'Analytical'
                        else:
                            cavity_name = 'Numerical'
                        if E_2 == -1:
                            two_lasers = 'True'
                        else:
                            two_lasers = 'False'
                        title = f'NA_1={NA_1}, ring_cavity={ring_cavity}, type={cavity_name},' \
                                f'two_lasers={two_lasers}, Cs={Cs}, defocus={defocus}'
                        print(f"Calculating {title}...", end='\r')
                        C = cavity_propagator(NA_1=NA_1, ring_cavity=ring_cavity, E_2=E_2, ignore_past_files=True)
                        aberrations_propagator = AberrationsPropagator(Cs=Cs, defocus=defocus)
                        M = Microscope([dummy_sample, first_lens, C, second_lens, aberrations_propagator])
                        pic = M.take_a_picture(first_wave)
                        plt.imshow(pic.values, extent=input_coordinate_system.limits)
                        plt.title(title)
                        plt.savefig(f'Figures\\{title}.png')
                        print(f"Calculating {title} - finished")

print("finished")

# %% compare phase masks of the numerical and analytical cavities:
N_POINTS = 64  # Resolution of image
l_1 = 1064e-9
l_2 = 1050e-9
input_coordinate_system = CoordinateSystem(dxdydz=(pixel_size, pixel_size), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)),
                          coordinates=input_coordinate_system,
                          E0=Joules_of_keV(1000))

first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)

# E_1 = 3.5e8
# E_1_n = 187045230
cavity_2f_analytical = CavityAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=False, starting_E_in_auto_E_search=1e3)
cavity_2f_analytical_ring = CavityAnalyticalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=True)
cavity_2f_numerical = CavityNumericalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=False, ignore_past_files=False)
cavity_2f_numerical_ring = CavityNumericalPropagator(l_1=l_1, l_2=l_2, E_1=-1, NA_1=NA_1, ring_cavity=True, ignore_past_files=True)

fourier_plane_wave = first_lens.propagate(first_wave)

phase_and_amplitude_mask_a = cavity_2f_analytical.phase_and_amplitude_mask(fourier_plane_wave)
phase_and_amplitude_mask_a_ring = cavity_2f_analytical_ring.phase_and_amplitude_mask(fourier_plane_wave)
phase_and_amplitude_mask_n = cavity_2f_numerical.phase_and_amplitude_mask(fourier_plane_wave)
phase_and_amplitude_mask_n_ring = cavity_2f_numerical_ring.phase_and_amplitude_mask(fourier_plane_wave)


# fig = plt.figure(figsize=(12, 12))
# ax1 = fig.add_subplot(221)
# im1 = ax1.imshow(np.real(np.angle(phase_and_amplitude_mask_a)), extent=fourier_plane_wave.coordinates.limits)
# divider = make_axes_locatable(ax1)
# ax1.set_title('analytical - phase')
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im1, cax=cax, orientation='vertical')
#
# ax2 = fig.add_subplot(222)
# im2 = ax2.imshow(np.abs(phase_and_amplitude_mask_a)**2, extent=fourier_plane_wave.coordinates.limits)
# divider = make_axes_locatable(ax2)
# ax2.set_title('analytical - intensity')
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im2, cax=cax, orientation='vertical')
#
# ax3 = fig.add_subplot(223)
# im3 = ax3.imshow(np.real(np.angle(phase_and_amplitude_mask_n)), extent=fourier_plane_wave.coordinates.limits)
# divider = make_axes_locatable(ax3)
# ax3.set_title('numerical - phase')
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im3, cax=cax, orientation='vertical')
#
# ax4 = fig.add_subplot(224)
# im4 = ax4.imshow(np.abs(phase_and_amplitude_mask_n)**2, extent=fourier_plane_wave.coordinates.limits)
# divider = make_axes_locatable(ax4)
# ax4.set_title('numerical - intensity')
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im4, cax=cax, orientation='vertical')
# plt.show()

# # %%
#
#
#
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(221)
im1 = ax1.imshow(np.real(np.angle(phase_and_amplitude_mask_a_ring)), extent=fourier_plane_wave.coordinates.limits)
divider = make_axes_locatable(ax1)
ax1.set_title('analytical - phase')
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(222)
im2 = ax2.imshow(np.abs(phase_and_amplitude_mask_a_ring)**2, extent=fourier_plane_wave.coordinates.limits)
divider = make_axes_locatable(ax2)
ax2.set_title('analytical - intensity')
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax3 = fig.add_subplot(223)
im3 = ax3.imshow(np.real(np.angle(phase_and_amplitude_mask_n_ring)), extent=fourier_plane_wave.coordinates.limits)
divider = make_axes_locatable(ax3)
ax3.set_title('numerical - phase')
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')

ax4 = fig.add_subplot(224)
im4 = ax4.imshow(np.abs(phase_and_amplitude_mask_n_ring)**2, extent=fourier_plane_wave.coordinates.limits)
divider = make_axes_locatable(ax4)
ax4.set_title('numerical - intensity')
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im4, cax=cax, orientation='vertical')
plt.show()
# %%
a = np.load("Data Arrays\\Static Data\\sight chart.png", allow_pickle=True)
#
#
#
