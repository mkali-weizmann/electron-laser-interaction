import matplotlib.pyplot as plt

from microscope import *

# %%
N_POINTS = 1024
input_coordinate_system = CoordinateSystem(lengths=(52.4e-9, 52.4e-9), n_points=(N_POINTS, N_POINTS))
first_wave = WaveFunction(psi=np.ones((N_POINTS, N_POINTS)),
                          coordinates=input_coordinate_system,
                          E0=KeV2Joules(300))

dummy_sample = SamplePropagator(dummy_potential='letters',
                                axes=tuple([first_wave.coordinates.axes[0],
                                            first_wave.coordinates.axes[1],
                                            np.linspace(-5e-10, 5e-10, 2)]))
first_lens = LensPropagator(focal_length=3.3e-3, fft_shift=True)

second_lens = LensPropagator(focal_length=3.3e-3, fft_shift=False)
cavity_2f = Cavity2FrequenciesAnalyticalPropagator(l_1=1064e-9, l_2=532e-9, E_1=3.42e-7, E_2=-1, NA=0.05)
cavity_1f = Cavity2FrequenciesAnalyticalPropagator(l_1=1064e-9, l_2=None, E_1=5.06e-7, E_2=None, NA=0.05)

M_2f = Microscope([dummy_sample, first_lens, cavity_2f, second_lens])
M_2f.propagate(first_wave)
M_1f = Microscope([dummy_sample, first_lens, cavity_1f, second_lens])
M_1f.propagate(first_wave)
M_no_f = Microscope([dummy_sample, first_lens, second_lens])
M_no_f.propagate(first_wave)

sub_box_size = 250
X, Y = M_2f.propagation_steps[2].input_wave.coordinates.X_grid, M_2f.propagation_steps[2].input_wave.coordinates.Y_grid
limits = [x * sub_box_size / N_POINTS for x in M_2f.propagation_steps[2].input_wave.coordinates.limits]

# %%
# M.plot_step(0, title="specimen - input wave (upper) and output (lower) wave", file_name="specimen.png")
# M.plot_step(1, title="first lens - input wave (upper) and output (lower) wave", file_name="first_lens.png")
# M.plot_step(2, title="cavity_2f - input wave (upper) and output (lower) wave", file_name="cavity_2f.png")
# M.plot_step(3, title="second lens wave - input (upper) and output (lower) wave", file_name="second_lens.png")


phase_map = np.angle(M_2f.propagation_steps[0].output_wave.psi[0:sub_box_size, 0:sub_box_size])
plt.title('total phase delay given by the sample')
plt.imshow(phase_map - np.max(phase_map), extent=M_2f.propagation_steps[0].output_wave.coordinates.limits)
plt.savefig('Figures\phase_map_2f.png')
plt.colorbar()
plt.show()

plt.imshow(np.abs(np.flip(M_2f.propagation_steps[3].output_wave.psi[-sub_box_size:, -sub_box_size:], axis=(0, 1))) ** 2, extent=M_2f.propagation_steps[3].output_wave.coordinates.limits)
plt.colorbar()
plt.title('double laser cavity')
plt.savefig('Figures\zoomed_final_image_2f.png')
plt.show()

plt.imshow(np.abs(np.flip(M_1f.propagation_steps[3].output_wave.psi[-sub_box_size:, -sub_box_size:], axis=(0, 1))) ** 2, extent=M_1f.propagation_steps[3].output_wave.coordinates.limits)
plt.colorbar()
plt.title('single laser cavity')
plt.savefig('Figures\zoomed_final_image_1f.png')
plt.show()

plt.imshow(np.abs(np.flip(M_no_f.propagation_steps[2].output_wave.psi[-sub_box_size:, -sub_box_size:], axis=(0, 1))) ** 2, extent=M_1f.propagation_steps[2].output_wave.coordinates.limits)
plt.colorbar()
plt.title('no cavity')
plt.savefig('Figures\zoomed_final_image_no_f.png')
plt.show()


phi_0 = cavity_2f.phi_0(X, Y, beta_electron=E2beta(M_2f.propagation_steps[2].input_wave.E0))
constant_phase_shift = cavity_2f.phase_shift(phi_0=phi_0, X_grid=X, Y_grid=Y)
attenuation_factor = cavity_2f.attenuation_factor(beta_electron=E2beta(M_2f.propagation_steps[2].input_wave.E0),
                                                  phi_0=phi_0, X_grid=X, Y_grid=Y)

plt.imshow(constant_phase_shift, extent=limits)
plt.title(r'phase shift - 2f')
plt.colorbar()
plt.savefig(r'Figures\phase_shift_2f.png')
plt.show()

plt.imshow(attenuation_factor, extent=limits)
plt.title(r'attenuation factor')
plt.colorbar()
plt.savefig(r'Figures\attenuation_factor.png')
plt.show()

phi_0_1f = cavity_1f.phi_0(X, Y, beta_electron=E2beta(M_1f.propagation_steps[2].input_wave.E0))
constant_phase_shift_1f = cavity_1f.phase_shift(phi_0=phi_0_1f, X_grid=X, Y_grid=Y)
plt.imshow(constant_phase_shift_1f, extent=limits)
plt.title(r'phase shift - 1f')
plt.colorbar()
plt.savefig(r'Figures\phase_shift_2f.png')
plt.show()


fig = plt.figure(figsize=(15, 15))
plt.imshow(np.abs(np.flip(M_1f.propagation_steps[3].output_wave.psi, axis=(0, 1))) ** 2, extent=M_1f.propagation_steps[3].output_wave.coordinates.limits)
plt.colorbar()
plt.title('single laser cavity')
plt.savefig(r'Figures\final_image_1f.png')
plt.show()

fig = plt.figure(figsize=(15, 15))
plt.imshow(np.abs(np.flip(M_2f.propagation_steps[3].output_wave.psi, axis=(0, 1))) ** 2, extent=M_2f.propagation_steps[3].output_wave.coordinates.limits)
plt.colorbar()
plt.title('double laser cavity')
plt.savefig(r'Figures\final_image_2f.png')
plt.show()

# %% Find optimal A for pi/2 phase shift - double frequency cavity_2f
# X, Y = M_1f.propagation_steps[2].input_wave.coordinates.grids
# N = 3
# As = np.linspace(5e-7, 5.1e-7, N)
# phases = np.zeros(N)
#
# for i, A in enumerate(As):
#     print(i)
#     cavity = CavityDoubleFrequencyPropagator(l_1=1064e-9, l_2=532e-9, E_1=A, NA=0.05, E_2=None)
#     M = Microscope([dummy_sample, first_lens, cavity], print_progress=True)
#     last_wave = M.take_a_picture(input_wave=first_wave)
#     phi_0 = cavity.phi_0(X, Y, beta_electron=E2beta(M.propagation_steps[2].input_wave.E0))
#     constant_phase_shift = cavity.phase_shift(phi_0=phi_0, X_grid=X, Y_grid=Y)
#     phases[i] = constant_phase_shift[512, 512]
#
# plt.plot(As, phases)
# plt.axhline(np.pi/2, color='r')
# plt.show()