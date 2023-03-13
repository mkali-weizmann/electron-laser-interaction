from microscope import *
from scipy import integrate

def predicted_phi_0(beta, power, y, lambda_laser, w_0, ring_cavity):
    spatial_envelope = np.exp(-2*y**2 / (w_0**2)) / w_0
    if ring_cavity:
        constant_coefficients = np.sqrt(1 / (2 * np.pi ** 3)) * FINE_STRUCTURE_CONST / (beta * gamma_of_beta(beta)) \
                                * power * lambda_laser ** 2 / (M_ELECTRON * C_LIGHT ** 3)
    else:
        constant_coefficients = np.sqrt(8 / np.pi ** 3) * FINE_STRUCTURE_CONST / (beta * gamma_of_beta(beta)) \
                                * power * lambda_laser ** 2 / (M_ELECTRON * C_LIGHT ** 3)
    phi_0 = spatial_envelope * constant_coefficients
    return phi_0


def rho(theta, beta):
    return 1-2*(beta*np.cos(theta))**2


def predicted_phi(beta, power, y, lambda_laser, w_0, theta, ring_cavity):
    if ring_cavity:
        return -predicted_phi_0(beta, power, y, lambda_laser, w_0, ring_cavity)
    else:
        return -predicted_phi_0(beta, power, y, lambda_laser, w_0, ring_cavity) * (1/2)*(1+rho(theta, beta))


N_POINTS = 256  # Resolution of image
l_1 = 1064e-9
NA_1 = 0.05
theta_polarization = 0 * np.pi
E_0 = Joules_of_keV(300)
n_z = 5000
ring_cavity = False
alpha_cavity = np.pi/6

x, y, t = np.array([0], dtype=np.float64), np.array([0], dtype=np.float64), np.array([0], dtype=np.float64)

coordinate_system = CoordinateSystem(axes=(x, y))
input_wave = WaveFunction(np.array([[1]]), coordinate_system, E_0)
print(f"{input_wave.beta=}")
power_1 = 9.56e16

C = CavityNumericalPropagator(l_1=l_1, power_1=power_1, power_2=None, NA_1=NA_1, theta_polarization=theta_polarization,
                              alpha_cavity=alpha_cavity, ring_cavity=ring_cavity, ignore_past_files=True, n_z=n_z,
                              debug_mode=True)

# C_a = CavityAnalyticalPropagator(l_1=l_1, power_1=C.power_1, power_2=None, theta_polarization=theta_polarization, NA_1=NA_1,
#                                  ring_cavity=C.ring_cavity)

Z, X, Y, T = C.generate_coordinates_lattice(x, y, t, input_wave.beta)

A = C.rotated_gaussian_beam_A(Z, X, Y, T,
                                    beta_electron=input_wave.beta, save_to_file=False)
plt.imshow(A[:, :, 0, 0])
plt.ylabel("z")
plt.xlabel("x")
plt.show()
# G = C.G_gauge(C.potential_envelope2vector_components(A, 'z', input_wave.beta), Z)
#
# grad_G = C.grad_G(Z, X, Y, T,
#                         beta_electron=input_wave.beta,
#                         A_z=C.potential_envelope2vector_components(A, "z", input_wave.beta),
#                         save_to_file=False,
#                         )


# phi_integrand = (
#             (1 - input_wave.beta**2)
#             * safe_abs_square(C.potential_envelope2vector_components(A, "z", input_wave.beta) - grad_G[:, :, :, :, 0])
#             + safe_abs_square(C.potential_envelope2vector_components(A, "x", input_wave.beta) - grad_G[:, :, :, :, 1])
#             + safe_abs_square(C.potential_envelope2vector_components(A, "y", input_wave.beta) - grad_G[:, :, :, :, 2])
#         )
#
# prefactor = -(
#                 E_CHARGE ** 2 / (2 * M_ELECTRON * gamma_of_beta(input_wave.beta) * input_wave.beta *
#                                               C_LIGHT * H_BAR)
#         )

# phi = prefactor * np.trapz(phi_integrand, x=Z, axis=0)[0, 0, 0]
# print(f"{phi=:.2e}")

# fig, ax = plt.subplots(figsize=(15, 15))
min_idx = 0  # n_z//2
max_idx = -1  # (n_z*3)//4
Z_plot = Z[min_idx:max_idx, 0, 0, 0]
A_plot = A[min_idx:max_idx, 0, 0, 0]
# G_plot = G[min_idx:max_idx, 0, 0, 0]
# G_grad_plot = grad_G[min_idx:max_idx, 0, 0, 0, :]
# integrand_plot = phi_integrand[min_idx:max_idx, 0, 0]
# plt.plot(Z_plot, C.potential_envelope2vector_components(A_plot, 'z', input_wave.beta), label="A_z")
# plt.plot(Z_plot, C.potential_envelope2vector_components(A_plot, 'x', input_wave.beta), label="A_x")
# plt.plot(Z_plot, C.potential_envelope2vector_components(A_plot, 'y', input_wave.beta), label="A_y")
# plt.title("A")
# plt.show()
# plt.plot(Z_plot, G_grad_plot[:, 0], label="G_z")
# plt.plot(Z_plot, G_grad_plot[:, 1], label="G_x")
# plt.plot(Z_plot, G_grad_plot[:, 2], label="G_y")
# plt.title("G")
# plt.legend()
# plt.show()
# plt.plot(Z_plot, G_plot, label="G")
# print(f"{phi[0, 0, 0]:.2e}")
# plt.show()
# plt.plot(Z_plot, integrand_plot)
# plt.show()
# fig, ax = plt.subplots(figsize=(15, 15))
# plt.plot(Z[:, 0, 0, 0], np.real(grad_G[:, 0, 0, 0, 0])*1e7, label="real(dG/dz)")
# plt.plot(Z[:, 0, 0, 0], np.imag(grad_G[:, 0, 0, 0, 0])*1e7, label="image(dG/dz)")
# plt.plot(Z[:, 0, 0, 0], np.real(grad_G[:, 0, 0, 0, 1])*1e7, label="real(dG/dx)")
# plt.plot(Z[:, 0, 0, 0], np.imag(grad_G[:, 0, 0, 0, 1])*1e7, label="image(dG/dx)")
# plt.legend()
# plt.show()
# fig, ax = plt.subplots(figsize=(15, 15))
# plt.plot(Z[:, 0, 0, 0], integrand[:, 0, 0, 0], label="integrand")
# plt.plot(Z[:, 0, 0, 0], A[:, 0, 0, 0]**2, label="A**2")
# plt.plot(Z[:, 0, 0, 0], grad_G[:, 0, 0, 0, 0]**2 , label=r"$\frac{d^{2}G}{dz^{2}}$")
# plt.plot(Z[:, 0, 0, 0], grad_G[:, 0, 0, 0, 1]**2, "--", label=r"$\frac{d^{2}G}{dx^{2}}$")
# plt.plot(Z[:, 0, 0, 0], A[:, 0, 0, 0] ** 2 * np.sin(theta_polarization) ** 2 +
#                                    (1-input_wave.beta**2) * A[:, 0, 0, 0] ** 2 * np.cos(theta_polarization) ** 2 +
#                                grad_G[:, 0, 0, 0, 1] ** 2, label=["e_100 integrand"])
# plt.legend()
# plt.show()

