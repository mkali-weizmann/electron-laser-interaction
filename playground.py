import matplotlib as plt, mpld3
import numpy as np
import plotly.express as px
from microscope import *
from scipy import signal


def predicted_phi_0(beta, power, y, lambda_laser, w_0):
    spatial_envelope = np.exp(-2*y**2 / (w_0**2)) / w_0
    constant_coefficients = np.sqrt(8 / np.pi**3) * FINE_STRUCTURE_CONST / (beta * gamma_of_beta(beta))\
           * power * lambda_laser**2 / (M_ELECTRON * C_LIGHT**3)
    phi_0 = spatial_envelope * constant_coefficients
    return phi_0

def rho(theta, beta):
    return 1-2*(beta*np.cos(theta))**2

def predicted_phi(beta, power, y, lambda_laser, w_0, theta):
    return -predicted_phi_0(beta, power, y, lambda_laser, w_0) * (1/2)*(1+rho(theta, beta))


N_POINTS = 256  # Resolution of image
l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
pixel_size = 1e-10
theta_polarization = 0 * np.pi
E_0 = Joules_of_keV(20)
n_z = 10000

x, y, t = np.array([0]), np.array([0], dtype=np.float64), np.array([0], dtype=np.float64)

coordinate_system = CoordinateSystem(axes=(x, y))
input_wave = WaveFunction(np.array([[1]]), coordinate_system, E_0)
power_1 = 221457.63217337712

C = CavityNumericalPropagator(
    l_1=l_1, power_1=power_1, power_2=None, theta_polarization=theta_polarization, NA_1=NA_1, ring_cavity=False,
    debug_mode=True, ignore_past_files=True, n_z=n_z)

C_a = CavityAnalyticalPropagator(l_1=l_1, power_1=C.power_1, power_2=None, theta_polarization=theta_polarization, NA_1=NA_1,
                                 ring_cavity=False)

Z, X, Y, T = C.generate_coordinates_lattice(x, y, t, input_wave.beta)

A = C.rotated_gaussian_beam_A(Z, X, Y, T,
                                    beta_electron=input_wave.beta, save_to_file=False)
grad_G = C.grad_G(Z, X, Y, T,
                        beta_electron=input_wave.beta,
                        A_z=C.potential_envelope2vector_components(A, "z", input_wave.beta),
                        save_to_file=False,
                        )

integrand = (  # Notice that the elements of grad_G are (z, x, y) and not (x, y, z).
                            safe_abs_square(
                                C.potential_envelope2vector_components(A, "x", input_wave.beta) - grad_G[:, :, :, :,
                                                                                                   1])
                            + safe_abs_square(
                        C.potential_envelope2vector_components(A, "y", input_wave.beta) - grad_G[:, :, :, :, 2])
                            + safe_abs_square(
                        C.potential_envelope2vector_components(A, "z", input_wave.beta) - grad_G[:, :, :, :, 0])
                    ) - input_wave.beta ** 2 * safe_abs_square(
            C.potential_envelope2vector_components(A, "z", input_wave.beta) - grad_G[:, :, :, :, 0])


prefactor = (
                E_CHARGE ** 2 / (2 * M_ELECTRON * gamma_of_beta(input_wave.beta) * input_wave.beta *
                                              C_LIGHT * H_BAR)
        )

phi_e_100 = - prefactor * np.trapz(A ** 2 * np.sin(theta_polarization) ** 2 +
                                   (1-input_wave.beta**2) * A ** 2 * np.cos(theta_polarization) ** 2 +
                               grad_G[:, :, :, :, 1] ** 2
                                   , x=Z, axis=0)[0, 0, 0]


phi_0_paper = predicted_phi_0(input_wave.beta, C.power_1, y[0], C.l_1, C.w_0_min)
phi_0_manual = E_CHARGE ** 2 / (4 * M_ELECTRON * input_wave.beta * gamma_of_beta(input_wave.beta) * C_LIGHT * H_BAR) * \
               np.trapz(A**2, x=Z, axis=0)[0, 0, 0] * 2
phi_0_analytical = C_a.phi_0(input_wave)[0, 0]


phi_manual = prefactor * np.trapz(integrand, x=Z, axis=0)[0, 0, 0]
phi_original_function = C.phi(input_wave)[0, 0, 0]
phi_paper = predicted_phi(beta=input_wave.beta, power=C.power_1, y=y[0], lambda_laser=C.l_1, w_0=C.w_0_min,
                          theta=C.theta_polarization)
phi_analytical = phi_0_analytical * (1 / 2) * (1 + C_a.rho(input_wave.beta) * np.cos(input_wave.coordinates.X_grid * 4 * np.pi / C_a.l_1))[0, 0]

print(f"phi_0:\n{phi_0_manual=}\n{phi_0_paper=}\n{phi_0_analytical=}\n")

print(f"phi:\n{phi_paper=}\n{phi_original_function=}\n{phi_manual=}\n{phi_e_100=}\n{phi_analytical=}")
# print("\n".join([f"{n_zs[i]}: {phis[i]}\n" for i in range(len(n_zs))]))
# fig, ax = plt.subplots(figsize=(15, 15))
# plt.plot(Z[:, 0, 0, 0], np.real(A[:, 0, 0, 0]), label="real(A)")
# plt.plot(Z[:, 0, 0, 0], np.imag(A[:, 0, 0, 0]), label="imag(A)")
# plt.legend()
# plt.show()
# fig, ax = plt.subplots(figsize=(15, 15))
# plt.plot(Z[:, 0, 0, 0], np.real(grad_G[:, 0, 0, 0, 0])*1e7, label="real(dG/dz)")
# plt.plot(Z[:, 0, 0, 0], np.imag(grad_G[:, 0, 0, 0, 0])*1e7, label="image(dG/dz)")
# plt.plot(Z[:, 0, 0, 0], np.real(grad_G[:, 0, 0, 0, 1])*1e7, label="real(dG/dx)")
# plt.plot(Z[:, 0, 0, 0], np.imag(grad_G[:, 0, 0, 0, 1])*1e7, label="image(dG/dx)")
# plt.legend()
# plt.show()
fig, ax = plt.subplots(figsize=(15, 15))
# plt.plot(Z[:, 0, 0, 0], integrand[:, 0, 0, 0], label="integrand")
plt.plot(Z[:, 0, 0, 0], A[:, 0, 0, 0]**2, label="A**2")
plt.plot(Z[:, 0, 0, 0], grad_G[:, 0, 0, 0, 0]**2 , label=r"$\frac{d^{2}G}{dz^{2}}$")
plt.plot(Z[:, 0, 0, 0], grad_G[:, 0, 0, 0, 1]**2, "--", label=r"$\frac{d^{2}G}{dx^{2}}$")
# plt.plot(Z[:, 0, 0, 0], A[:, 0, 0, 0] ** 2 * np.sin(theta_polarization) ** 2 +
#                                    (1-input_wave.beta**2) * A[:, 0, 0, 0] ** 2 * np.cos(theta_polarization) ** 2 +
#                                grad_G[:, 0, 0, 0, 1] ** 2, label=["e_100 integrand"])
plt.legend()
plt.show()

