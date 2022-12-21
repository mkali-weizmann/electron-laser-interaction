import time

import numpy as np
import matplotlib.pyplot as plt
from microscope import *
from scipy import integrate
from time import sleep
from scipy.special import erf, erfi, erfc

l_1 = 1064e-9
l_2 = 532e-9
k_l = l2k(l_2)
omega_l = k_l * C_LIGHT
delta_omega = l2w(l_2) - l2w(l_1)
beta_electron = 0.7765254931470489
gamma = beta2gamma(beta_electron)
NA = 0.1
w_0 = l_2 / (pi * NA)
x_R = x_R_gaussian(w_0, l_2)
beta_lattice = (l_1 - l_2) / (l_1 + l_2)
theta_polarization = 0
alpha_cavity = np.arcsin(beta_lattice / beta_electron)
E_1 = 6.46e7
E_2 = E_1 * (l_1 / l_2)
standard_dviations = 5
sufficient_lambda_fraction = 0.1
z_limit = standard_dviations * w_0
Z = np.linspace(-z_limit, z_limit, 1000) / np.cos(alpha_cavity)
prefactor = - 1 / H_BAR * E_CHARGE ** 2 / (2 * M_ELECTRON * beta2gamma(beta_electron) * beta_electron * C_LIGHT)
x_0 = 0
t_0 = 0
y_0 = 0
dr = l_2 / 1000

def z_t(t):
    return beta_electron * C_LIGHT * t


def t_z(z):
    return z / (beta_electron * C_LIGHT)


def w(x):
    return w_0 * np.sqrt(1 + (x / x_R) ** 2)


def R_inverse(x):
    return x / (x ** 2 + x_R ** 2)


def psi(x):
    return np.arctan(x / x_R)


def x_grating(x, y, z):
    return np.cos(k_l * (x + (y ** 2 + z ** 2) * R_inverse(x)) + psi(x))


def envelope(x, y, z):
    return w_0 / w(x) * np.exp(
        np.clip(-(y ** 2 + z ** 2) / (w(x) ** 2), a_min=-500, a_max=None))


def temporal_component(t):
    return np.cos(omega_l * t)


def rotated_gaussian_beam_A(x: [float, np.ndarray],
                            y: [float, np.ndarray],
                            z: [float, np.ndarray],
                            t: [float, np.ndarray],
                            return_vector: bool = True) -> Union[np.ndarray, float]:

    x_tilde = x * np.cos(alpha_cavity) - z * np.sin(alpha_cavity)
    z_tilde = x * np.sin(alpha_cavity) + z * np.cos(alpha_cavity)

    A_1 = gaussian_beam(x=x_tilde, y=y, z=z_tilde, E=E_1, lamda=l_1, NA=NA, t=t,
                        mode="potential")
    A_2 = gaussian_beam(x=x_tilde, y=y, z=z_tilde, E=E_2, lamda=l_2, NA=NA, t=t,
                        mode="potential")

    # This is not the electric potential, but rather only the amplitude factor that is shared among the different
    # components of the electromagnetic potential.
    A_scalar = A_1 + A_2

    # This is the rotated field vector: the vector is achieved by starting with a polarized light in the z axis,
    # then rotating it in the y-z plane by theta_polarization (from z towards y) and then rotating it in the
    # z-x plane by alpha_cavity (from x towards z).
    # (0, 0, 1) -> (0, sin(t), cos(t)) -> (sin(a)cos(t), sin(t), cos(a)cos(t))
    if not return_vector:
        return A_scalar
    else:
        A_vector = np.stack((-A_scalar * np.cos(theta_polarization) * np.sin(alpha_cavity),
                             A_scalar * np.sin(theta_polarization),
                             A_scalar * np.cos(theta_polarization) * np.cos(alpha_cavity)),
                            axis=-1)
        return A_vector


def A_z_integrand(Z, x, y, t):
    t_z_values = t_z(Z)
    T = t + t_z_values
    A_z_integrand_values = rotated_gaussian_beam_A(x, y, Z, T)
    return A_z_integrand_values


def Z_z(z):
    # Z_values = np.arange(start=min(-standard_dviations * w_0, z - standard_dviations * w_0),
    #                  stop=z,
    #                  step=l_2 * sufficient_lambda_fraction)
    # Z_values = np.linspace(z - 2 * standard_dviations * w_0, z, 1000)
    Z_values = Z[Z < z]
    return Z_values


def G_discrete_integral(z_idx, A_z):
    Z_values = Z[0:z_idx]
    A_z_integrand_values = A_z[0:z_idx]

    G_values = np.trapz(A_z_integrand_values, Z_values)

    # if len(Z_values) < 2:
    #     return 0
    # else:
    #     G_values = np.sum(A_z_integrand_values) * (Z_values[1] - Z_values[0])
    return G_values


def phi_integrand(z_idx,
                  A_integrand_values,
                  A_z_integrand_values,
                  A_z_integrand_values_dx,
                  A_z_integrand_values_dy,
                  A_z_integrand_values_dz):

    A_values = A_integrand_values[z_idx, :]

    G = G_discrete_integral(z_idx, A_z_integrand_values)
    G_dx = G_discrete_integral(z_idx, A_z_integrand_values_dx)
    G_dy = G_discrete_integral(z_idx, A_z_integrand_values_dy)
    G_dz = G_discrete_integral(z_idx+1, A_z_integrand_values_dz)

    grad_G = np.stack((G_dx - G, G_dy - G, G_dz - G), axis=-1) / dr

    integrand = np.sum(np.abs(A_values - grad_G)**2) - beta_electron**2 * np.sum(np.abs(A_values[2] - A_values[2])**2)

    return integrand, A_values, G, grad_G


def phi_integrand_array(t):
    # z = standard_dviations * w_0
    # Z = np.arange(start=-z, stop=z, step=l_2 * sufficient_lambda_fraction)
    # Z = np.linspace(start=-z, stop=z, num=1000)

    A_integrand_values = A_z_integrand(Z, x_0, y_0, t)
    A_z_integrand_values = A_integrand_values[:, 2]
    A_z_integrand_values_dx = A_z_integrand(Z, x_0+dr, y_0, t)[:, 2]
    A_z_integrand_values_dy = A_z_integrand(Z, x_0, y_0+dr, t)[:, 2]
    A_z_integrand_values_dz = A_z_integrand(Z, x_0, y_0, t - t_z(dr))[:, 2]

    phi_integrand_array_values = np.zeros(len(Z))
    A_values = np.zeros((len(Z), 3), dtype=np.complex128)
    G_values = np.zeros(len(Z), dtype=np.complex128)
    grad_G = np.zeros((len(Z), 3), dtype=np.complex128)
    for i, z in enumerate(Z):
        phi_integrand_array_values[i], A_values[i, :], G_values[i], grad_G[i, :] = phi_integrand(i, A_integrand_values, A_z_integrand_values,A_z_integrand_values_dx, A_z_integrand_values_dy, A_z_integrand_values_dz)

    return phi_integrand_array_values, Z


def phi_single_t(t):
    phi_integrand_array_values, Z = phi_integrand_array(t)
    return np.trapz(phi_integrand_array_values, Z, axis=0) * prefactor


def phi(t=None):
    if t is None:
        t = np.array([0, pi / (2 * delta_omega), pi / delta_omega])

    phi_values = np.zeros(len(t))
    for i, T in enumerate(t):
        phi_values[i] = phi_single_t(T)
    return phi_values, t


def extract_amplitude_attenuation(phi_values):
    if len(phi_values) == 3:
        phi_0 = 1 / 2 * (phi_values[0] + phi_values[2])
        varphi = np.arctan2(phi_values[0] - phi_0, phi_values[1] - phi_0)
        sin_varphi = np.sin(varphi)
        if np.abs(sin_varphi) < 1e-3:
            A = (phi_values[1] - phi_0) / np.cos(varphi)
        else:
            A = (phi_values[0] - phi_0) / sin_varphi
        phase_and_amplitude_mask = jv(0, A) * np.exp(1j * phi_0)
    else:
        phase_factor = np.exp(1j * phi_values)
        energy_bands = np.fft.fft(phase_factor, norm='forward')
        phase_and_amplitude_mask = energy_bands[0]
    return phase_and_amplitude_mask


# t_100 = np.linspace(0, 6 * pi / delta_omega, 100)
# phi_values_100, t_100 = phi(0, 0, t_100)

phi_values_3, t_3 = phi()


phase_and_amplitude_mask_values_3 = extract_amplitude_attenuation(phi_values_3)
print(np.angle(phase_and_amplitude_mask_values_3))
print(np.abs(phase_and_amplitude_mask_values_3))


# phase_and_amplitude_mask_values_100 = extract_amplitude_attenuation(phi_values_100)
# print(np.abs(phase_and_amplitude_mask_values_100))
# print(np.angle(phase_and_amplitude_mask_values_100))

# plt.plot(t_3, phi_values_3)
# plt.plot(t_100, phi_values_100)
# plt.show()