import time

import numpy as np
import matplotlib.pyplot as plt
from microscope import *
from scipy import integrate
from time import sleep
from scipy.special import erf, erfi, erfc


def plot_3d_complex(x_real, y_complex):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_real, np.real(y_complex), np.imag(y_complex))
    ax.set_xlabel('t')
    ax.set_ylabel('y_real')
    ax.set_zlabel('y_imag')
    plt.show()


l = 532e-9
k_l = l2k(l)
omega_l = k_l * C_LIGHT
beta = 0.5
gamma = beta2gamma(beta)
Na = 0.2
w_0 = l / (pi * Na)
x_R = x_R_gaussian(w_0, l)


def z_t(t):
    return beta * C_LIGHT * t


def t_z(z):
    return z / (beta * C_LIGHT)


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


def A_z(x, y, z, t):
    return envelope(x, y, z) * x_grating(x, y, z) * temporal_component(t)


def A_z_integrand(s, x, y, z, t):
    return A_z(x, y, z - beta * C_LIGHT * (t - s), s)


def A_z_integrand_variable_change(Z, x, y, z, t):
    return 1 / (beta * C_LIGHT) * A_z(x, y, Z, (Z - z) / (beta * C_LIGHT) + t)


def G_integral(x, y, z, t):
    return C_LIGHT * beta / omega_l * \
           integrate.quad(A_z_integrand, min([t_z(-16 * w_0), t - t_z(16 * w_0)]), t, args=(x, y, z, t), epsabs=1e-20)[
               0]


def s(t, n=1000):
    # min([t_z(-20 * w_0),
    return np.linspace(t - t_z(20 * w_0), t, n)


def G_discrete_integral(x, y, z, t):
    s_values = s(t)
    A_z_integrand_values = A_z_integrand(s_values, x, y, z, t)
    return C_LIGHT * beta / omega_l * np.trapz(A_z_integrand_values, s_values, axis=0)


def G_Osip(x, y, z, t):
    return C_LIGHT * beta / omega_l ** 2 * envelope(x, y, z) * x_grating(x, y, z) * np.sin(omega_l * t)


def I_integrand(s, x, y, z, t):
    sigma_tilde_squared = (w(x) / (beta * C_LIGHT)) ** 2
    R_tilde_inverse_squared = k_l * (beta * C_LIGHT) ** 2 * R_inverse(x)
    t_tilde = t - z / (beta * C_LIGHT)
    a_squared = 1 / (1 / sigma_tilde_squared + 1j * R_tilde_inverse_squared)
    return np.real(np.exp(-(s - t_tilde) ** 2 / a_squared + 1j * omega_l * s))


def I_numerical(x, y, z, t):
    return integrate.quad(I_integrand, min([t_z(-6 * w_0), t - t_z(3 * w_0)]), t, args=(x, y, z, t), epsrel=1e-2)[0]


def I(x, y, z, t):
    sigma_tilde_squared = (w(x) / (beta * C_LIGHT)) ** 2
    R_tilde_inverse_squared = k_l * (beta * C_LIGHT) ** 2 * R_inverse(x)
    t_tilde = t - z / (beta * C_LIGHT)
    a_squared = 1 / (1 / sigma_tilde_squared + 1j * R_tilde_inverse_squared)
    coefficient = 1 / 2 * np.sqrt(a_squared) * np.exp(1j * t_tilde * omega_l - a_squared * omega_l ** 2 / 4) * np.sqrt(
        pi)
    erf_term = erfc((t_tilde - t + 1 / 2 * 1j * a_squared * omega_l) / np.sqrt(a_squared))
    I_values = np.real(coefficient * erf_term)
    return I_values


def G_mine(x, y, z, t):
    I_values = I(x, y, z, t)
    y_envelope = w_0 / w(x) * np.exp(-y ** 2 / (w(x) ** 2))
    return beta * C_LIGHT / omega_l * I_values * y_envelope


# def phi(G_function, x, y, t, z_boundaries: Tuple[float, float], n=1000):
#     z = np.linspace(z_boundaries[0], z_boundaries[1], n)
#
#     return

def z_for_loop(x, y, z, t, func):
    results_array = np.zeros_like(z)
    if isinstance(t, float):
        for i in range(len(z)):
            results_array[i] = func(x, y, z[i], t)
    else:
        for i in range(len(z)):
            results_array[i] = func(x, y, z[i], t[i])
    return results_array


def phi_integrand(z, x, y, t):
    G_values = G_discrete_integral(x, y, z, t)
    A_values = A_z(x, y, z, t)
    return



# %%
# z_array = np.linspace(-5*w_0, 5*w_0, 100)

# N = 1000
# results_array = np.zeros((N, 5))
#
# for i in range(N):
#     t_0 = (np.random.random() - 1 / 2) * t_z(6 * w_0)
#     y_0 = (np.random.random() - 1 / 2) * 6 * w_0
#     x_0 = (np.random.random() - 1 / 2) * 6 * w_0
#     z_array = (np.random.random() - 1 / 2) * 6 * w_0
#
#     # G_discrete_integral_values = z_for_loop(x_0, y_0, z_array, t_0, G_discrete_integral)
#     G_discrete_integral_values = G_discrete_integral(x_0, y_0, z_array, t_0)
#     # plt.plot(z_array / w_0, G_discrete_integral_values, '.', label='discrete integral')
#
#     # G_integral_values = z_for_loop(x_0, y_0, z_array, t_0, G_integral)
#     # plt.plot(z_array / w_0, G_integral_values, label='integral')
#
#     # G_mine_values = z_for_loop(x_0, y_0, z_array, t_0, G_mine)
#     # plt.plot(z_array, G_mine_values, '--', label='mine')
#
#     # G_Osip_values = z_for_loop(x_0, y_0, z_array, t_0, G_Osip)
#     G_Osip_values = G_Osip(x_0, y_0, z_array, t_0)
#     # plt.plot(z_array / w_0, G_Osip_values, label='Osip')
#
#     # I_values = z_for_loop(x_0, y_0, z_array, t_0, I)
#     # plt.plot(z_array, I_values, label='I')
#
#     # I_numerical_values = z_for_loop(x_0, y_0, z_array, t_0, I_numerical)
#     # plt.plot(z_array, I_numerical_values, '--', label='I numerical')
#
#     rel_distance = np.abs(G_Osip_values - G_discrete_integral_values)
#     abs_size = np.abs(G_discrete_integral_values)
#
#     # plt.legend()
#     # plt.title(f't_0={t_0 / t_z(w_0):.2f}, x_0={x_0 / w_0:.2f}, y_0={y_0 / w_0:.2f}')
#     # plt.ylim(-5e-24, 5e-24)
#     # print(f't_0 = {t_0 / t_z(w_0):.2f}\nx_0 = {x_0 / w_0:.2f}\ny_0 = {y_0 / w_0:.2f}\n')
#     if not i % 100:
#         print(i, end='\r')
#     # plt.show()
#
#     results_array[i, :] = np.array([x_0, y_0, t_0, rel_distance, abs_size])
#
# # %%
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("t")
# ax.view_init(30, -30)
# ax.scatter(xs=results_array[:, 0] / w_0, ys=results_array[:, 1] / w_0, zs=results_array[:, 2] / t_z(w_0),
#            c=results_array[:, 3], cmap='gray')
#
# plt.show()
# # %%
# percentile_98_distance = np.percentile(results_array[:, 3], 98)
# percentile_98_value = np.percentile(results_array[:, 4], 98)
#
# plt.hist(results_array[:, 3], bins=100, label='Absolute distance')
# plt.title('Absolute distance')
# plt.axvline(percentile_98_distance, color='red', label=f'98% percentile = {percentile_98_distance:.2e}')
# plt.legend()
# plt.show()
#
# plt.hist(results_array[:, 4], bins=100, label='Absolute value')
# plt.axvline(percentile_98_value, color='red', label=f'98% percentile = {percentile_98_value:.1e}')
# plt.axvline(percentile_98_distance, color='green', label=f'98% percentile of distance = {percentile_98_distance:.1e}')
# plt.title('Absolute value')
# plt.legend()
# plt.show()
#
#
# # %%
# t_0 = 0.94
# x_0 = 0.14
# y_0 = -0.54
# z_0 = 0
#
# Z = np.linspace(z_0 - 5 * w_0, z_0, 100)
#
# A_z_integrand_values = A_z_integrand_variable_change(Z, x_0 * w_0, y_0 * w_0, z_0 * w_0, t_0 * t_z(w_0))
# plt.plot(Z, A_z_integrand_values)
#
# plt.show()
# %%
t_0 = 8
z_max = 3
x_0 = -1.1
y_0 = -1
z = np.linspace(-z_max, z_max, 1000)
t = (z_max-z) / (C_LIGHT * beta) + t_0
G_Osip_values = z_for_loop(x_0*w_0, y_0*w_0, z*w_0, t*w_0, G_Osip)
G_discrete_integral_values = z_for_loop(x_0*w_0, y_0*w_0, z*w_0, t*w_0, G_discrete_integral)
plt.plot(z, G_Osip_values, label='Osip')
plt.plot(z, G_discrete_integral_values, label='discrete integral')
plt.xlabel('z')
plt.ylabel('G')
plt.legend()
plt.show()



