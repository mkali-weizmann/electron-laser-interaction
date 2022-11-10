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
    plt.show()

l=532e-9
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
    return x / (x**2 + x_R**2)

def psi(x):
    return np.arctan(x / x_R)

def x_grating(x, y, z):
    return np.cos(k_l * (x+(y**2 + z**2) * R_inverse(x)) + psi(x))

def envelope(x, y, z):
    return w_0 / w(x) * np.exp(
                np.clip(-(y**2 + z**2) / (w(x)**2), a_min=-500, a_max=None))

def temporal_component(t):
    return np.cos(omega_l * t)

def A_z(x, y, z, t):
    return envelope(x, y, z) * x_grating(x, y, z) * temporal_component(t)

def A_z_integrand(s, x, y, z, t):
    return A_z(x, y, z - beta * C_LIGHT * (t-s), s)

def G_integral(x, y, z, t):
    return C_LIGHT * beta / omega_l * \
           integrate.quad(A_z_integrand, min([t_z(-6*w_0), t-t_z(3*w_0)]), t, args=(x, y, z, t), epsabs=1e-20)[0]

def s(t, n=1000):
    return np.linspace(min([t_z(-6*w_0), t-t_z(3*w_0)]), t, n)

def G_discrete_integral(x, y, z, t):
    s_values = s(t)
    A_z_integrand_values = A_z_integrand(s_values, x, y, z, t)
    return C_LIGHT * beta / omega_l * np.trapz(A_z_integrand_values, s_values)

def G_Osip(x, y, z, t):
    return C_LIGHT * beta / omega_l**2 * envelope(x, y, z) * x_grating(x, y, z) * np.sin(omega_l * t)

def I_integrand(s, x, y, z, t):
    sigma_tilde_squared = (w(x) / (beta * C_LIGHT)) ** 2
    R_tilde_inverse_squared = k_l * (beta * C_LIGHT) ** 2 * R_inverse(x)
    t_tilde = t - z / (beta * C_LIGHT)
    a_squared = 1 / (1 / sigma_tilde_squared + 1j * R_tilde_inverse_squared)
    return np.real(np.exp(-(s - t_tilde)**2 / a_squared + 1j * omega_l * s))

def I_numerical(x, y, z, t):
    return integrate.quad(I_integrand, min([t_z(-6*w_0), t-t_z(3*w_0)]), t, args=(x, y, z, t), epsabs=1e-20)[0]

def I(x, y, z, t):
    sigma_tilde_squared = (w(x) / (beta * C_LIGHT)) ** 2
    R_tilde_inverse_squared = k_l * (beta * C_LIGHT) ** 2 * R_inverse(x)
    t_tilde = t - z / (beta * C_LIGHT)
    a_squared = 1/(1 / sigma_tilde_squared + 1j * R_tilde_inverse_squared)
    coefficient = 1/2 * np.sqrt(a_squared) * np.exp(1j*t_tilde*omega_l - a_squared*omega_l**2 / 4) * np.sqrt(pi)
    erf_term = erfc((t_tilde - t + 1/2 * 1j * a_squared * omega_l) / np.sqrt(a_squared))
    I_values = np.real(coefficient * erf_term)
    return I_values
    # coefficient = 1 / 2 * np.sqrt(pi) * np.sqrt(1/a_inverse_squared) * np.exp(-1 / 4 * 1 / a_inverse_squared * omega_l)
    # complex_exponent_term = np.exp(1j * t_tilde * omega_l)
    # erf_term = 1 + erf((-z / (beta * C_LIGHT) - 1j / 2 * 1 / a_inverse_squared * omega_l) * np.sqrt(a_inverse_squared))
    # I_values = np.real(coefficient * complex_exponent_term * erf_term)
    # return I_values

def G_mine(x, y, z, t):
    I_values = I(x, y, z, t)
    y_envelope = w_0 / w(x) * np.exp(-y**2 / (w(x)**2))
    return beta * C_LIGHT / omega_l * I_values * y_envelope

def z_for_loop(x, y, z, t, func):
    results_array = np.zeros_like(z)
    for i in range(len(z)):
        results_array[i] = func(x, y, z[i], t)
    return results_array


z_array = np.linspace(-3*w_0, 3*w_0, 80)
t_0 = t_z(4*w_0)
y_0 = -w_0
x_0 = w_0
z_0 = 2*w_0


G_discrete_integral_values = z_for_loop(x_0, y_0, z_array, t_0, G_discrete_integral)
plt.plot(z_array, G_discrete_integral_values, '.', label='discrete integral')

# G_integral_values = z_for_loop(x_0, y_0, z_array, t_0, G_integral)
# plt.plot(z_array, G_integral_values, label='integral')

G_mine_values = z_for_loop(x_0, y_0, z_array, t_0, G_mine)
plt.plot(z_array, G_mine_values, '--', label='mine')

G_Osip_values = z_for_loop(x_0, y_0, z_array, t_0, G_Osip)
plt.plot(z_array, G_Osip_values, label='Osip')

I_values = z_for_loop(x_0, y_0, z_array, t_0, I)
# plt.plot(z_array, I_values, label='I')

I_numerical_values = z_for_loop(x_0, y_0, z_array, t_0, I_numerical)
# plt.plot(z_array, I_numerical_values, '--', label='I numerical')

plt.legend()
plt.show()

