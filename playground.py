# %%
import scipy
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# %%
def A(x, y):
    return 1

def V(x, y):
    return 1

def transfer_function(x_tag: float, y_tag: float, x: float, y: float, dz: float, k: float):
    dx = x - x_tag
    dy = y - y_tag
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    cos_theta = dz / r
    return np.exp(1j * k * r) * cos_theta / r

# %%

def integrand_1d(rho, dz, k):
    r = np.sqrt(rho**2 + dz**2)
    first_term = 2 * pi * dz * rho / r**2
    second_term = np.array([np.cos(k*r), np.sin(k*r)])
    return first_term * second_term

def integrand_2d(x_tag: float, y_tag: float, x: float, y: float, dz: float, k: float, part: str='real'):
    integrand_complex = A(x, y) * transfer_function(x, y, x_tag, y_tag, dz, k)
    if part == 'real':
        return np.real(integrand_complex)
    elif part == 'imag':
        return np.imag(integrand_complex)
    else:
        raise ValueError('part must be either "real" or "imag"')

def complex_integration_2d(x_tag: float, y_tag: float, x: float, y: float, dz: float, k: float, part: str='real' , boundary=1) -> complex:
    real_part = scipy.integrate.dblquad(integrand_2d, -boundary, boundary, lambda x: -np.sqrt(boundary**2-x**2), lambda x: np.sqrt(boundary**2-x**2), args=(0, 0, dz, k, 'real'))
    imag_part = scipy.integrate.dblquad(integrand_2d, -boundary, boundary, lambda x: -np.sqrt(boundary**2-x**2), lambda x: np.sqrt(boundary**2-x**2), args=(0, 0, dz, k, 'imag'))
    return real_part[0] + 1j * imag_part[0]

# %%
x_min=-10
x_max=10
y_min=-10
y_max=10

k=1e9
dz=(2*pi)/k * 50
A = 10

As = np.logspace(-1, 3.2, 40)
N = len(As)
Is = np.zeros((N, 2), dtype=complex)
# %%
for i in range(N):
    rho = As[i] * dz
    integral_1d = scipy.integrate.quad_vec(integrand_1d, 1e-15, rho, args=(dz, k),epsabs=1e-11)    
    Is[i, :] = integral_1d[0]

# %%
plt.plot(As, Is)
plt.xscale('log')

# %%
# I_real = scipy.integrate.dblquad(integrand, x_min, x_max, lambda x: y_min, lambda x: y_max, args=(0, 0, dz, k, 'real'))
# I_imag = scipy.integrate.dblquad(integrand, x_min, x_max, lambda x: y_min, lambda x: y_max, args=(0, 0, dz, k, 'imag'))


boundaries = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
N = len(boundaries)
Is = np.zeros(N, dtype=complex)

for i in range(N):
    rho = boundaries[i]
    I_real = scipy.integrate.dblquad(integrand_2d, -rho, rho, lambda x: -np.sqrt(rho**2-x**2), lambda x: np.sqrt(rho**2-x**2), args=(0, 0, dz, k, 'real'), epsabs=1e-19)
    # I_imag = scipy.integrate.dblquad(integrand_imag, -i, i, lambda x: -np.sqrt(i**2-x**2), lambda x: np.sqrt(i**2-x**2), args=(0, 0, dz, k))
    Is[i] = I_real[0]# + 1j * I_imag[0]
    print(boundaries[i], end='\r')


# %%
plt.plot(boundaries, np.real(Is))
plt.xscale('log')

# %% Fatal error in launcher: Unable to create process using install scipy': The system cannot find the file specified
