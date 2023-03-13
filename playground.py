from microscope import *
from scipy import integrate

N_POINTS = 256  # Resolution of image
l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
theta_polarization = 1/2 * np.pi
E_0 = Joules_of_keV(300)
n_z = 15000
ring_cavity = True
alpha_cavity = 26*2*pi / 360

x, y, t = np.array([0], dtype=np.float64), np.array([0], dtype=np.float64), np.linspace(0, 1e-14, 50)[18]

coordinate_system = CoordinateSystem(axes=(x, y))
input_wave = WaveFunction(np.array([[1]]), coordinate_system, E_0)
power_1 = 9.56e16

C = CavityNumericalPropagator(l_1=l_1, l_2=l_2, power_1=power_1, power_2=-1, NA_1=NA_1, NA_2=-1,
                              theta_polarization=theta_polarization, alpha_cavity=alpha_cavity, ring_cavity=ring_cavity,
                              ignore_past_files=True, n_z=n_z, debug_mode=True, t=np.array([t]))

# C_a = CavityAnalyticalPropagator(l_1=l_1, power_1=C.power_1, power_2=None, theta_polarization=theta_polarization,
#                                  NA_1=NA_1,ring_cavity=C.ring_cavity)

Z, X, Y, T = C.generate_coordinates_lattice(x, y, t, input_wave.beta)

A = C.rotated_gaussian_beam_A(Z, X, Y, T,
                                    beta_electron=input_wave.beta, save_to_file=False)
# plt.imshow(A[:, 0, 0, :], aspect=A.shape[-1] / A.shape[0])
# plt.ylabel("z")
# plt.xlabel("t")
# plt.show()
G = C.G_gauge(C.potential_envelope2vector_components(A, 'z', input_wave.beta), Z)

grad_G = C.grad_G(Z, X, Y, T,
                        beta_electron=input_wave.beta,
                        A_z=C.potential_envelope2vector_components(A, "z", input_wave.beta),
                        save_to_file=False,
                        )

integrand = (
            (1 - input_wave.beta**2)
            * safe_abs_square(C.potential_envelope2vector_components(A, "z", input_wave.beta) - grad_G[:, :, :, :, 0])
            + safe_abs_square(C.potential_envelope2vector_components(A, "x", input_wave.beta) - grad_G[:, :, :, :, 1])
            + safe_abs_square(C.potential_envelope2vector_components(A, "y", input_wave.beta) - grad_G[:, :, :, :, 2])
        )

prefactor = -(
                E_CHARGE ** 2 / (2 * M_ELECTRON * gamma_of_beta(input_wave.beta) * input_wave.beta *
                                              C_LIGHT * H_BAR)
        )
#
phi = prefactor * np.trapz(integrand, x=Z, axis=0)[0, 0, :]
print(f"{phi=}")
phi_direct = C.phi(input_wave=input_wave)[0, 0, 0]
print(f"{phi_direct=}")
# plt.plot(phi)
# plt.show()
