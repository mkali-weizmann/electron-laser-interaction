# %%
import numpy as np
from numpy import pi
from scipy.integrate import quad_vec
from scipy.special import jv
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Type, Tuple
from warnings import warn
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate

M_ELECTRON = 9.1093837e-31
C_LIGHT = 299792458
H_BAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
FINE_STRUCTURE_CONST = 7.299e-3

np.seterr(all='raise')


# %%
# def k2E(k: float) -> float:
#     return np.sqrt(M_ELECTRON**2 * C_LIGHT**4 + H_BAR**2 * k**2*C_LIGHT**2)


def E2l(E: float) -> float:
    return 2 * pi / E2k(E)


def E2V(E: float) -> float:
    return E / E_CHARGE


def V2E(V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return V * E_CHARGE


def E2k(E):
    return E / (C_LIGHT * H_BAR) * np.sqrt(1 + 2 * M_ELECTRON * C_LIGHT ** 2 / E)


def k2beta(k: float) -> float:
    return p2beta(k * H_BAR)


def beta2k(beta: float) -> float:
    return beta2p(beta) / H_BAR


def E2beta(E: float) -> float:
    return k2beta(E2k(E))


def V2k(V0: float, V: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
    if V is None:
        V = 0
    E = V2E(V0 + V)
    return E2k(E)


def k2l(k: float) -> float:
    return 2 * pi / k


def l2k(l: float) -> float:
    return 2 * pi / l


def k2w(k: float, n=1) -> float:
    return C_LIGHT * k / n


def w2k(w: float, n=1) -> float:
    return n * w / C_LIGHT


def l2w(l: float) -> float:
    return k2w(k2l(l))


def w2l(w: float) -> float:
    return k2l(w2k(w))


def p2beta(p: float) -> float:
    return p / (np.sqrt(C_LIGHT ** 2 * M_ELECTRON ** 2 + p ** 2))


def beta2p(beta: float) -> float:
    return M_ELECTRON * C_LIGHT * beta * np.sqrt(1 - beta ** 2)


def beta2gamma(beta: float) -> float:
    return 1 / np.sqrt(1 - beta ** 2)


def gamma2beta(gamma: float) -> float:
    return np.sqrt(1 - 1 / gamma ** 2)


def ls2beta(l1: float, l2: float):
    return (1 - l1 / l2) / (1 + l1 / l2)


def KeV2Joules(keV: float) -> float:
    return keV * 1.602176634e-16


def Joules2keV(J: float) -> float:
    return J / 1.602176634e-16


def x_R_gaussian(w0: float, l: float) -> float:
    # l is the wavelength of the laser
    return pi * w0 ** 2 / l


def NA2w0(NA: float, l: float) -> float:
    return l / (np.pi * NA)


def w02NA(w0: float, l: float) -> float:
    return l / (np.pi * w0)


def w_x_gaussian(w_0: float, x: Union[float, np.ndarray], x_R: Optional[float] = None, l_laser: Optional[float] = None):
    # l is the wavelength of the laser
    if x_R is None:
        x_R = x_R_gaussian(w_0, l_laser)
    return w_0 * np.sqrt(1 + (x / x_R) ** 2)


def gouy_phase_gaussian(x: Union[float, np.ndarray], x_R: Optional[float] = None,
                        w_0: Optional[float] = None, l_laser: Optional[float] = None):
    # l is the wavelength of the laser
    if x_R is None:
        x_R = x_R_gaussian(w_0, l_laser)
    return np.arctan(x / x_R)


def gaussian_beam(x: [float, np.ndarray], y: [float, np.ndarray], z: Union[float, np.ndarray],
                  A, lamda: float, w_0: Optional[float] = None, NA: Optional[float] = None,
                  t: Optional[float, np.ndarray] = None,
                  mode: str = "intensity", is_cavity: bool = True) -> Union[np.ndarray, float]:
    if w_0 is None:
        w_0 = NA2w0(NA, lamda)
    # Calculates the electric field of a gaussian beam
    x_R = x_R_gaussian(w_0, lamda)
    w_x = w_x_gaussian(w_0=w_0, x=x, x_R=x_R)
    gouy_phase = gouy_phase_gaussian(x, x_R)
    k = l2k(lamda)
    if isinstance(x, float) and x == 0:
        R_x = 1e10
    elif isinstance(x, np.ndarray):
        x_safe = np.where(x == 0, 1e-10, x)  # ARBITRARY
        R_x = x_safe * (1 + (x_R / x_safe) ** 2)
    else:
        R_x = x * (1 + (x_R ** 2 / (x ** 2 + 1e-20)))
    other_phase = k * x + k * (z ** 2 + y ** 2) / (2 * R_x)
    envelope = A * (w_0 / w_x) * np.exp(-(z ** 2 + y ** 2) / w_x ** 2)

    if mode == "intensity":
        if is_cavity:
            return envelope ** 2 * 4 * np.cos(other_phase + gouy_phase) ** 2
        else:
            return envelope ** 2

    elif mode == "field":
        total_phase = other_phase - gouy_phase
        if t is not None:
            time_phase = np.exp(1j * ((C_LIGHT / k) * t))
        else:
            time_phase = 1
        if is_cavity:
            return envelope * np.cos(total_phase) * time_phase
        else:
            return envelope * np.exp(1j * total_phase) * time_phase

    elif mode == "phase":
        total_phase = other_phase - gouy_phase
        if t is not None:
            time_phase = (C_LIGHT / k) * t
        else:
            time_phase = 0
        return total_phase + time_phase


def rotated_gaussian_beam_Az(x: [float, np.ndarray], y: [float, np.ndarray], z: Union[float, np.ndarray],
                             A, lamda: float, alpha_cavity: float, theta_polarization: float,
                             w_0: Optional[float] = None, NA: Optional[float] = None,
                             t: Optional[float, np.ndarray] = None, mode: str = "intensity"
                             ) -> Union[np.ndarray, float]:
    return gaussian_beam(x * np.cos(alpha_cavity) - z * np.sin(alpha_cavity), y,
                         x * np.sin(alpha_cavity) + z * np.cos(alpha_cavity), A=A, lamda=lamda, w_0=w_0, t=t, mode=mode
                         ) * np.cos(theta_polarization)


def A_integrand(Z: np.ndarray,
                x: np.ndarray,
                y: np.ndarray,
                z: np.ndarray,
                t: np.ndarray,
                beta_electron: float,
                NA: float,
                lamda_1: float,
                lamda_2: float,
                alpha_cavity: float,
                theta_polarization: float = 0,
                ) -> float:
    T = (Z - z) / (beta_electron * C_LIGHT) + t
    A1 = rotated_gaussian_beam_Az(x=x, y=y, z=Z, A=1, lamda=lamda_1, alpha_cavity=alpha_cavity,
                                  theta_polarization=theta_polarization, NA=NA, t=T, mode="field")
    A2 = rotated_gaussian_beam_Az(x=x, y=y, z=Z, A=1, lamda=lamda_2, alpha_cavity=alpha_cavity,
                                  theta_polarization=theta_polarization, NA=NA, t=T, mode="field")
    jacobian = 1 / (beta_electron * C_LIGHT)
    return (A1 + A2) * jacobian


def G_gause(x: np.ndarray,
            y: np.ndarray,
            z: np.float,
            t: np.ndarray,
            beta_electron: float,
            NA: float,
            lamda_1: float,
            lamda_2: float,
            alpha_cavity: float,
            theta_polarization: float = 0
            ) -> np.ndarray:
    integral = integrate.quad_vec(A_integrand,
                                  - 20 * NA2w0(NA, max(lamda_1, lamda_2)),  # ARBITRARY
                                  z,
                                  args=(
                                  x, y, z, t, beta_electron, NA, lamda_1, lamda_2, alpha_cavity, theta_polarization))
    return integral[0]


def average_intensity(A: float, w0: float, k_laser: float,
                      X: [float, np.ndarray], Y: [float, np.ndarray], mode="intensity", is_cavity=True) -> np.ndarray:
    # Calculates the average intensity of a function over a 2D plane
    def integrand(z_tag):
        return gaussian_beam(X, Y, z_tag, A, w0, k_laser, mode=mode, is_cavity=is_cavity)

    integral = quad_vec(integrand, -10 * w0, 10 * w0)  # ARBITRARY - CHANGE THAT WHEN POSSIBLE
    return integral[0] / (20 * w0)


def ASPW_propagation(U_z0: np.ndarray, dxdydz: tuple[float, ...], k: float) -> np.ndarray:
    dx, dy, dz = dxdydz
    U_fft = np.fft.fftn(U_z0)
    k_x = np.fft.fftfreq(U_z0.shape[0], d=dx) * 2 * pi
    k_y = np.fft.fftfreq(U_z0.shape[1], d=dy) * 2 * pi
    k_x, k_y = np.meshgrid(k_x, k_y)
    phase_argument = np.exp(1j * np.sqrt(k ** 2 - k_x ** 2 - k_y ** 2) * dz)
    U_fft *= phase_argument
    U_dz = np.fft.ifftn(U_fft)
    return U_dz


def propagate_through_potential_slice(incoming_wave: np.ndarray, averaged_potential: np.ndarray,
                                      dz: float, E0) -> np.ndarray:
    # CHECK THAT THIS my original version CORRESPONDS TO KIRKLAND'S VERSION
    # V0 = E0 / E_CHARGE
    # dk = V2k(V0) - V2k(V0, averaged_potential)
    # psi_output = incoming_wave * np.exp(-1j * dk * dz)
    # KIRKLAND'S VERSION: (TWO VERSION AGREE!)
    sigma = beta2gamma(E2beta(E0)) * M_ELECTRON * E_CHARGE / (H_BAR ** 2 * E2k(E0))
    psi_output = incoming_wave * np.exp(1j * sigma * dz * averaged_potential)
    return psi_output


############################################################################################################
@dataclass()
class CoordinateSystem:
    def __init__(self,
                 axes: Optional[Tuple[np.ndarray, ...]] = None,  # The axes of the incoming wave function
                 lengths: Optional[Tuple[float, ...]] = None,  # the lengths of the sample in the x, y directions
                 n_points: Optional[Tuple[int, ...]] = None,  # the number of points in the x, y directions
                 dxdydz: Optional[Tuple[float, ...]] = None):  # the step size in the x, y directions

        # Generates a coordinate system for a given number of points and lengths

        if axes is not None:
            self.axes: Tuple[np.ndarray, ...] = axes
        elif lengths is not None:
            dim = len(lengths)
            if n_points is not None and dxdydz is not None:
                raise (ValueError("You can and only specify one out of n and dxdydz"))
            elif n_points is not None:
                self.axes: Tuple[np.ndarray, ...] = tuple(
                    np.linspace(-lengths[i] / 2, lengths[i] / 2, n_points[i]) for i in range(dim))
            elif dxdydz is not None:
                self.axes: Tuple[np.ndarray, ...] = tuple(
                    np.arange(-lengths[i] / 2, lengths[i] / 2, dxdydz[i]) for i in range(dim))
            else:
                raise (ValueError("You must specify either n or dxdydz"))
        else:
            raise (ValueError("Either lengths or axes must be not None"))

    @property
    def dim(self) -> int:
        return len(self.axes)

    @property
    def lengths(self) -> Tuple[float, ...]:
        return tuple(axis[-1] - axis[0] for axis in self.axes)

    @property
    def n_points(self) -> Tuple[int, ...]:
        return tuple(len(axis) for axis in self.axes)

    @property
    def dxdydz(self) -> Tuple[float, ...]:
        return tuple(axis[1] - axis[0] for axis in self.axes)

    @property
    def grids(self) -> Tuple[np.ndarray, ...]:
        return tuple(np.meshgrid(*self.axes))

    @property
    def X_grid(self) -> np.ndarray:
        return self.grids[0]

    @property
    def Y_grid(self) -> np.ndarray:
        return self.grids[1]

    @property
    def Z_grid(self) -> np.ndarray:
        if self.dim == 3:
            return self.grids[2]
        else:
            raise (ValueError("This coordinate system is not 3D"))

    @property
    def x_axis(self) -> np.ndarray:
        return self.axes[0]

    @property
    def y_axis(self) -> np.ndarray:
        return self.axes[1]

    @property
    def z_axis(self) -> np.ndarray:
        if self.dim == 3:
            return self.axes[2]
        else:
            raise (ValueError("This coordinate system is not 3D"))

    @property
    def dx(self) -> float:
        return self.dxdydz[0]

    @property
    def dy(self) -> float:
        return self.dxdydz[1]

    @property
    def dz(self) -> float:
        if self.dim == 3:
            return self.dxdydz[2]
        else:
            raise (ValueError("This coordinate system is not 3D"))

    @property
    def limits(self) -> Tuple[float, ...]:
        limits_pairs = tuple((axis[0], axis[-1]) for axis in self.axes)
        limits = tuple(lim for t in limits_pairs for lim in t)
        return limits


class WaveFunction:
    def __init__(self,
                 psi: np.ndarray,  # The input wave function in one z=const plane
                 coordinates: CoordinateSystem,  # The coordinate system of the input wave
                 E0: float,  # Energy of the particle
                 ):
        self.psi: np.ndarray = psi
        self.coordinates = coordinates
        self.E0: float = E0


class Propagator:
    def propagate(self, state: WaveFunction) -> WaveFunction:
        raise NotImplementedError()


@dataclass
class PropagationStep:
    input_wave: WaveFunction
    output_wave: WaveFunction
    propagator: Propagator


class Microscope:
    def __init__(self, propagators: List[Propagator], print_progress: bool = False):
        self.propagators = propagators
        self.propagation_steps: List[PropagationStep] = []
        self.print_progress = print_progress

    def take_a_picture(self, input_wave: WaveFunction) -> WaveFunction:
        for propagator in self.propagators:
            if self.print_progress:
                print(f"Propagating with {propagator}")
            output_wave = propagator.propagate(input_wave)
            self.propagation_steps.append(PropagationStep(input_wave, output_wave, propagator))
            input_wave = output_wave
        return input_wave

    def plot_step(self, step: int, clip=True, title=None, file_name=None):
        step = self.propagation_steps[step]
        psi_input = step.input_wave.psi
        psi_output = step.output_wave.psi
        extent_input = step.input_wave.coordinates.limits
        extent_output = step.output_wave.coordinates.limits
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(221)
        input_abs = np.abs(psi_input) ** 2
        input_arg = np.angle(psi_input)
        output_abs = np.abs(psi_output) ** 2
        output_arg = np.angle(psi_output)

        if clip:
            input_abs = np.clip(input_abs, 0, np.percentile(input_abs, 99))
            input_arg = np.clip(input_arg, 0, np.percentile(input_arg, 99))
            output_abs = np.clip(output_abs, 0, np.percentile(output_abs, 99))
            output_arg = np.clip(output_arg, 0, np.percentile(output_arg, 99))
        im1 = ax1.imshow(input_abs, extent=extent_input)
        ax1.set_title(r"$\left|\psi_{i}\right|^{2}$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(input_arg, extent=extent_input)
        ax2.set_title(r"$arg\left(\psi_{i}\right)$")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        ax3 = fig.add_subplot(223)
        im3 = ax3.imshow(output_abs, extent=extent_output)
        ax3.set_title(r"$\left|\psi_{o}\right|^{2}$")
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')
        ax4 = fig.add_subplot(224)
        im4 = ax4.imshow(output_arg, extent=extent_output)
        ax4.set_title(r"$arg\left(\psi_{o}\right)$")
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im4, cax=cax, orientation='vertical')
        if title is not None:
            fig.suptitle(title)

        if file_name is not None:
            fig.savefig("Figures\\" + file_name)
        plt.show()


class CavityDoubleFrequencyPropagator(Propagator):
    def __init__(self,
                 l_1: float = 1064 * 1e-9,
                 l_2: Optional[float] = 532 * 1e-9,
                 A_1: float = 1,
                 A_2: Optional[float] = -1,
                 Na: float = 0.2,
                 alpha_lattice: Optional[float] = None,  # tilt angle of the lattice (of the cavity)
                 theta_polarization: Optional[float] = None):

        self.l_1: float = l_1  # Laser's frequency
        self.A_1: float = A_1  # Laser's amplitude
        if A_2 == -1:  # -1 means that the second laser is defined by the condition for equal amplitudes in the
            # lattices' frame
            self.A_2: float = A_1 * (l_1 / l_2)
        else:
            self.A_2: float = A_2
        self.l_2 = l_2
        self.Na: float = Na  # Cavity's numerical aperture
        self.alpha_lattice: float = alpha_lattice  # cavity_2f's angle with respect to microscope's x-axis. positive
        # # number means the part of the cavity_2f in the positive x direction is tilted downwards toward the z axis.
        # in the cavity_2f.
        self.theta_polarization: Optional[float] = theta_polarization  # polarization angle of the laser

    def phi_0(self, x: np.ndarray, y: np.ndarray, beta_electron: float) -> np.ndarray:
        # Gives the phase acquired by a narrow electron beam centered around (x, y) by passing in the cavity_2f.
        # Does not include the relativistic correction.
        # According to equation e_10 and equation gaussian_beam_potential_total_phase in my lyx file
        if self.alpha_lattice is None:
            alpha_lattice = np.arcsin(self.beta_lattice / beta_electron)
        else:
            alpha_lattice = self.alpha_lattice
            warn("alpha_lattice is not None. Using the value given by the user, Note that the calculations assume"
                 "that the lattice satisfy cos(alpha_lattice) = beta_lattice / beta_electron")
        x_lattice = x / np.cos(alpha_lattice)
        x_R = x_R_gaussian(self.w_0, self.l)
        w_x = w_x_gaussian(w_0=self.w_0, x=x_lattice, x_R=x_R)
        # The next two lines are based on equation e_11 in my lyx file
        constant_coefficients = (E_CHARGE ** 2 * self.w_0 ** 2 * np.sqrt(pi) * self.A ** 2) / \
                                (H_BAR * 4 * np.sqrt(2) * M_ELECTRON * C_LIGHT * beta_electron * beta2gamma(
                                    beta_electron))
        spatial_envelope = np.exp(  # Shouldn't be here a w_0 term? No! it is in the previous term.
            np.clip(-2 * y ** 2 / w_x ** 2, a_min=-500, a_max=None)) / w_x  # ARBITRARY CLIPPING FOR STABILITY
        if self.A_2 is None:  # For the case of a single laser, add the spatial cosine
            gouy_phase = gouy_phase_gaussian(x_lattice, x_R, self.w_0)
            cosine_squared = 4 * np.cos(4 * np.pi * x_lattice / self.l + gouy_phase) ** 2
            spatial_envelope *= cosine_squared
        return constant_coefficients * spatial_envelope

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        phi_0 = self.phi_0(input_wave.coordinates.X_grid, input_wave.coordinates.Y_grid, E2beta(input_wave.E0))
        phase_shift = self.phase_shift(phi_0)
        if self.A_2 is not None:  # For the case of a double laser:
            attenuation_factor = self.attenuation_factor(phi_0)
        else:
            attenuation_factor = 1
        output_wave = input_wave.psi * np.exp(-1j * phase_shift) * attenuation_factor  # ARBITRARY ARBITRARY - I FIXED
        # A SIGN MISTAKE BY ADDING A SIGN
        return WaveFunction(output_wave, input_wave.coordinates, input_wave.E0)

    def phase_shift(self, phi_0: Optional[np.ndarray] = None, X_grid: Optional[np.ndarray] = None,
                    Y_grid: Optional[np.ndarray] = None, beta_electron: Optional[float] = None):
        if phi_0 is None:
            phi_0 = self.phi_0(X_grid, Y_grid, beta_electron)
        if self.A_2 is not None:  # For the case of a double laser:
            return phi_0 * (2 + (self.Gamma_plus / self.Gamma_minus) ** 2 +
                            (self.Gamma_minus / self.Gamma_plus) ** 2)
        else:
            return phi_0

    def attenuation_factor(self, phi_0: Optional[np.ndarray] = None, X_grid: Optional[np.ndarray] = None,
                           Y_grid: Optional[np.ndarray] = None, beta_electron: Optional[float] = None):
        if phi_0 is None:
            phi_0 = self.phi_0(X_grid, Y_grid, beta_electron)
        return jv(0, 2 * phi_0 * self.rho(beta_electron)) ** 2

    def rho(self, beta_electron: float):
        return 1 - 2 * beta_electron ** 2 * np.cos(self.theta_polarization) ** 2

    @property
    def w_0(self) -> float:
        # The width of the gaussian beam of the first laser
        # based on https://en.wikipedia.org/wiki/Gaussian_beam
        return self.l / (pi * np.arcsin(self.Na))

    @property
    def beta_lattice(self) -> float:
        if self.A_2 is not None:
            return (self.l_1 - self.l_2) / (self.l_1 + self.l_2)
        else:
            return 0

    @property
    def Gamma_plus(self) -> float:
        return np.sqrt((1 + self.beta_lattice) / (1 - self.beta_lattice))

    @property
    def Gamma_minus(self) -> float:
        return np.sqrt((1 - self.beta_lattice) / (1 + self.beta_lattice))

    @property
    def A(self) -> float:
        # The effective amplitude of the lattice in the moving frame. in case of a single frequency that is just the
        # amplitude of the first laser.
        if self.A_2 is None:
            return self.A_1
        else:
            return self.A_1 * self.Gamma_plus

    @property
    def l(self) -> float:
        # The effective wavelength of the lattice in the moving frame. in case of a single frequency that is just
        # the wavelength of the first laser.
        if self.l_2 is None:
            return self.l_1
        else:
            return self.l_1 / self.Gamma_plus

    @property
    def A_plus(self) -> float:
        return self.A_2 * self.Gamma_plus

    @property
    def A_minus(self) -> float:
        return self.A_1 * self.Gamma_minus

    @property
    def k(self) -> float:
        return l2k(self.l)


# class CavityPropagator(Propagator):
#     def __init__(self,
#                  l: float = 532 * 1e-9,
#                  A: float = 1,
#                  Na: float = 0.2,
#                  alpha_lattice: float = 0,
#                  theta_polarization: float = 0,
#                  relativistic_interaction: bool = True):  # Calculate it manually later):
#         self.l: float = l  # Laser's frequency
#         self.A: float = A  # Laser's amplitude
#         self.Na: float = Na  # Cavity's numerical aperture
#         self.alpha_lattice: float = alpha_lattice  # cavity_2f's angle with respect to microscope's x-axis. positive
#         # # number means the part of the cavity_2f in the positive x direction is tilted downwards toward the z axis.
#         # in the cavity_2f.
#         self.theta_polarization: float = theta_polarization  # polarization angle of the laser
#         self.relativistic_interaction: bool = relativistic_interaction
#
#     def propagate(self, input_wave: WaveFunction) -> WaveFunction:
#         potential = self.generate_ponderomotive_potential(input_wave.coordinates.grids,
#                                                           beta_electron=E2beta(input_wave.E0))
#         output_wave = propagate_through_potential_slice(input_wave.psi, potential, dz=self.w0 * 3,  # ARBITRARY,
#                                                         E0=input_wave.E0)  # Change dz
#         amplitude_reduction_factor = jv(0, potential)
#         output_wave *= amplitude_reduction_factor
#         return WaveFunction(output_wave, input_wave.coordinates, input_wave.E0)
#
#     @property
#     def w0(self) -> float:
#         # The width of the gaussian beam of the first laser
#         return self.l / (pi * np.arcsin(self.Na))
#
#     def w_x(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         # The width of the gaussian beam
#         return self.w0 * np.sqrt(1 + (x / self.w0) ** 2)
#
#     def generate_ponderomotive_potential(self, grids, beta_electron: Optional[float] = None):
#         X, Y = grids[0], grids[1]
#         if beta_electron is None and self.theta_polarization is None:  # For the non-relativistic case
#             coefficient = -E_CHARGE ** 2 / (4 * M_ELECTRON * l2w(self.l) ** 2)
#             potential = coefficient * average_intensity(A=self.A, w0=self.w0, k_laser=l2k(self.l), X=X, Y=Y,
#                                                         mode='intensity', is_cavity=True)
#         elif beta_electron is not None and self.theta_polarization is not None:  # For the relativistic case
#             rho = 1 - 2 * beta_electron ** 2 * (np.cos(self.theta_polarization)) ** 2
#             # envelope_squared_summed = average_intensity(A=self.A, w_0=self.w_0, k_laser=l2k(self.l), X=X, Y=Y,
#             #                                             mode='intensity', is_cavity=False)
#             envelope_squared_summed = E_CHARGE ** 2 * np.sqrt(pi) * self.A ** 2 * self.w0 ** 2 / (
#                     4 * M_ELECTRON * C_LIGHT * beta_electron * beta2gamma(beta_electron) * np.sqrt(2))
#             spatial_cosine = np.cos(2 * l2k(self.l) * X)
#             constants = E_CHARGE ** 2 / (4 * M_ELECTRON * beta2gamma(beta_electron))
#             potential = (1 / 2) * constants * envelope_squared_summed * (1 + rho * spatial_cosine)
#         else:
#             raise ValueError("input both beta_lattice and gamma_lattice for the relativistic case or neither for"
#                              "the non-relativistic case")
#         return potential


class SamplePropagator(Propagator):
    def __init__(self,
                 coordinates: Optional[CoordinateSystem] = None,
                 axes: Optional[Tuple[np.ndarray, ...]] = None,
                 potential: Optional[np.ndarray] = None,
                 path_to_potential_file: Optional[str] = None,
                 dummy_potential: Optional[str] = None,
                 ):

        if coordinates is not None:
            self.coordinates = coordinates
        elif axes is not None:
            self.coordinates = CoordinateSystem(axes=axes)
        if potential is not None:
            self.potential = potential
        elif path_to_potential_file is not None:
            self.potential = np.load(path_to_potential_file)
        elif dummy_potential is not None:
            self.generate_dummy_potential(dummy_potential)
        else:
            raise ValueError("You must specify either a potential or a path to a potential file or a dummy potential")

    def generate_dummy_potential(self, potential_type: str = 'one gaussian'):
        X, Y, Z = self.coordinates.grids
        lengths = self.coordinates.lengths
        if potential_type == 'one gaussian':
            potential = 100 * np.exp(-(
                    X ** 2 / (2 * (lengths[0] / 3) ** 2) + Y ** 2 / (2 * (lengths[1] / 3) ** 2) + Z ** 2 / (
                    2 * (lengths[1] / 3) ** 2)))
        elif potential_type == 'two gaussians':
            potential = 100 * np.exp(-((X - lengths[0] / 4) ** 2 / (2 * (lengths[0] / 6) ** 2) +
                                       Y ** 2 / (2 * (lengths[1] / 4) ** 2) +
                                       Z ** 2 / (2 * (lengths[0] / 8) ** 2))) + \
                        100 * np.exp(-((X + lengths[0] / 4) ** 2 / (2 * (lengths[0] / 6) ** 2) + Y ** 2 / (
                    2 * (lengths[1] / 4) ** 2) + Z ** 2 / (2 * (lengths[0] / 8) ** 2)))
        elif potential_type == 'a letter':
            potential_2d = np.load("example_letter.npy")
            potential = np.tile(potential_2d[:, :, np.newaxis], (1, 1, self.coordinates.grids[0].shape[2]))

        elif potential_type == 'letters':
            potential_2d = np.load("letters_big_1024.npy")
            potential = np.tile(potential_2d[:, :, np.newaxis], (1, 1, self.coordinates.grids[0].shape[2]))
        else:
            raise NotImplementedError("This potential type is not implemented, enter 'one gaussian' or "
                                      "'two gaussians' or 'a letter'")
        self.potential = potential

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        if self.potential is None:
            warn("No potential is defined, generating a dummy potential of two gaussians")
            self.generate_dummy_potential('two gaussians')

        output_wave = input_wave.psi.copy()
        for i in range(self.potential.shape[2]):
            output_wave = ASPW_propagation(output_wave, self.coordinates.dxdydz, E2k(input_wave.E0))
            output_wave = propagate_through_potential_slice(output_wave,
                                                            self.potential[:, :, i],
                                                            self.coordinates.dz, input_wave.E0)
        return WaveFunction(output_wave, input_wave.coordinates, input_wave.E0)

    def plot_potential(self, layer=None):
        if layer is None:
            plt.imshow(np.sum(self.potential))
        elif isinstance(layer, float):
            plt.imshow(self.potential[:, :, int(np.round(self.potential.shape[2] * layer))])
        elif isinstance(layer, int):
            plt.imshow(self.potential[:, :, layer])
        plt.show()


class LorentzNRotationPropagator(Propagator):
    # Rotate the wavefunction by theta and makes a lorentz transformation on it by beta_lattice
    def __init__(self, beta: float, theta: float):
        self.beta = beta
        self.theta = theta

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        X = input_wave.coordinates.X_grid
        phase_factor = np.exp(1j * (input_wave.E0 / H_BAR * self.beta / C_LIGHT + np.sin(self.theta)) * X)
        output_psi = input_wave.psi * phase_factor
        output_x_axis = input_wave.coordinates.x_axis / beta2gamma(self.beta)
        output_coordinates = CoordinateSystem((output_x_axis, input_wave.coordinates.y_axis))
        return WaveFunction(output_psi, output_coordinates, input_wave.E0)


class LensPropagator(Propagator):
    def __init__(self, focal_length: float, fft_shift: bool):
        self.focal_length = focal_length
        self.fft_shift = fft_shift

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        psi_FFT = np.fft.fftn(input_wave.psi, norm='ortho')
        fft_freq_x = np.fft.fftfreq(input_wave.psi.shape[0], input_wave.coordinates.dxdydz[0])
        fft_freq_y = np.fft.fftfreq(input_wave.psi.shape[1], input_wave.coordinates.dxdydz[1])

        if self.fft_shift:
            psi_FFT = np.fft.fftshift(psi_FFT)
            fft_freq_x = np.fft.fftshift(fft_freq_x)
            fft_freq_y = np.fft.fftshift(fft_freq_y)

        scale_factor = self.focal_length * E2l(input_wave.E0)
        new_axes = tuple([fft_freq_x * scale_factor, fft_freq_y * scale_factor])
        new_coordinates = CoordinateSystem(new_axes)
        output_wave = WaveFunction(psi_FFT, new_coordinates, input_wave.E0)
        return output_wave
