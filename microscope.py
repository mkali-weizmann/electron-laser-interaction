# %%
from abc import ABC

import numpy as np
from numpy import pi
from scipy.special import jv
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Callable
from warnings import warn
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path

M_ELECTRON = 9.1093837e-31
C_LIGHT = 299792458
H_BAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
FINE_STRUCTURE_CONST = 7.299e-3

np.seterr(all='raise')
from matplotlib.widgets import Slider, Button, RadioButtons


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


def manipulate_plot(imdata_callable: Callable,
                    values: Tuple[Tuple[float, float, float], ...],
                    labels: Optional[Tuple[str, ...]] = None):
    # from matplotlib import use
    # use('TkAgg')

    N_params = len(values)

    if labels is None:
        labels = [f'param{i}' for i in range(N_params)]

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0.25, bottom=(N_params) * 0.04 + 0.15)
    initial_data = imdata_callable(*[v[0] for v in values])
    if initial_data.ndim == 2:
        img = plt.imshow(initial_data)
    elif initial_data.ndim == 1:
        img, = plt.plot(initial_data)
    else:
        raise ValueError('imdata_callable must return 1 or 2 dimensional data')

    # axcolor = 'lightgoldenrodyellow'
    sliders_axes = [plt.axes([0.25, 0.04 * (i + 1), 0.65, 0.03], facecolor=(0.97, 0.97, 0.97)) for i in range(N_params)]
    sliders = [Slider(ax=sliders_axes[i],
                      label=labels[i],
                      valmin=values[i][0],
                      valmax=values[i][1],
                      valinit=values[i][0],
                      valstep=values[i][2]) for i in range(N_params)]

    def update(val):
        values = [slider.val for slider in sliders]
        if initial_data.ndim == 2:
            img.set_data(imdata_callable(*values))
        else:
            img.set_ydata(imdata_callable(*values))
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    if initial_data.ndim == 2:
        plt.colorbar(ax=ax)
    plt.show()


def gaussian_beam(x: [float, np.ndarray], y: [float, np.ndarray], z: Union[float, np.ndarray],
                  E,  # The amplitude of the electric field, not the potential A.
                  lamda: float, w_0: Optional[float] = None, NA: Optional[float] = None,
                  t: Optional[Union[float, np.ndarray]] = None,
                  mode: str = "intensity", is_cavity: bool = True) -> Union[np.ndarray, float]:
    if w_0 is None:
        w_0 = NA2w0(NA, lamda)
    # Calculates the electric field of a gaussian beam
    x_R = x_R_gaussian(w_0, lamda)
    w_x = w_x_gaussian(w_0=w_0, x=x, x_R=x_R)
    gouy_phase = gouy_phase_gaussian(x, x_R)
    k = l2k(lamda)
    R_x_inverse = x / (x_R ** 2 + x ** 2)
    other_phase = k * x + k * (z ** 2 + y ** 2) / 2 * R_x_inverse
    envelope = E * (w_0 / w_x) * np.exp(np.clip(-2 * (y ** 2 + z ** 2) / w_x ** 2, a_min=-500, a_max=None))

    if mode == "intensity":
        if is_cavity:  # No time dependence
            return envelope ** 2 * 4 * np.cos(other_phase + gouy_phase) ** 2
        else:
            return envelope ** 2

    elif mode in ["field", "potential"]:
        total_phase = other_phase - gouy_phase
        if t is not None:
            time_phase = np.exp(1j * ((C_LIGHT * k) * t))
        else:
            time_phase = 1
        if mode == "potential":  # The ratio between E and A in Gibbs gauge is E=w*A or A=E/w
            potential_factor = 1 / (C_LIGHT * k)  # == omega
        else:
            potential_factor = 1
        if is_cavity:
            return envelope * np.cos(total_phase) * time_phase * potential_factor
        else:
            return envelope * np.exp(1j * total_phase) * time_phase * potential_factor

    elif mode == "phase":
        total_phase = other_phase - gouy_phase
        if t is not None:
            time_phase = C_LIGHT * k * t
        else:
            time_phase = 0
        return total_phase + time_phase


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


class Cavity2FrequenciesPropagator(Propagator):
    def __init__(self,
                 l_1: float = 1064 * 1e-9,
                 l_2: Optional[float] = 532 * 1e-9,
                 E_1: float = 1,
                 E_2: Optional[float] = -1,
                 NA: float = 0.2,
                 alpha_cavity: Optional[float] = None,  # tilt angle of the lattice (of the cavity)
                 theta_polarization: Optional[float] = None):

        self.l_1: float = l_1  # Laser's frequency
        self.E_1: float = E_1  # Laser's amplitude
        if E_2 == -1:
            # -1 means that the second laser is defined by the condition for equal amplitudes in the lattices' frame
            self.E_2: float = E_1 * (l_1 / l_2)
        else:
            self.E_2: float = E_2
        self.l_2 = l_2
        self.NA: float = NA  # Cavity's numerical aperture
        self.alpha_cavity: Optional[float] = alpha_cavity  # cavity angle with respect to microscope's x-axis. positive
        # # number means the part of the cavity in the positive x direction is tilted downwards toward the positive z
        # direction.
        self.theta_polarization: Optional[float] = theta_polarization  # polarization angle of the laser

    @property
    def w_0(self) -> float:
        # The width of the gaussian beam of the first laser
        # based on https://en.wikipedia.org/wiki/Gaussian_beam
        return self.l / (pi * np.arcsin(self.NA))

    @property
    def w_0_min(self) -> float:

        return min((self.l_1, self.l_2)) / (pi * np.arcsin(self.NA))

    @property  # The velocity at which the standing wave is propagaing
    def beta_lattice(self) -> float:
        if self.E_2 is not None:
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
        if self.E_2 is None:
            return self.E_1
        else:
            return self.E_1 * self.Gamma_plus

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
        return self.E_2 * self.Gamma_plus

    @property
    def A_minus(self) -> float:
        return self.E_1 * self.Gamma_minus

    @property
    def k(self) -> float:
        return l2k(self.l)

    @property
    def min_l(self):
        return min(self.l_1, self.l_2)

    @property
    def max_l(self):
        return max(self.l_1, self.l_2)

    def beta_electron2alpha_cavity(self, beta_electron: Optional[float] = None) -> float:
        if beta_electron is not None and self.alpha_cavity is None:
            return np.arcsin(self.beta_lattice / beta_electron)
        elif beta_electron is None and self.alpha_cavity is None:
            raise ValueError("Either beta_electron or alpha_cavity must be given")
        else:
            warn("alpha_cavity is not None. Using the value given by the user, Note that the calculations assume"
                 "that the lattice satisfy sin(alpha_cavity) = beta_lattice / beta_electron")
            return self.alpha_cavity


class Cavity2FrequenciesAnalyticalPropagator(Cavity2FrequenciesPropagator):
    def __init__(self,
                 l_1: float = 1064 * 1e-9,
                 l_2: Optional[float] = 532 * 1e-9,
                 E_1: float = 1,
                 E_2: Optional[float] = -1,
                 NA: float = 0.2,
                 alpha_cavity: Optional[float] = None,  # tilt angle of the lattice (of the cavity)
                 theta_polarization: Optional[float] = None):

        super().__init__(l_1, l_2, E_1, E_2, NA, alpha_cavity, theta_polarization)

    def phi_0(self, x: np.ndarray, y: np.ndarray, beta_electron: float) -> np.ndarray:
        # Gives the phase acquired by a narrow electron beam centered around (x, y) by passing in the cavity_2f.
        # Does not include the relativistic correction.
        # According to equation e_10 and equation gaussian_beam_potential_total_phase in my lyx file
        alpha_cavity = self.beta_electron2alpha_cavity(beta_electron)
        x_lattice = x / np.cos(alpha_cavity)
        x_R = x_R_gaussian(self.w_0, self.l)
        w_x = w_x_gaussian(w_0=self.w_0, x=x_lattice, x_R=x_R)
        # The next two lines are based on equation e_11 in my lyx file
        constant_coefficients = (E_CHARGE ** 2 * self.w_0 ** 2 * np.sqrt(pi) * self.A ** 2) / \
                                (H_BAR * 4 * np.sqrt(2) * M_ELECTRON * C_LIGHT * beta_electron * beta2gamma(
                                    beta_electron))
        spatial_envelope = np.exp(  # Shouldn't be here a w_0 term? No! it is in the previous term.
            np.clip(-2 * y ** 2 / w_x ** 2, a_min=-500, a_max=None)) / w_x  # ARBITRARY CLIPPING FOR STABILITY
        if self.E_2 is None:  # For the case of a single laser, add the spatial cosine
            gouy_phase = gouy_phase_gaussian(x_lattice, x_R, self.w_0)
            cosine_squared = 4 * np.cos(4 * np.pi * x_lattice / self.l + gouy_phase) ** 2
            spatial_envelope *= cosine_squared
        return constant_coefficients * spatial_envelope

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        phi_0 = self.phi_0(input_wave.coordinates.X_grid, input_wave.coordinates.Y_grid, E2beta(input_wave.E0))
        phase_shift = self.phase_shift(phi_0)
        if self.E_2 is not None:  # For the case of a double laser:
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
        if self.E_2 is not None:  # For the case of a double laser:
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


class Cavity2FrequenciesNumericalPropagator(Cavity2FrequenciesPropagator):
    def __init__(self,
                 l_1: float = 1064 * 1e-9,
                 l_2: Optional[float] = 532 * 1e-9,
                 E_1: float = 1,
                 E_2: Optional[float] = -1,
                 NA: float = 0.2,
                 alpha_cavity: Optional[float] = None,  # tilt angle of the lattice (of the cavity)
                 theta_polarization: Optional[float] = None,
                 ignore_past_files: bool = False,
                 debug_mode: bool = False,
                 n_z: int = 500,  # ARBITRARY,
                 n_t: int = 100):  # ARBITRARY
        super().__init__(l_1, l_2, E_1, E_2, NA, alpha_cavity, theta_polarization)
        self.ignore_past_files = ignore_past_files
        self.n_z = n_z
        self.n_t = n_t
        self.debug_mode = debug_mode

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        setup_file_path = self.setup_to_path(input_wave=input_wave)
        # if this setup was calculated once in the past and the user did not ask to ignore past files, load the file
        if os.path.isfile(setup_file_path) and not self.ignore_past_files:
            phase_and_amplitude_mask = np.load(setup_file_path)
        else:
            phase_and_amplitude_mask = self.generate_phase_and_amplitude_mask(input_wave)
            np.save(setup_file_path, phase_and_amplitude_mask)
        output_wave_psi = input_wave.psi * phase_and_amplitude_mask
        return WaveFunction(output_wave_psi, input_wave.coordinates, input_wave.E0)

    def generate_phase_and_amplitude_mask(self, input_wave: WaveFunction):
        phi_values = self.phi(input_wave.coordinates.x_axis, input_wave.coordinates.y_axis, E2beta(input_wave.E0))
        phase_and_amplitude_mask = self.extract_0_energy_level_amplitude(phi_values)
        return phase_and_amplitude_mask

    def setup_to_path(self, input_wave: WaveFunction) -> str:
        # This function is used to generate a unique name for each setup, such that we can load previous results if they
        # exist.
        path = f'Data Arrays\\Phase Masks\\2f_l1{self.l_1 * 1e9:.4g}_l2{self.l_2 * 1e9:.4g}_' \
               f'E1{self.E_1:.3g}_E2{self.E_2:.3g}_NA{self.NA * 100:.4g}_' \
               f'alpha{self.beta_electron2alpha_cavity(E2beta(input_wave.E0)) /2*np.pi * 360:.0f}_' \
               f'theta{self.theta_polarization * 100:.4g}_E{input_wave.E0:.2g}_' \
               f'Nx{input_wave.coordinates.x_axis.size}_Ny{input_wave.coordinates.y_axis.size}_Nz{self.n_z}_' \
               f'Nt{self.n_t}_DX{input_wave.coordinates.limits[1]:.4g}_DY{input_wave.coordinates.limits[3]:.4g}.npy'

        return path

    def rotated_gaussian_beam_A(self,
                                x: [float, np.ndarray],
                                y: [float, np.ndarray],
                                z: [float, np.ndarray],
                                t: [float, np.ndarray],
                                beta_electron: Optional[float] = None
                                ) -> Union[np.ndarray, float]:
        alpha_cavity = self.beta_electron2alpha_cavity(beta_electron)

        x_tilde = x * np.cos(alpha_cavity) - z * np.sin(alpha_cavity)
        z_tilde = x * np.sin(alpha_cavity) + z * np.cos(alpha_cavity)

        A_1 = gaussian_beam(x=x_tilde, y=y, z=z_tilde, E=self.E_1, lamda=self.l_1, NA=self.NA, t=t,
                            mode="potential")
        A_2 = gaussian_beam(x=x_tilde, y=y, z=z_tilde, E=self.E_2, lamda=self.l_2, NA=self.NA, t=t,
                            mode="potential")

        A = A_1 + A_2

        return A

    def generate_coordinates_lattice(self,
                           x: [float, np.ndarray],
                           y: [float, np.ndarray],
                           t: [float, np.ndarray],
                           beta_electron: float):
        z = np.linspace(-5, 5, self.n_z)  # ARBITRARY this number is in units of spot size
        # z = np.linspace(-80e-6, 80e-6, self.n_z)  # DELETE ME
        X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')
        alpha = self.beta_electron2alpha_cavity(beta_electron)
        Z *= w_x_gaussian(w_0=NA2w0(self.NA, self.min_l), x=X/np.cos(alpha), l_laser=self.min_l) / np.cos(alpha)  # This scales
        # the z axis to the size of the beam spot_size at each x. x is divided by cos_alpha because the beam is tilted
        Z -= X * np.tan(alpha)  # This make the Z coordinates centered around
        # the cavity axis - which depend on the angle of the cavity and the x coordinate.
        T += Z / (C_LIGHT * beta_electron)  # This makes T be the time of the electron that passed through z=0 at
        # time t and then got to Z after/before Z / (beta * c) time.

        return X, Y, Z, T

    @staticmethod
    def G_gauge(A_shifted: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = Z[:, :, 1, :] - Z[:, :, 0, :]
        # This dZ assumes different dz for different x's. I use this variating dZ because I want to integrate over, for
        # example, range of [-4w(x), 4*w(x)] for every x, and the spot size w(x) changes with x.
        # Since dz does not vary along z, it is enough to take the first two values of dZ.
        G_gauge_values = np.cumsum(A_shifted, axis=2) * dZ[:, :, np.newaxis, :]

        return G_gauge_values  # integral over z

    def phi_integrand(self,
                      x: [float, np.ndarray],
                      y: [float, np.ndarray],
                      t: [float, np.ndarray],
                      beta_electron: float,
                      save_to_file: bool = False):
        # WITHOUT THE PREFACTORS: this is the integrand of the integral over z of the phase shift.
        X, Y, Z, T = self.generate_coordinates_lattice(x, y, t, beta_electron)
        A = self.rotated_gaussian_beam_A(X, Y, Z, T, beta_electron)
        G = self.G_gauge(A, Z) * np.cos(self.theta_polarization)  # Cos because this is the z component of A
        dG_dx = np.gradient(G, x[1] - x[0], axis=0)

        if save_to_file:
            np.save("Data Arrays\\Debugging Arrays\\A.npy", A)
            np.save("Data Arrays\\Debugging Arrays\\G.npy", G)

        if self.theta_polarization == pi / 2:
            raise NotImplementedError("The case of theta_polarization = pi/2 is not implemented yet")

        integrand = (np.abs(A) * np.sin(self.theta_polarization)) ** 2 + \
                    (1 - beta_electron ** 2) * (
                            np.clip(np.abs(A), a_min=1e-50, a_max=None) * np.cos(self.theta_polarization)) ** 2 + \
                    np.clip(np.abs(dG_dx), a_min=1e-50, a_max=None) ** 2
        return integrand, Z  # The Z is returned so that it can be used in the integration over z.

    def phi_single_batch(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, beta_electron: float,
                         save_to_file: bool = False):
        phi_integrand, Z = self.phi_integrand(x, y, t, beta_electron, save_to_file=save_to_file)
        if save_to_file:
            np.save("Data Arrays\\Debugging Arrays\\phi_integrand.npy", phi_integrand)
        prefactor = - 1 / H_BAR * E_CHARGE ** 2 / (2 * M_ELECTRON * beta2gamma(beta_electron) * beta_electron * C_LIGHT)
        phi_values = prefactor * np.trapz(phi_integrand, x=Z, axis=2)
        return phi_values

    def phi(self, x: np.ndarray, y: np.ndarray, beta_electron: float):
        # This if else clause is for the case of a single laser or a double laser:
        if self.l_2 is None:  # For the case of a single laser there is no time dependence for the phase mask.
            t = np.array([0])
            phi_values = self.phi_single_batch(x, y, t, beta_electron)
        else:
            # This whole while loop gymnastics is to make sure that the time step is small enough to avoid arrays that
            # are too big for the memory of the computer to hold. That is, we divide the calculation into batches
            # of time steps that are small enough to fit in the memory.
            grid_size_per_t = len(x) * len(y) * self.n_z
            grid_size_max = 1e7
            # The maximal number of time steps that can be calculated in one batch (the total capacity divided by the
            # size of one time step array), 1 in case of a batch smaller than 1.
            n_t_batch_max = max(1, int(np.floor(grid_size_max / grid_size_per_t)))
            total_t_needed = self.n_t  # ARBITRARY
            phi_values = np.zeros((len(x),
                                   len(y),
                                   total_t_needed))
            # For a complete description of the phase shift it is enough to look at one cycle of field (ignoring global
            # phase), which is given by delta omega between the lasers (denoted as w):
            delta_w = np.abs(l2w(self.l_1) - l2w(self.l_2))
            t = np.linspace(0, 20 * pi / delta_w, total_t_needed)  # ARBITRARY total_t_needed
            n_t_done = 0
            last_run_save_to_file = False
            while n_t_done < total_t_needed:
                if self.debug_mode:
                    print('n_t_done = ', n_t_done)
                    if n_t_done + n_t_batch_max >= total_t_needed:
                        last_run_save_to_file = True
                n_t_batch = min(n_t_batch_max, total_t_needed - n_t_done)
                t_temp = t[n_t_done:n_t_done + n_t_batch]
                phi_values[:, :, n_t_done:n_t_done + n_t_batch] = self.phi_single_batch(x,
                                                                                        y,
                                                                                        t_temp, beta_electron,
                                                                                        save_to_file=last_run_save_to_file)
                n_t_done += n_t_batch
        if self.debug_mode:
            print('Finished calculating phi for every x, y, z, t')
            np.save('Data Arrays\\Debugging Arrays\\phi_values.npy', phi_values)
        return phi_values

    def extract_0_energy_level_amplitude(self, phi_values: np.ndarray):
        phase_factor = np.exp(1j * phi_values)
        if self.l_2 is None:  # If there is only one mode, then the phase is constant in time and there is no amplitude
            # modulation.
            return phase_factor
        else:
            energy_bands = np.fft.fft(phase_factor, axis=-1, norm='forward')
            phase_amplitude_mask = np.sum(energy_bands[:, :, [0, 1, -1]], axis=2)  # ARBITRARY
            if self.debug_mode:
                np.save("Data Arrays\\Debugging Arrays\\phase_amplitude_mask.npy", phase_amplitude_mask)
            return phase_amplitude_mask


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


if __name__ == '__main__':
    C = Cavity2FrequenciesNumericalPropagator(l_1=1064 * 1e-9,
                                              l_2=532 * 1e-9,
                                              E_1=3.175e9,
                                              E_2=-1,
                                              NA=0.1,
                                              n_z=800,
                                              n_t=100,
                                              alpha_cavity=None,  # tilt angle of the lattice (of the cavity)
                                              theta_polarization=0,
                                              ignore_past_files=True,
                                              debug_mode=True)
    n_x = 120
    n_y = 1
    input_coordinate_system = CoordinateSystem(lengths=(300e-6, 0),
                                               n_points=(n_x, n_y))
    input_wave = WaveFunction(psi=np.ones((n_x, n_y)),
                              coordinates=input_coordinate_system,
                              E0=KeV2Joules(300))

    output_wave = C.propagate(input_wave)
