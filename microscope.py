# %%
import numpy as np
from numpy import pi
from scipy.special import jv
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Callable
from warnings import warn
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
from scipy import integrate
from matplotlib.widgets import Slider

M_ELECTRON = 9.1093837e-31
C_LIGHT = 299792458
H_BAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
FINE_STRUCTURE_CONST = 7.299e-3

np.seterr(all='raise')


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


def l2k(lambda_laser: float) -> float:
    return 2 * pi / lambda_laser


def k2w(k: float, n=1) -> float:
    return C_LIGHT * k / n


def w2k(w: float, n=1) -> float:
    return n * w / C_LIGHT


def l2w(lambda_laser: float) -> float:
    return k2w(k2l(lambda_laser))


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


def x_R_gaussian(w0: float, lambda_laser: float) -> float:
    # l is the wavelength of the laser
    return pi * w0 ** 2 / lambda_laser


def NA2w0(NA: float, lambda_laser: float) -> float:
    return lambda_laser / (np.pi * NA)


def w02NA(w0: float, lambda_laser: float) -> float:
    return lambda_laser / (np.pi * w0)


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


def R_x_inverse_gaussian(x, x_R=None, l_laser=None, w_0=None):
    if x_R is None:
        x_R = x_R_gaussian(w_0, l_laser)
    return x / (x ** 2 + x_R ** 2)


def manipulate_plot(imdata_callable: Callable,
                    values: Tuple[Tuple[float, float, float], ...],
                    labels: Optional[Tuple[str, ...]] = None):
    # from matplotlib import use
    # use('TkAgg')

    N_params = len(values)

    if labels is None:
        labels = [f'param{i}' for i in range(N_params)]

    fig, ax = plt.subplots()

    plt.subplots_adjust(left=0.25, bottom=N_params * 0.04 + 0.15)
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
                  E: float,  # The amplitude of the electric field, not the potential A.
                  lambda_laser: float,
                  w_0: Optional[float] = None,
                  NA: Optional[float] = None,
                  t: Optional[Union[float, np.ndarray]] = None,
                  mode: str = "intensity",
                  standing_wave: bool = True,
                  forward_propagation: bool = True
                  ) -> Union[np.ndarray, float]:
    if w_0 is None:
        w_0 = NA2w0(NA, lambda_laser)
    # Calculates the electric field of a gaussian beam
    x_R = x_R_gaussian(w_0, lambda_laser)
    w_x = w_x_gaussian(w_0=w_0, x=x, x_R=x_R)
    gouy_phase = gouy_phase_gaussian(x, x_R)
    R_x_inverse = R_x_inverse_gaussian(x, x_R)
    k = l2k(lambda_laser)
    other_phase = k * x + k * (z ** 2 + y ** 2) / 2 * R_x_inverse
    envelope = E * (w_0 / w_x) * np.exp(np.clip(-(y ** 2 + z ** 2) / w_x ** 2, a_min=-500, a_max=None))  # The clipping
    if forward_propagation is True:
        propagation_sign = 1
    else:
        propagation_sign = -1
    # is to prevent underflow errors.

    if mode == "intensity":
        if standing_wave:  # No time dependence
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
            potential_factor = 1 / (C_LIGHT * k)  # potential_factor = 1/omega
        else:
            potential_factor = 1
        if standing_wave:
            return envelope * np.cos(total_phase) * time_phase * potential_factor
        else:
            return envelope * np.exp(propagation_sign * 1j * total_phase) * time_phase * potential_factor

    elif mode == "phase":
        total_phase = propagation_sign * (other_phase - gouy_phase)
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


@dataclass()
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

    def take_a_picture(self, input_wave: WaveFunction) -> Tuple[np.ndarray, CoordinateSystem]:
        output_wave = self.propagate(input_wave)
        image = self.expose_camera(output_wave)
        return image

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        self.propagation_steps = []
        for propagator in self.propagators:
            if self.print_progress:
                print(f"Propagating with {propagator}")
            output_wave = propagator.propagate(input_wave)
            self.propagation_steps.append(PropagationStep(input_wave, output_wave, propagator))
            input_wave = output_wave
        return input_wave

    def expose_camera(self, output_wave: Optional[WaveFunction] = None) -> Tuple[np.ndarray, CoordinateSystem]:
        if len(self.propagation_steps) == 0:
            raise ValueError("You must propagate the wave first")

        if output_wave is None:
            output_wave = self.propagation_steps[-1].output_wave

        output_wave_intensity = np.abs(output_wave.psi) ** 2
        output_wave_intensity_shot_noise = np.random.poisson(output_wave_intensity)  # This actually assumes an
        # independent shot noise for each pixel, which is not true, but it's a good approximation for many pixels.
        return output_wave_intensity_shot_noise, self.propagation_steps[-1].output_wave.coordinates

    def plot_step(self, propagator: Propagator, clip=True, title=None, file_name=None):
        step_idx = self.propagators.index(propagator)
        step = self.propagation_steps[step_idx]
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
            potential_2d = np.load("Data Arrays\\Static Data\\letters_big_1024.npy")
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


class AberrationsPropagator(Propagator):
    def __init__(self, Cs, defocus,
                 atigmatism_parameter: float = 0,
                 astigmatism_orientation: float = 0):
        self.Cs = Cs
        self.defocus = defocus
        self.astigmatism_parameter = atigmatism_parameter
        self.astigmatism_orientation = astigmatism_orientation

    # This is somewhat inefficient, because we could add all the aberrations in the fourier plane, but I want to
    # separate the cavity calculations from the aberrations calculations, and in particular not to assume a 4f system,
    # and not to be dependent on knowing the focal length of the lens when adding aberrations. anyway in the typical
    # sizes of images (up to 1000X1000 pixels) it does not take long to fft.

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        psi_FFT = np.fft.fftn(input_wave.psi, norm='ortho')
        fft_freq_x = np.fft.fftfreq(input_wave.psi.shape[0], input_wave.coordinates.dxdydz[0]) * 2 * np.pi
        fft_freq_y = np.fft.fftfreq(input_wave.psi.shape[1], input_wave.coordinates.dxdydz[1]) * 2 * np.pi
        aberrations_mask = self.aberrations_mask(fft_freq_x, fft_freq_y, input_wave.E0)
        psi_FFT_aberrated = psi_FFT * aberrations_mask
        psi_aberrated = np.fft.ifftn(psi_FFT_aberrated, norm='ortho')
        output_wave = WaveFunction(psi_aberrated, input_wave.coordinates, input_wave.E0)
        return output_wave

    def aberrations_mask(self, f_x: np.array, f_y: np.array, E0: float):
        k_x, k_y = np.meshgrid(f_x, f_y, indexing='ij')
        k_squared = k_x ** 2 + k_y ** 2
        phi_k = np.arctan2(k_y, k_x)
        f_defocus_aberration = self.defocus + \
                                    self.astigmatism_parameter * np.cos(2 * (phi_k - self.astigmatism_orientation))
        lambda_electron = E2l(E0)
        phase = np.pi * ((
                                 1 / 2) * self.Cs * lambda_electron ** 3 * k_squared ** 2 - f_defocus_aberration)
        return np.exp(1j * phase)  # Check sign!


class Cavity2FrequenciesPropagator(Propagator):
    def __init__(self,
                 l_1: float = 1064 * 1e-9,
                 l_2: Optional[float] = 532 * 1e-9,
                 E_1: float = 1,
                 E_2: Optional[float] = -1,
                 NA_1: float = 0.1,
                 NA_2: Optional[float] = -1,
                 theta_polarization: float = np.pi / 2,
                 alpha_cavity: Optional[float] = None,  # tilt angle of the lattice (of the cavity)
                 ring_cavity: bool = False
                 ):

        self.l_1: float = l_1  # Laser's frequency
        self.E_1: float = E_1  # Laser's amplitude
        if E_2 == -1:
            # -1 means that the second laser is defined by the condition for equal amplitudes in the lattices' frame
            self.E_2: float = E_1 * (l_1 / l_2)
        else:
            self.E_2: float = E_2
        self.l_2 = l_2
        self.NA_1: float = NA_1  # Cavity's numerical aperture
        if NA_2 == -1:
            if l_2 is None:
                self.NA_2 = None
            else:
                self.NA_2 = NA_1 * np.sqrt(self.l_2 / self.l_1)
        else:
            self.NA_2 = NA_2
        self.alpha_cavity: Optional[float] = alpha_cavity  # cavity angle with respect to microscope's x-axis. positive
        # # number means the part of the cavity in the positive x direction is tilted downwards toward the positive z
        # direction.
        self.theta_polarization: float = theta_polarization  # polarization angle of the laser
        self.ring_cavity: bool = ring_cavity

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        raise NotImplementedError

    def NA_lorentz_transform(self, NA):
        return np.sin(np.arctan(beta2gamma(self.beta_lattice) * np.tan(np.arcsin(NA))))

    @property
    def NA_min(self):
        return min(self.NA_1, self.NA_2)

    @property
    def NA_max(self):
        return max(self.NA_1, self.NA_2)

    @property
    def w_0_lattice_frame(self) -> float:
        # The width of the gaussian beam of the first laser - of the only laser if there is one laser, and of the
        # lattice in the moving frame if there are two (the formula is from wikipedia)
        return self.lambda_laser / (pi * np.arcsin(self.NA_lorentz_transform(self.NA_1)))

    @property
    def w_0_min(self) -> float:
        return min(self.l_1 / (pi * np.arcsin(self.NA_1)), self.l_2 / (pi * np.arcsin(self.NA_2)))

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
            return self.E_1 / l2w(self.lambda_laser)
        else:
            return self.E_1 * self.Gamma_plus / l2w(self.lambda_laser)

    @property
    def lambda_laser(self) -> float:
        # The effective wavelength of the lattice in the moving frame. in case of a single frequency that is just
        # the wavelength of the first laser.
        if self.l_2 is None:
            return self.l_1
        else:
            return self.l_1 / self.Gamma_plus

    @property
    def k(self) -> float:
        # If there is only one laser it is the k of that laser and if there are two it is the k of the lattice in the
        # lattice's frame
        return l2k(self.lambda_laser)

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
                 NA_1: float = 0.1,
                 NA_2: Optional[float] = -1,
                 theta_polarization: float = np.pi / 2,
                 alpha_cavity: Optional[float] = None,  # tilt angle of the lattice (of the cavity)
                 debug_mode: bool = False,
                 ring_cavity: bool = True):

        super().__init__(l_1, l_2, E_1, E_2, NA_1, NA_2, theta_polarization, alpha_cavity, ring_cavity)
        self.debug_mode = debug_mode

    def phi_0(self, x: np.ndarray, y: np.ndarray, beta_electron: float) -> np.ndarray:
        # Gives the phase acquired by a narrow electron beam centered around (x, y) by passing in the cavity_2f.
        # Does not include the relativistic correction.
        # According to equation e_10 and equation e_1 in my readme file
        alpha_cavity = self.beta_electron2alpha_cavity(beta_electron)
        x_lattice = x / np.cos(alpha_cavity)
        x_R = x_R_gaussian(self.w_0_lattice_frame, self.lambda_laser)
        w_x = w_x_gaussian(w_0=self.w_0_lattice_frame, x=x_lattice, x_R=x_R)
        # The next two lines are based on equation e_11 in my readme file
        constant_coefficients = (E_CHARGE ** 2 * self.w_0_lattice_frame ** 2 * np.sqrt(pi) * self.A ** 2) / \
                                (H_BAR * 4 * np.sqrt(2) * M_ELECTRON * C_LIGHT * beta_electron * beta2gamma(
                                    beta_electron))
        spatial_envelope = np.exp(  # Shouldn't be here a w_0 term? No! it is in the previous term.
            np.clip(-2 * y ** 2 / w_x ** 2, a_min=-500, a_max=None)) / w_x  # ARBITRARY CLIPPING FOR STABILITY
        if self.E_2 is None:  # For the case of a single laser, add the spatial cosine
            gouy_phase = gouy_phase_gaussian(x_lattice, x_R, self.w_0_lattice_frame)
            cosine_squared = 4 * np.cos(4 * np.pi * x_lattice / self.lambda_laser + gouy_phase) ** 2
            spatial_envelope *= cosine_squared
        return constant_coefficients * spatial_envelope

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        phi_0 = self.phi_0(input_wave.coordinates.X_grid, input_wave.coordinates.Y_grid, E2beta(input_wave.E0))
        phase_shift = self.phase_shift(phi_0)
        if self.E_2 is not None:  # For the case of a double laser:
            attenuation_factor = self.attenuation_factor(E2beta(input_wave.E0), phi_0, input_wave.coordinates.X_grid,
                                                         input_wave.coordinates.Y_grid)
        else:
            attenuation_factor = 1
        output_wave = input_wave.psi * np.exp(-1j * phase_shift) * attenuation_factor  # ARBITRARY ARBITRARY - I FIXED
        # A SIGN MISTAKE BY ADDING A SIGN
        return WaveFunction(output_wave, input_wave.coordinates, input_wave.E0)

    def phase_shift(self, phi_0: Optional[np.ndarray] = None, X_grid: Optional[np.ndarray] = None,
                    Y_grid: Optional[np.ndarray] = None, beta_electron: Optional[float] = None):
        if phi_0 is None:
            phi_0 = self.phi_0(X_grid, Y_grid, beta_electron)
        if self.E_2 is not None:  # For the case of a double laser, explanation in equation eq:e_32 in my readme file
            if self.ring_cavity:
                return 2 * phi_0
            else:
                return phi_0 * (2 + (self.Gamma_plus / self.Gamma_minus) ** 2 +
                                (self.Gamma_minus / self.Gamma_plus) ** 2)
        else:
            return phi_0

    def attenuation_factor(self, beta_electron: float, phi_0: Optional[np.ndarray] = None,
                           X_grid: Optional[np.ndarray] = None,
                           Y_grid: Optional[np.ndarray] = None):
        if phi_0 is None:
            phi_0 = self.phi_0(X_grid, Y_grid, beta_electron)
        return jv(0, 2 * phi_0 * self.rho(beta_electron))

    def rho(self, beta_electron: float):
        return 1 - 2 * beta_electron ** 2 * np.cos(self.theta_polarization) ** 2


class Cavity2FrequenciesNumericalPropagator(Cavity2FrequenciesPropagator):
    def __init__(self,
                 l_1: float = 1064 * 1e-9,
                 l_2: Optional[float] = 532 * 1e-9,
                 E_1: float = 1,
                 E_2: Optional[float] = -1,
                 NA_1: float = 0.1,
                 NA_2: Optional[float] = -1,
                 theta_polarization: float = np.pi / 2,
                 alpha_cavity: Optional[float] = None,  # tilt angle of the lattice (of the cavity)
                 ring_cavity: bool = True,
                 ignore_past_files: bool = False,
                 debug_mode: bool = False,
                 n_t: int = 3):  # ARBITRARY
        super().__init__(l_1, l_2, E_1, E_2, NA_1, NA_2, theta_polarization, alpha_cavity, ring_cavity)
        self.ignore_past_files = ignore_past_files
        self.n_t = n_t
        self.debug_mode = debug_mode

    def propagate(self, input_wave: WaveFunction) -> WaveFunction:
        phase_and_amplitude_mask = self.generate_phase_and_amplitude_mask(input_wave=input_wave)
        output_wave_psi = input_wave.psi * phase_and_amplitude_mask
        return WaveFunction(output_wave_psi, input_wave.coordinates, input_wave.E0)

    def generate_phase_and_amplitude_mask(self, input_wave: WaveFunction):
        setup_file_path = self.setup_to_path(input_wave=input_wave)
        # if this setup was calculated once in the past and the user did not ask to ignore past files, load the file
        if os.path.isfile(setup_file_path) and not self.ignore_past_files:
            phase_and_amplitude_mask = np.load(setup_file_path)
        else:
            phi_values = self.phi(input_wave.coordinates.x_axis, input_wave.coordinates.y_axis, E2beta(input_wave.E0))
            phase_and_amplitude_mask = self.extract_0th_energy_level_amplitude(phi_values)
            np.save(setup_file_path, phase_and_amplitude_mask)
        return phase_and_amplitude_mask

    def extract_0th_energy_level_amplitude(self, phi_values: np.ndarray):
        phase_factor = np.exp(1j * phi_values)
        if self.l_2 is None:  # If there is only one mode, then the phase is constant in time and there is no amplitude
            # modulation.
            return phase_factor
        else:
            if self.n_t == 3:
                # This assumes that if there are exactly 3 time steps, then they are [0, pi/(2delta_w), pi/delta_w]:
                # The derivation is in equation eq:27 in my readme file.
                phi_0 = 1 / 2 * (phi_values[:, :, 0] + phi_values[:, :, 2])
                varphi = np.arctan2(phi_values[:, :, 0] - phi_0, phi_values[:, :, 1] - phi_0)
                sin_varphi = np.sin(varphi)
                problematic_elements = np.abs(sin_varphi) < 1e-3  # This e-3 clipping value will not affect the result.
                sin_varphi_no_small_values = np.where(problematic_elements, 1, sin_varphi)
                cos_varphi_no_small_values = np.where(problematic_elements, np.cos(varphi), 1)
                A_computed_with_sin = np.where(problematic_elements,
                                               0,
                                               (phi_values[:, :, 0] - phi_0) / sin_varphi_no_small_values
                                               )
                A_computed_with_cos = np.where(problematic_elements,
                                               (phi_values[:, :, 1] - phi_0) / cos_varphi_no_small_values,
                                               0)
                A = A_computed_with_sin + A_computed_with_cos
                phase_and_amplitude_mask = jv(0, A) * np.exp(1j * phi_0)
            else:
                # NOT ACCURATE UNLESS N_T IS BIG!
                energy_bands = np.fft.fft(phase_factor, axis=-1, norm='forward')
                phase_and_amplitude_mask = energy_bands[:, :, 0]
            if self.debug_mode:
                np.save("Data Arrays\\Debugging Arrays\\phase_amplitude_mask.npy", phase_and_amplitude_mask)
            return phase_and_amplitude_mask

    def phi(self, x: np.ndarray, y: np.ndarray, beta_electron: float):
        # This if else clause is for the case of a single laser or a double laser:
        if self.l_2 is None:  # For the case of a single laser there is no time dependence for the phase mask.
            t = np.array([0])
            phi_values = self.phi_single_batch(x, y, t, beta_electron)
        else:
            # This whole while loop gymnastics is to make sure that the time step is small enough to avoid arrays that
            # are too big for the memory of the computer to hold. That is, we divide the calculation into batches
            # of time steps that are small enough to fit in the memory.
            grid_size_per_t = len(x) * len(y) * len(self.generate_z_vector(x, beta_electron))
            grid_size_max = 1e7
            # The maximal number of time steps that can be calculated in one batch (the total capacity divided by the
            # size of one time step array), 1 in case of a batch smaller than 1.
            n_t_batch_max = max(1, int(np.floor(grid_size_max / grid_size_per_t)))
            phi_values = np.zeros((len(x),
                                   len(y),
                                   self.n_t))  # no n_z because it will be integrated over.
            # For a complete description of the phase shift it is enough to look at one cycle of field (ignoring global
            # phase), which is given by delta omega between the lasers (denoted as w):
            delta_w = np.abs(l2w(self.l_1) - l2w(self.l_2))
            if self.n_t == 3:
                t = np.array([0, pi / (2 * delta_w), pi / delta_w])
            else:
                t = np.linspace(0, 20 * pi / delta_w, self.n_t)  # ARBITRARY total_t_needed
            n_t_done = 0
            first_run_save_to_file = True
            while n_t_done < self.n_t:
                if self.debug_mode:
                    print('n_t_done = ', n_t_done)

                n_t_batch = min(n_t_batch_max, self.n_t - n_t_done)
                t_temp = t[n_t_done:n_t_done + n_t_batch]
                phi_values[:, :, n_t_done:n_t_done + n_t_batch] = self.phi_single_batch(x,
                                                                                        y,
                                                                                        t_temp, beta_electron,
                                                                                        save_to_file=first_run_save_to_file)
                first_run_save_to_file = False
                n_t_done += n_t_batch
        if self.debug_mode:
            print('Finished calculating phi for every x, y, z, t')
            np.save('Data Arrays\\Debugging Arrays\\phi_values.npy', phi_values)
        return phi_values

    def phi_single_batch(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, beta_electron: float,
                         save_to_file: bool = False):
        phi_integrand, Z = self.phi_integrand(x, y, t, beta_electron, save_to_file=save_to_file)
        if save_to_file:
            np.save("Data Arrays\\Debugging Arrays\\phi_integrand.npy", phi_integrand)
        prefactor = - 1 / H_BAR * E_CHARGE ** 2 / (2 * M_ELECTRON * beta2gamma(beta_electron) * beta_electron * C_LIGHT)
        phi_values = prefactor * np.trapz(phi_integrand, x=Z, axis=2)
        return phi_values

    def phi_integrand(self,
                      x: [float, np.ndarray],
                      y: [float, np.ndarray],
                      t: [float, np.ndarray],
                      beta_electron: float,
                      save_to_file: bool = False):
        # WITHOUT THE PREFACTORS: this is the integrand of the integral over z of the phase shift.
        X, Y, Z, T = self.generate_coordinates_lattice(x, y, t, beta_electron)
        A_vector = self.rotated_gaussian_beam_A(X, Y, Z, T, beta_electron)

        grad_G = self.grad_G(X, Y, Z, T,
                             A_z=A_vector[:, :, :, :, 2],
                             beta_electron=beta_electron,
                             save_to_file=save_to_file)

        if save_to_file:
            np.save("Data Arrays\\Debugging Arrays\\A.npy", A_vector)

        integrand = np.sum(np.abs(A_vector - grad_G) ** 2, axis=-1) - \
                    beta_electron ** 2 * np.abs(A_vector[:, :, :, :, 2] - grad_G[:, :, :, :, 2]) ** 2
        return integrand, Z  # The Z is returned so that it can be used later in the integration of the integrand
        # over z.

    def generate_coordinates_lattice(self,
                                     x: [float, np.ndarray],
                                     y: [float, np.ndarray],
                                     t: [float, np.ndarray],
                                     beta_electron: float):

        alpha_cavity = self.beta_electron2alpha_cavity(beta_electron)

        z = self.generate_z_vector(x=x, beta_electron=beta_electron)

        X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')

        Z *= w_x_gaussian(w_0=NA2w0(self.NA_max, self.min_l), x=X / np.cos(alpha_cavity), l_laser=self.min_l) / np.cos(
            alpha_cavity)  # This scales the z axis to the size of the beam spot_size at each x. x is divided by
        # cos_alpha because the beam is tilted, and so does the spot size itself.
        Z -= X * np.tan(alpha_cavity)  # This make the Z coordinates centered around
        # the cavity axis - which depend on the angle of the cavity and the x coordinate.
        T += Z / (C_LIGHT * beta_electron)  # This makes T be the time of the electron that passed through z=0 at
        # time t and then got to Z after/before: Z/(beta * c) time. (that is, t=t(z))

        return X, Y, Z, T

    def rotated_gaussian_beam_A(self,
                                x: [float, np.ndarray],
                                y: [float, np.ndarray],
                                z: [float, np.ndarray],
                                t: [float, np.ndarray],
                                beta_electron: Optional[float] = None,
                                return_vector: bool = True) -> Union[np.ndarray, float]:

        alpha_cavity = self.beta_electron2alpha_cavity(beta_electron)

        # The derivation of those rotated coordinates is in eq:e_25 in the readme file.
        x_tilde = x * np.cos(alpha_cavity) - z * np.sin(alpha_cavity)
        z_tilde = x * np.sin(alpha_cavity) + z * np.cos(alpha_cavity)

        if self.ring_cavity:
            standing_wave = False
            forward_propagation_l_1 = self.l_1 < self.l_2  # if l_1 is the shorter wavelength then it propagates forward
        else:
            standing_wave = True
            forward_propagation_l_1 = None
        A_1 = gaussian_beam(x=x_tilde, y=y, z=z_tilde, E=self.E_1, lambda_laser=self.l_1, NA=self.NA_1, t=t,
                            mode="potential", standing_wave=standing_wave, forward_propagation=forward_propagation_l_1)
        A_2 = gaussian_beam(x=x_tilde, y=y, z=z_tilde, E=self.E_2, lambda_laser=self.l_2, NA=self.NA_2, t=t,
                            mode="potential", standing_wave=standing_wave,
                            forward_propagation=not forward_propagation_l_1)

        # This is not the electric potential, but rather only the amplitude factor that is shared among the different
        # components of the electromagnetic potential. each component is multiplied by the corresponding trigonometric
        # functions, depending on the polarization, and the tilt of the cavity.
        A_scalar = A_1 + A_2

        # This is the rotated field vector: the vector is achieved by starting with a polarized light in the z axis,
        # then rotating it in the y-z plane by theta_polarization (from z towards y) and then rotating it in the
        # z-x plane by alpha_cavity (from x towards z).
        # (0, 0, 1) -> (0, sin(t), cos(t)) -> (sin(a)cos(t), sin(t), cos(a)cos(t))
        if not return_vector:
            return A_scalar
        else:
            A_vector = np.stack((-A_scalar * np.cos(self.theta_polarization) * np.sin(alpha_cavity),
                                 A_scalar * np.sin(self.theta_polarization),
                                 A_scalar * np.cos(self.theta_polarization) * np.cos(alpha_cavity)),
                                axis=-1)
            return A_vector

    def grad_G(self, X, Y, Z, T, beta_electron, A_z=None, save_to_file: bool = False):

        # You might ask yourself - we already have a lattice full of G values - why not subtract the lattice from
        # itself to get the gradient? The answer is that the lattice is too sparse to get a good gradient.
        # It could have been possible in the z axis where we must have dense z values for the later integral, but in the
        # z axis we also have the time component increasing with Z, so subtracting the lattice from itself will give
        # dG=G(z+dz, t(z+dz))-G(z, t(z)), while we want dG=G(z+dz, t(z))-G(z, t(z))-G

        dr = self.min_l / 1000  # ARBITRARY I checked manually, and at around 400 the value of
        # the derivative stabilizes, so 1000 is a safe margin. Anyway I don't think we are not close to precision
        # error problems.

        z_component_factor = np.cos(self.beta_electron2alpha_cavity(beta_electron)) * np.cos(self.theta_polarization)

        if A_z is None:
            A_z = self.rotated_gaussian_beam_A(X, Y, Z, T, beta_electron, return_vector=False) * z_component_factor

        # The values of A, but shifted a bit, for the gradient calculation:
        A_z_dX = self.rotated_gaussian_beam_A(X + dr, Y, Z, T, beta_electron, return_vector=False) * z_component_factor
        A_z_dY = self.rotated_gaussian_beam_A(X, Y + dr, Z, T, beta_electron, return_vector=False) * z_component_factor
        A_z_dZ = self.rotated_gaussian_beam_A(X, Y, Z, T - dr / (beta_electron * C_LIGHT), beta_electron,
                                              return_vector=False) * z_component_factor  # Explanation in eq:e_26 in
        # my readme file

        G = self.G_gauge(A_z, Z)
        G_dX = self.G_gauge(A_z_dX, Z)
        G_dY = self.G_gauge(A_z_dY, Z)
        G_dZ = self.G_gauge(A_z_dZ, Z) + A_z_dZ * dr  # Explanation in eq:e_26 in my readme file

        grad_G = np.stack([(G_dX - G) / dr, (G_dY - G) / dr, (G_dZ - G) / dr], axis=-1)

        if save_to_file:
            np.save("Data Arrays\\Debugging Arrays\\G.npy", G)
            np.save("Data Arrays\\Debugging Arrays\\grad_G.npy", grad_G)

        return grad_G

    @staticmethod
    def G_gauge(A_shifted: np.ndarray, Z: np.ndarray) -> np.ndarray:
        G_gauge_values = integrate.cumtrapz(A_shifted, x=Z, axis=2, initial=0)
        return G_gauge_values  # integral over z

    def setup_to_path(self, input_wave: WaveFunction) -> str:
        # This function is used to generate a unique name for each setup, such that we can load previous results if they
        # exist, and not calculate everything again in each run.
        path = f'Data Arrays\\Phase Masks\\2f_l1{self.l_1 * 1e9:.4g}_l2{self.l_2 * 1e9:.4g}_' \
               f'E1{self.E_1:.3g}_E2{self.E_2:.3g}_NA1{self.NA_1 * 100:.4g}_NA2{self.NA_2 * 100:.4g}_' \
               f'alpha{self.beta_electron2alpha_cavity(E2beta(input_wave.E0)) / 2 * np.pi * 360:.0f}_' \
               f'theta{self.theta_polarization * 100:.4g}_E{input_wave.E0:.2g}_ring{self.ring_cavity}_' \
               f'Nx{input_wave.coordinates.x_axis.size}_Ny{input_wave.coordinates.y_axis.size}_Nt{self.n_t}_' \
               f'DX{input_wave.coordinates.limits[1]:.4g}_DY{input_wave.coordinates.limits[3]:.4g}.npy'

        return path

    def generate_z_vector(self, x, beta_electron):
        # This function generates a linearly spaced z values in units of the spot size with a number of points that
        # is required for accurate integration of A in the widest integration range (between -5 and 5 w(x) for the
        # maximal w(x))
        alpha_cavity = self.beta_electron2alpha_cavity(beta_electron)
        integral_limit_in_spot_size_units = 5
        max_z_integration_interval = w_x_gaussian(w_0=NA2w0(self.NA_max, self.max_l),
                                                  x=np.max(np.abs(x)) / np.cos(alpha_cavity),
                                                  l_laser=self.max_l) / np.cos(alpha_cavity) * \
                                     (2 * integral_limit_in_spot_size_units)
        required_n_z_for_integration_over_maximal_interval = int(
            max_z_integration_interval / (1 / (1 + 1 / beta_electron) * self.min_l))
        # The denominator is chosen such that the G_gauge function is accurate. In general the spacing dz should be
        # around the effective wavelength that is seen by the passing electron. The effective wavelength is (
        # 1+1/beta) * l. This is because: cos(kx) * cos(wt) -> cos(kx) * cos(w * (x/(beta*c))) = cos(kx) * cos(
        # k/beta * x) = 1/2 * ( cos((1+1/beta)kx) + cos((1-1/beta)kx)

        z = np.linspace(-integral_limit_in_spot_size_units,
                        integral_limit_in_spot_size_units,
                        required_n_z_for_integration_over_maximal_interval)
        return z


if __name__ == '__main__':
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
    # cavity_2f = Cavity2FrequenciesAnalyticalPropagator(l_1=1064e-9, l_2=532e-9, E_1=8e8, E_2=-1, NA_1=0.05)
    cavity_2f = Cavity2FrequenciesNumericalPropagator(l_1=1064e-9, l_2=532e-9, E_1=8e8, E_2=-1, NA_1=0.05)
    cavity_1f = Cavity2FrequenciesAnalyticalPropagator(l_1=1064e-9, l_2=None, E_1=5.06e-7, E_2=None, NA_1=0.05)
    aberration_propagator = AberrationsPropagator(Cs=0, defocus=0, atigmatism_parameter=0, astigmatism_orientation=0)

    M_2f = Microscope([dummy_sample, first_lens, cavity_2f, second_lens, aberration_propagator])
    M_2f.propagate(first_wave)
    M_2f.plot_step(aberration_propagator)
