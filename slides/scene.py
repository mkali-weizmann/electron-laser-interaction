from manim import *
from manim_slides import Slide
import numpy as np
from scipy import special
from typing import Union, Optional, Callable

# manim -pql slides/scene.py Microscope
# manim-slides convert Microscope slides/presentation.html


TRACKER_TIME = ValueTracker(0)
TRACKER_SCANNING_SAMPLE = ValueTracker(0)
TRACKER_SCANNING_CAMERA = ValueTracker(0)
TRACKER_TIME_LASER = ValueTracker(0)
TRACKER_PHASE_MODULATION = ValueTracker(0)
MICROSCOPE_Y = -0.75
BEGINNING = - 7
FIRST_LENS_X = -2.5
POSITION_LENS_1 = np.array([FIRST_LENS_X, MICROSCOPE_Y, 0])
SECOND_LENS_X = 4
POSITION_LENS_2 = np.array([SECOND_LENS_X, MICROSCOPE_Y, 0])
INITIAL_VERTICAL_LENGTH = 1
FINAL_VERTICAL_LENGTH = 2
POSITION_CAMERA = np.array([SECOND_LENS_X + 2, MICROSCOPE_Y, 0])
END = SECOND_LENS_X + 2
POSITION_WAIST = np.array([(2 * FIRST_LENS_X + SECOND_LENS_X) / 3, MICROSCOPE_Y, 0])
W_0 = 0.14
X_R = (FIRST_LENS_X - SECOND_LENS_X) / 9
W_0_LASER = 0.14
X_R_LASER = 0.5
POSITION_SAMPLE = np.array([-5, MICROSCOPE_Y, 0])
HEIGHT_SAMPLE = INITIAL_VERTICAL_LENGTH
WIDTH_SAMPLE = HEIGHT_SAMPLE * (2 / 3)
HEIGHT_CAMERA = FINAL_VERTICAL_LENGTH
WIDTH_CAMERA = WIDTH_SAMPLE
POSITION_AXES_1 = np.array([-2.2, 2.5, 0])
POSITION_AXES_2 = np.array([2.5, 2.5, 0])
HEIGHT_SCANNING_AXES = 2.5
WIDTH_SCANNING_AXES = 2.5
WAVELENGTH = 0.5
WAVELENGTH_LASER = 0.3
LENGTH_LASER_BEAM = WAVELENGTH_LASER * 6
AXES_RANGE = 1
AMPLITUDE_SIZE = 0.8
PHASE_SHIFT_AMPLITUDE = 0.2
COLOR_INTENSITIES = GREEN
COLOR_UNPERTURBED_AMPLITUDE = GOLD_B
COLOR_PERTURBED_AMPLITUDE = BLUE
COLOR_PHASE_SHIFT_AMPLITUDE = PURPLE_B
COLOR_SCANNING_DOT = GREEN
ZOOM_RATIO = 0.3
POSITION_TITLE = np.array([-6, 2.5, 0])
VELOCITIES_RATIO = WAVELENGTH_LASER / WAVELENGTH
TITLE_COUNTER = 0


def matrix_rgb(mat: np.ndarray):
    return (rgb_to_color(mat[i, :]) for i in range(mat.shape[0]))


def noise_function_1(x):
    return 0.1 * np.sin(3 * x) + 0.2 * np.sin(2 * x)


def noise_function_2(x):
    return 0.1 * np.sin(2 * x) - 0.2 * np.sin(3 * x)

def gaussian_beam_R_x(x, x_R):
    if x == 0:
        return 1000000
    else:
        return x * (1 + x_R ** 2 / x ** 2)


def gaussian_beam_w_x(x, w_0, x_R):
    return w_0 * np.sqrt(1 + (x / x_R) ** 2)


def generate_waves(start_point: Union[np.ndarray, list],
                   end_point: Union[np.ndarray, list],
                   wavelength, width: float,
                   tracker: ValueTracker):
    if isinstance(start_point, list):
        start_point = np.array(start_point, dtype='float64')
    if isinstance(end_point, list):
        end_point = np.array(end_point, dtype='float64')
    path_direction = end_point - start_point
    path_length = np.linalg.norm(path_direction)
    path_direction = path_direction / path_length
    orthogonal_direction = np.cross(path_direction, [0, 0, 1])
    orthogonal_direction[2] = 0  # make sure it's in the xy plane
    orthogonal_direction = orthogonal_direction / np.linalg.norm(orthogonal_direction)
    n_waves = int(path_length // wavelength)
    waves = [Line(
        start=start_point + np.mod(i * wavelength, path_length) * path_direction + width / 2 * orthogonal_direction,
        end=start_point + np.mod(i * wavelength, path_length) * path_direction - width / 2 * orthogonal_direction)
        for i in range(n_waves)]
    waves = VGroup(*waves)
    for i, wave in enumerate(waves):
        wave.add_updater(
            lambda m, i=i: m.move_to(start_point +
                                     np.mod(i * wavelength + tracker.get_value(), path_length) * path_direction)
            .set_opacity(there_and_back_with_pause(
                np.mod(tracker.get_value() + i * wavelength, path_length) / path_length))
        )
    return waves


def dummy_attenuating_function(x: float):
    return np.max([0, 1 - np.abs(x) / 2])


def points_generatoer_gaussian_beam(x: float,
                                    w_0: float,
                                    x_R: float,
                                    center: np.ndarray,
                                    k_vec: np.ndarray,
                                    noise_generator: Callable = lambda t: 0):
    w_x = w_0 * np.sqrt(1 + (x / x_R) ** 2)
    R_x_inverse = x / (x ** 2 + x_R ** 2)
    R_x_inverse = np.tanh(R_x_inverse / 2)  # Renormalizing for the esthetic of the visualization
    transverse_direction = np.cross(k_vec, [0, 0, 1])
    array = np.array([center + x * k_vec - w_x * transverse_direction,
                      center + x * k_vec - w_x * transverse_direction / 2 + R_x_inverse * w_x * k_vec,
                      center + x * k_vec + w_x * transverse_direction / 2 + R_x_inverse * w_x * k_vec,
                      center + x * k_vec + w_x * transverse_direction]) + noise_generator(x) * np.sqrt(
        np.abs((w_x - w_0)) / w_0)

    return array


def points_generator_plane_wave(x: float,
                                center: np.ndarray,
                                k_vec: np.ndarray,
                                width: float,
                                noise_generator: Callable = lambda t: 0):
    transverse_direction = np.cross(k_vec, [0, 0, 1])
    array = np.array([center + x * k_vec - transverse_direction * width / 2,
                      center + x * k_vec - (1 / 3) * transverse_direction * width / 2,
                      center + x * k_vec + (1 / 3) * transverse_direction * width / 2,
                      center + x * k_vec + transverse_direction * width / 2]) + noise_generator(x) * width
    return array


def generate_bazier_wavefront(points: np.ndarray,
                              colors: Optional[Union[np.ndarray, str]] = None,
                              opacities: Optional[np.ndarray] = None, **kwargs):
    if isinstance(colors, str):
        colors = color_to_rgb(colors)
    if colors is not None and opacities is not None:
        colors = (colors.T * opacities).T
    elif colors is None and opacities is not None:
        colors = np.ones((opacities.size, 3))
        colors = (colors.T * opacities).T
    elif colors is None and opacities is None:
        colors = np.ones((1, 3))
    elif colors is not None and opacities is None:
        if colors.ndim == 1:
            colors = colors[np.newaxis, :]
    else:
        raise ValueError(f'Invalid input for:\n{colors=}\nand:\n{opacities=}')

    if points.shape[0] == 2:
        diff = points[1, :] - points[0, :]
        points = np.array([points[0],
                           points[0] + diff / 3,
                           points[0] + diff / 3 * 2,
                           points[1]])

    curve = CubicBezier(points[0, :],
                        points[1, :],
                        points[2, :],
                        points[3, :], **kwargs)
    curve.set_stroke(matrix_rgb(colors)).set_fill(None, opacity=0.0)
    return curve


def generate_bazier_wavefronts(points_generator: Callable,
                               tracker: ValueTracker,
                               wavelength: float,
                               start_parameter: float,
                               end_parameter: float,
                               colors_generator: Callable = lambda t: None,
                               opacities_generator: Callable = lambda t: None,
                               pause_ratio: float = 1.0 / 3, **kwargs):
    length = end_parameter - start_parameter
    n = int(length // wavelength)
    generators = [
        lambda i=i: generate_bazier_wavefront(
            points_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter),
            colors_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter),
            opacities_generator((np.mod(tracker.get_value(), 1) + i) * wavelength + start_parameter), **kwargs)
        for i in range(n)]
    waves = [always_redraw(generators[i]) for i in range(n)]
    for i, wave in enumerate(waves):
        wave.add_updater(
            lambda m, i=i: m.set_stroke(opacity=there_and_back_with_pause((np.mod(tracker.get_value(), 1) + i) / n,
                                                                          pause_ratio)))
    waves = VGroup(*waves)
    return waves


def generate_wavefronts_start_to_end_gaussian(start_point: Union[np.ndarray, list],
                                              end_point: Union[np.ndarray, list],
                                              tracker: ValueTracker,
                                              wavelength: float,
                                              x_R,
                                              w_0,
                                              center=None,
                                              colors_generator: Callable = lambda t: None,
                                              opacities_generator: Callable = lambda t: None,
                                              noise_generator: Callable = lambda t: 0, **kwargs):
    if isinstance(start_point, list):
        start_point = np.array(start_point)
    if isinstance(end_point, list):
        end_point = np.array(end_point)
    if isinstance(center, list):
        center = np.array(center)
    path = end_point - start_point
    length = np.linalg.norm(path)
    k_vec = path / length
    if center is None:
        center = start_point + length / 2 * k_vec

    start_parameter = k_vec @ (start_point - center)
    end_parameter = k_vec @ (end_point - center)
    points_generator = lambda t: points_generatoer_gaussian_beam(x=t, w_0=w_0, x_R=x_R, center=center, k_vec=k_vec,
                                                                 noise_generator=noise_generator)
    waves = generate_bazier_wavefronts(points_generator=points_generator, tracker=tracker,
                                       wavelength=wavelength, start_parameter=start_parameter,
                                       end_parameter=end_parameter,
                                       colors_generator=colors_generator, opacities_generator=opacities_generator,
                                       pause_ratio=1 / 4, **kwargs)
    return waves


def generate_wavefronts_start_to_end_flat(start_point: Union[np.ndarray, list],
                                          end_point: Union[np.ndarray, list],
                                          tracker: ValueTracker,
                                          wavelength: float,
                                          colors_generator: Callable = lambda t: None,
                                          opacities_generator: Callable = lambda t: None,
                                          noise_generator: Callable = lambda t: 0,
                                          width=1, **kwargs):
    if isinstance(start_point, list):
        start_point = np.array(start_point)
    if isinstance(end_point, list):
        end_point = np.array(end_point)
    path = end_point - start_point
    length = np.linalg.norm(path)
    if width is None:
        width = length / 4
    k_vec = path / length
    center = start_point
    start_parameter = 0
    end_parameter = length
    points_generator = lambda x: points_generator_plane_wave(x=x, center=center, k_vec=k_vec, width=width,
                                                             noise_generator=noise_generator)
    waves = generate_bazier_wavefronts(points_generator, tracker, wavelength=wavelength,
                                       start_parameter=start_parameter, end_parameter=end_parameter,
                                       colors_generator=colors_generator, opacities_generator=opacities_generator,
                                       **kwargs)
    return waves


def generate_scanning_axes(dot_start_point: Union[np.ndarray, list],
                           dot_end_point: Union[np.ndarray, list],
                           axes_position: Union[np.ndarray, list],
                           tracker: ValueTracker,
                           function_to_plot: Callable,
                           axis_x_label: str,
                           axis_y_label: str):
    ax = Axes(x_range=[0, AXES_RANGE, AXES_RANGE / 4], y_range=[0, AXES_RANGE, AXES_RANGE / 4],
              x_length=WIDTH_SCANNING_AXES, y_length=HEIGHT_SCANNING_AXES, tips=False).move_to(axes_position)

    labels = ax.get_axis_labels(
        Tex(axis_x_label).scale(0.5), Tex(axis_y_label).scale(0.5)
    )

    def scanning_dot_generator():
        scanning_dot = Dot(point=dot_start_point + tracker.get_value() * (dot_end_point - dot_start_point),
                           color=COLOR_SCANNING_DOT)
        return scanning_dot

    scanning_dot = always_redraw(scanning_dot_generator)

    def scanning_dot_x_axis_generator():
        scanning_dot_x_axis_start_point = ax.c2p(0, 0)
        scanning_dot_x_axis_end_point = ax.c2p(AXES_RANGE, 0)
        scanning_dot_x_axis = Dot(point=scanning_dot_x_axis_start_point +
                                        tracker.get_value() * (scanning_dot_x_axis_end_point -
                                                               scanning_dot_x_axis_start_point),
                                  color=COLOR_SCANNING_DOT)
        return scanning_dot_x_axis

    scanning_dot_x_axis = always_redraw(scanning_dot_x_axis_generator)

    if function_to_plot is not None:
        amplitude_graph = ax.plot(function_to_plot, color=COLOR_INTENSITIES)
        return ax, labels, scanning_dot, scanning_dot_x_axis, amplitude_graph
    else:
        return ax, labels, scanning_dot, scanning_dot_x_axis


class Microscope(Slide, MovingCameraScene):  # , ZoomedScene
    def construct(self):
        self.wait(1)
        self.next_slide()
        title_0 = Tex("k", color=BLACK).to_corner(UL).scale(0.5)
        title_1 = Tex("1) Microscope").scale(0.5).next_to(title_0, DOWN).align_to(title_0, LEFT)
        title_2 = Tex("2) Phase Object").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        title_3 = Tex("3) Waves Decomposition").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        title_4 = Tex("4) Phase Mask").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        title_5 = Tex("5) Phase Mask + Attenuation").scale(0.5).next_to(title_1, DOWN).align_to(title_0, LEFT)
        titles_square = Rectangle(height=title_1.height + 0.1,
                                  width=title_1.width + 0.1,
                                  stroke_width=2).move_to(title_1.get_center()).set_fill(opacity=0)
        y_0 = title_0.get_center()[1]
        y_1 = title_1.get_center()[1]
        dy = 0.47
        bad_title = Tex(
            "Transmission Electron Microscope image enhancement\nusing second order free electron-photon interaction",
            color=WHITE).scale(0.75)
        good_title = Tex("Shooting laser on electrons\nmake images good", color=WHITE).scale(0.75)
        self.play(FadeIn(bad_title, shift=DOWN))
        self.next_slide()
        self.play(FadeOut(bad_title, shift=DOWN), FadeIn(good_title, shift=DOWN))
        self.next_slide()
        self.play(FadeOut(good_title, shift=DOWN))
        incoming_waves = generate_wavefronts_start_to_end_flat(start_point=[BEGINNING, MICROSCOPE_Y, 0],
                                                               end_point=POSITION_SAMPLE,
                                                               tracker=TRACKER_TIME,
                                                               wavelength=WAVELENGTH,
                                                               width=HEIGHT_SAMPLE
                                                               )

        sample_outgoing_waves = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                      end_point=POSITION_LENS_1,
                                                                      wavelength=WAVELENGTH,
                                                                      width=HEIGHT_SAMPLE,
                                                                      tracker=TRACKER_TIME)

        second_lens_outgoing_waves = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                           end_point=POSITION_CAMERA,
                                                                           wavelength=WAVELENGTH,
                                                                           width=HEIGHT_CAMERA,
                                                                           tracker=TRACKER_TIME)
        lens_1 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=BLUE).move_to(POSITION_LENS_1)
        lens_2 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=BLUE).move_to(POSITION_LENS_2)
        sample = Rectangle(height=HEIGHT_SAMPLE, width=WIDTH_SAMPLE, color=BLUE).move_to(POSITION_SAMPLE)
        camera = Rectangle(height=HEIGHT_CAMERA, width=WIDTH_CAMERA, color=GRAY, fill_color=GRAY_A,
                           fill_opacity=0.3).move_to(POSITION_CAMERA)

        gaussian_beam_waves = generate_wavefronts_start_to_end_gaussian(start_point=POSITION_LENS_1,
                                                                        end_point=POSITION_LENS_2,
                                                                        tracker=TRACKER_TIME,
                                                                        wavelength=WAVELENGTH,
                                                                        x_R=X_R,
                                                                        w_0=W_0,
                                                                        center=POSITION_WAIST)
        microscope_VGroup = VGroup(incoming_waves, sample, lens_1, sample_outgoing_waves, lens_2,
                                   gaussian_beam_waves,
                                   second_lens_outgoing_waves, camera)
        self.updated_object_animation([microscope_VGroup, title_0, title_1, title_2, titles_square], FadeIn)
        self.next_slide()
        self.start_loop()
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.end_loop()
        image = ImageMobject(np.uint8([[2, 100], [40, 5], [170, 50]])).move_to(POSITION_SAMPLE)
        image.width = WIDTH_SAMPLE
        image.set_z_index(sample.get_z_index() - 1)
        self.play(FadeIn(image))
        self.next_slide()
        sample_outgoing_waves_opacities = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                                end_point=POSITION_LENS_1,
                                                                                wavelength=WAVELENGTH,
                                                                                width=HEIGHT_SAMPLE,
                                                                                tracker=TRACKER_TIME,
                                                                                opacities_generator=lambda t: np.array(
                                                                                    [1, np.cos(2 * t) ** 2,
                                                                                     np.sin(2 * t) ** 2,
                                                                                     1 - 0.1 * np.cos(t) ** 2]))
        second_lens_outgoing_waves_opacities = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                     end_point=POSITION_CAMERA,
                                                                                     wavelength=WAVELENGTH,
                                                                                     width=HEIGHT_CAMERA,
                                                                                     tracker=TRACKER_TIME,
                                                                                     opacities_generator=lambda
                                                                                         t: np.array(
                                                                                         [1 - 0.1 * np.cos(
                                                                                             (2 - t)) ** 2,
                                                                                          np.sin(2 * (2 - t)) ** 2,
                                                                                          np.cos(2 * (2 - t)) ** 2,
                                                                                          1]))
        gaussian_beam_waves_opacities = generate_wavefronts_start_to_end_gaussian(start_point=POSITION_LENS_1,
                                                                                  end_point=POSITION_LENS_2,
                                                                                  tracker=TRACKER_TIME,
                                                                                  wavelength=WAVELENGTH,
                                                                                  x_R=X_R,
                                                                                  w_0=W_0,
                                                                                  center=POSITION_WAIST,
                                                                                  opacities_generator=lambda
                                                                                      t: np.array([0,
                                                                                                   0.5 + 0.5 * np.cos(
                                                                                                       6 * t + 1) ** 2,
                                                                                                   np.cos(8 * t) ** 2,
                                                                                                   0.2 + 0.8 * np.cos(
                                                                                                       5 * t - 1) ** 2,
                                                                                                   0]))
        self.play(sample_outgoing_waves.animate.become(sample_outgoing_waves_opacities),
                  second_lens_outgoing_waves.animate.become(second_lens_outgoing_waves_opacities),
                  gaussian_beam_waves.animate.become(gaussian_beam_waves_opacities), run_time=2, rate_func=linear)
        self.remove(sample_outgoing_waves, second_lens_outgoing_waves, gaussian_beam_waves)
        self.add(sample_outgoing_waves_opacities, second_lens_outgoing_waves_opacities, gaussian_beam_waves_opacities)
        self.next_slide()
        self.start_loop()
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.end_loop()

        ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1 = generate_scanning_axes(
            dot_start_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP,
            dot_end_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT - HEIGHT_SAMPLE / 2 * UP,
            axes_position=POSITION_AXES_1,
            tracker=TRACKER_SCANNING_SAMPLE,
            function_to_plot=lambda x: 1 - 0.2 * np.exp(-(6 * (x - 0.7)) ** 2),
            axis_x_label="Position",
            axis_y_label="Intensity")

        self.play(Create(ax_1), Write(labels_1), run_time=2)
        self.play(Create(scanning_dot_1), Create(scanning_dot_x_axis_1))
        self.play(TRACKER_SCANNING_SAMPLE.animate.set_value(1), Create(amplitude_graph_1), run_time=2)
        self.next_slide()

        ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2 = generate_scanning_axes(
            dot_start_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP,
            dot_end_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT + HEIGHT_CAMERA / 2 * UP,
            axes_position=POSITION_AXES_2,
            tracker=TRACKER_SCANNING_CAMERA,
            function_to_plot=lambda x: 1 - 0.2 * np.exp(-(6 * (x - 0.7)) ** 2),
            axis_x_label="Position",
            axis_y_label="Intensity")

        camera_scanner_group = VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2)
        self.play(Create(ax_2), Write(labels_2), run_time=2)
        self.next_slide()
        self.play(Create(scanning_dot_2), Create(scanning_dot_x_axis_2))
        self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), Create(amplitude_graph_2), run_time=2)
        self.next_slide()
        self.play(FadeOut(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2,
                                 ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1)))

        TRACKER_SCANNING_SAMPLE.set_value(0)

        self.play(FadeOut(image))
        phase_image = ImageMobject(np.uint8([[[250, 0, 0], [100, 0, 0]],
                                             [[20, 0, 0], [100, 0, 0]],
                                             [[0, 0, 0], [220, 0, 0]]])).move_to(POSITION_SAMPLE)
        phase_image.width = WIDTH_SAMPLE
        phase_image.set_z_index(sample.get_z_index() - 1)
        self.play(FadeOut(title_0, shift=dy * UP),
                  title_1.animate.move_to([title_1.get_center()[0], y_0, 0]),
                  title_2.animate.move_to([title_2.get_center()[0], y_1, 0]),
                  FadeIn(title_3, shift=dy * UP),
                  titles_square.animate.set_width(title_2.width + 0.1).move_to([title_2.get_center()[0], y_1, 0]))
        self.next_slide()
        self.play(FadeIn(phase_image))

        sample_outgoing_waves_moises = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                             end_point=POSITION_LENS_1,
                                                                             wavelength=WAVELENGTH,
                                                                             width=HEIGHT_SAMPLE,
                                                                             tracker=TRACKER_TIME,
                                                                             noise_generator=lambda t: np.array(
                                                                                 [[0, 0, 0],
                                                                                  [0.1 * np.sin(2 * np.pi * t), 0, 0],
                                                                                  [0.05 * np.cos(2 * np.pi * t), 0, 0],
                                                                                  [0.1 * np.cos(t), 0, 0]]))
        second_lens_outgoing_waves_moises = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                  end_point=POSITION_CAMERA,
                                                                                  wavelength=WAVELENGTH,
                                                                                  width=HEIGHT_CAMERA,
                                                                                  tracker=TRACKER_TIME,
                                                                                  noise_generator=lambda
                                                                                      t: np.array(
                                                                                      [[0.1 * np.cos(t), 0, 0],
                                                                                       [0.05 * np.cos(2 * np.pi * t), 0,
                                                                                        0],
                                                                                       [0.1 * np.sin(2 * np.pi * t), 0,
                                                                                        0],
                                                                                       [0, 0, 0]]))
        gaussian_beam_waves_moises = generate_wavefronts_start_to_end_gaussian(start_point=POSITION_LENS_1,
                                                                               end_point=POSITION_LENS_2,
                                                                               tracker=TRACKER_TIME,
                                                                               wavelength=WAVELENGTH,
                                                                               x_R=X_R,
                                                                               w_0=W_0,
                                                                               center=POSITION_WAIST,
                                                                               noise_generator=lambda
                                                                                   t: np.array(
                                                                                   [[0.08 * np.sin(5 * t - 1), 0, 0],
                                                                                    [0.09 * np.sin(3 * t + 1), 0, 0],
                                                                                    [0.07 * np.cos(8 * t), 0, 0],
                                                                                    [0, 0, 0]]))


        self.play(sample_outgoing_waves_opacities.animate.become(sample_outgoing_waves_moises),
                  second_lens_outgoing_waves_opacities.animate.become(second_lens_outgoing_waves_moises),
                  gaussian_beam_waves_opacities.animate.become(gaussian_beam_waves_moises), run_time=2,
                  rate_func=linear)
        self.remove(sample_outgoing_waves_opacities, second_lens_outgoing_waves_opacities,
                    gaussian_beam_waves_opacities)
        self.add(sample_outgoing_waves_moises,
                 second_lens_outgoing_waves_moises,
                 gaussian_beam_waves_moises)
        self.next_slide()
        self.start_loop()
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.end_loop()
        microscope_VGroup.remove(second_lens_outgoing_waves, sample_outgoing_waves, gaussian_beam_waves)
        microscope_VGroup.add(second_lens_outgoing_waves_moises,
                              sample_outgoing_waves_moises,
                              gaussian_beam_waves_moises)

        ax_complex_amplitude = Axes(x_range=[-AXES_RANGE, AXES_RANGE, 0.25],
                                    y_range=[-AXES_RANGE, AXES_RANGE, 0.25],
                                    x_length=WIDTH_SCANNING_AXES,
                                    y_length=HEIGHT_SCANNING_AXES,
                                    tips=False).move_to(POSITION_AXES_1)
        labels_complex_amplitude = ax_complex_amplitude.get_axis_labels(
            Tex(r'$\text{Re}\left(\psi\right)$').scale(0.3), Tex(r'$\text{Im}\left(\psi\right)$').scale(0.3)
        )

        def circ_complex_amplitude_generator():
            return Circle(
                radius=np.linalg.norm(ax_complex_amplitude.c2p((AMPLITUDE_SIZE, 0)) - ax_complex_amplitude.c2p((0, 0))),
                color=WHITE, stroke_opacity=0.3).move_to(ax_complex_amplitude.c2p(0, 0))

        circ_complex_amplitude = always_redraw(circ_complex_amplitude_generator)

        def arrow_complex_amplitude_generator():
            arrow_complex_amplitude = Line(
                start=ax_complex_amplitude.c2p(0, 0),
                end=ax_complex_amplitude.c2p(AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                    6 * PI * TRACKER_SCANNING_SAMPLE.get_value())),
                                             AMPLITUDE_SIZE * np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                                                 6 * PI * TRACKER_SCANNING_SAMPLE.get_value()))),
                color=BLUE_B, z_index=ax_complex_amplitude.z_index + 1)
            return arrow_complex_amplitude

        line_complex_amplitude = always_redraw(arrow_complex_amplitude_generator)
        # The dot can not have an always_redraw updater because it is going to change color.
        dot_complex_amplitude = Dot(point=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0), color=COLOR_SCANNING_DOT)
        dot_complex_amplitude.add_updater(lambda m: m.move_to(
            ax_complex_amplitude.c2p(
                AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(6 * PI * TRACKER_SCANNING_SAMPLE.get_value())),
                AMPLITUDE_SIZE * np.sin(
                    PHASE_SHIFT_AMPLITUDE * np.sin(6 * PI * TRACKER_SCANNING_SAMPLE.get_value())))).set_z_index(
            line_complex_amplitude.z_index + 1))
        self.play(Create(ax_complex_amplitude), Create(labels_complex_amplitude), Create(circ_complex_amplitude))
        # Here it is separated because the dot has to be created after the axes, or it glitches..
        scanning_dot_1.move_to(POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP)
        self.play(Create(line_complex_amplitude), Create(dot_complex_amplitude), Create(scanning_dot_1))
        complex_amplitude_ax_group = VGroup(ax_complex_amplitude, labels_complex_amplitude)
        complex_amplitude_graph_group = VGroup(complex_amplitude_ax_group, circ_complex_amplitude,
                                               line_complex_amplitude, dot_complex_amplitude)
        self.next_slide()
        self.play(TRACKER_SCANNING_SAMPLE.animate.set_value(1), run_time=4)
        TRACKER_SCANNING_CAMERA.set_value(0)
        self.next_slide()
        self.play(Create(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2)))
        constant_intensity_function = ax_2.plot(lambda x: 0.3, color=COLOR_INTENSITIES)
        camera_scanner_group.add(constant_intensity_function)
        camera_scanner_group.remove(amplitude_graph_2)
        self.next_slide()
        self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), Create(constant_intensity_function), run_time=2)
        self.next_slide()
        self.updated_object_animation([microscope_VGroup, phase_image, camera_scanner_group, scanning_dot_1],
                                      FadeOut)
        self.play(complex_amplitude_ax_group.animate.move_to([0, 0, 0]).scale(scale_factor=2.5),
                  dot_complex_amplitude.animate.set_fill(COLOR_SCANNING_DOT),
                  FadeOut(title_1, shift=dy * UP),
                  title_2.animate.move_to([title_2.get_center()[0], y_0, 0]),
                  title_3.animate.move_to([title_3.get_center()[0], y_1, 0]),
                  FadeIn(title_4, shift=dy * UP),
                  titles_square.animate.set_width(title_3.width + 0.1).move_to([title_3.get_center()[0], y_1, 0])
                  )  # ,
        self.next_slide()

        def line_amplitude_perturbation_generator():
            line_amplitude_perturbation = Line(
                start=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0),
                end=ax_complex_amplitude.c2p(
                    AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                        6 * PI * TRACKER_SCANNING_SAMPLE.get_value())),
                    AMPLITUDE_SIZE * np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                        6 * PI * TRACKER_SCANNING_SAMPLE.get_value()))),
                color=COLOR_PERTURBED_AMPLITUDE)
            return line_amplitude_perturbation

        line_amplitude_perturbation = always_redraw(line_amplitude_perturbation_generator)
        line_complex_amplitude.clear_updaters()
        line_complex_amplitude.add_updater(lambda m: m.become(Line(start=ax_complex_amplitude.c2p(0, 0),
                                                                   end=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0),
                                                                   color=COLOR_UNPERTURBED_AMPLITUDE,
                                                                   z_index=ax_complex_amplitude.z_index + 1)),
                                           )
        tex = MathTex(r"\psi_{\text{out}}\left(x\right)="
                      r"\psi_{\text{unperturbed}}+i\psi_{\text{perturbation}}\left(x\right)").next_to(
            ax_complex_amplitude.get_bottom(), RIGHT + UP).scale(0.6)
        tex[0][8:20].set_color(COLOR_UNPERTURBED_AMPLITUDE)
        tex[0][22:].set_color(COLOR_PERTURBED_AMPLITUDE)
        self.play(FadeIn(tex), FadeIn(line_amplitude_perturbation))
        self.play(TRACKER_SCANNING_SAMPLE.animate.set_value(2), run_time=4)
        self.next_slide()
        self.play(complex_amplitude_ax_group.animate.move_to(POSITION_AXES_1).scale(scale_factor=1 / 2.5),
                  FadeOut(tex))
        line_complex_amplitude.clear_updaters()
        self.play(FadeIn(microscope_VGroup), FadeIn(phase_image), FadeOut(complex_amplitude_graph_group))

        sample_outgoing_unperturbed_waves = generate_wavefronts_start_to_end_flat(start_point=POSITION_SAMPLE,
                                                                                  end_point=POSITION_LENS_1,
                                                                                  wavelength=WAVELENGTH,
                                                                                  width=HEIGHT_SAMPLE,
                                                                                  tracker=TRACKER_TIME,
                                                                                  colors_generator=lambda
                                                                                      t: COLOR_UNPERTURBED_AMPLITUDE)
        sample_outgoing_perturbed_waves_1 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_SAMPLE + 0.2 * UP,
            end_point=POSITION_LENS_1 - 0.2 * UP,
            wavelength=WAVELENGTH,
            width=HEIGHT_SAMPLE,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        sample_outgoing_perturbed_waves_2 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_SAMPLE - 0.2 * UP,
            end_point=POSITION_LENS_1 + 0.2 * UP,
            wavelength=WAVELENGTH,
            width=HEIGHT_SAMPLE,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

        gaussian_beam_waves_unperturbed = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_LENS_1,
            end_point=POSITION_LENS_2,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST,
            colors_generator=lambda
                t: COLOR_UNPERTURBED_AMPLITUDE)
        gaussian_beam_waves_perturbed_1 = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_LENS_1 + W_0 * UP,
            end_point=POSITION_LENS_2 + 4 * W_0 * UP,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST + 2.3 * W_0 * UP,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        gaussian_beam_waves_perturbed_2 = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_LENS_1 - W_0 * UP,
            end_point=POSITION_LENS_2 - 4 * W_0 * UP,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST - 2.3 * W_0 * UP,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)

        second_lens_outgoing_waves_unperturbed = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                       end_point=POSITION_CAMERA,
                                                                                       wavelength=WAVELENGTH,
                                                                                       width=HEIGHT_CAMERA,
                                                                                       tracker=TRACKER_TIME,
                                                                                       colors_generator=lambda
                                                                                           t: COLOR_UNPERTURBED_AMPLITUDE)
        second_lens_outgoing_waves_purterbed_1 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_LENS_2 + 0.4 * UP,
            end_point=POSITION_CAMERA + 0.8 * UP,
            wavelength=WAVELENGTH,
            width=HEIGHT_CAMERA,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        second_lens_outgoing_waves_purterbed_2 = generate_wavefronts_start_to_end_flat(
            start_point=POSITION_LENS_2 - 0.4 * UP,
            end_point=POSITION_CAMERA - 0.8 * UP,
            wavelength=WAVELENGTH,
            width=HEIGHT_CAMERA,
            tracker=TRACKER_TIME,
            colors_generator=lambda t: COLOR_PERTURBED_AMPLITUDE)
        self.next_slide()
        self.updated_object_animation([sample_outgoing_waves_moises,
                                       gaussian_beam_waves_moises,
                                       second_lens_outgoing_waves_moises], FadeOut)
        self.updated_object_animation([sample_outgoing_unperturbed_waves,
                                       sample_outgoing_perturbed_waves_1,
                                       sample_outgoing_perturbed_waves_2,
                                       gaussian_beam_waves_unperturbed,
                                       gaussian_beam_waves_perturbed_1,
                                       gaussian_beam_waves_perturbed_2,
                                       second_lens_outgoing_waves_unperturbed,
                                       second_lens_outgoing_waves_purterbed_1,
                                       second_lens_outgoing_waves_purterbed_2
                                       ], FadeIn)
        self.next_slide()
        self.start_loop()
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.end_loop()

        laser_waves = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_WAIST + LENGTH_LASER_BEAM * UP,
            end_point=POSITION_WAIST - LENGTH_LASER_BEAM * UP,
            tracker=TRACKER_TIME_LASER,
            wavelength=WAVELENGTH_LASER,
            x_R=X_R_LASER,
            w_0=W_0_LASER,
            center=POSITION_WAIST,
            colors_generator=lambda t: RED)

        orange_rgb = color_to_rgb(COLOR_PHASE_SHIFT_AMPLITUDE)
        white_rgb = color_to_rgb(COLOR_UNPERTURBED_AMPLITUDE)
        phase_shift_color_generator = lambda x: white_rgb * (1 - sigmoid(x)) + orange_rgb * sigmoid(x)
        gaussian_beam_waves_phase_shifted = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_LENS_1,
            end_point=POSITION_LENS_2,
            tracker=TRACKER_TIME,
            wavelength=WAVELENGTH,
            x_R=X_R,
            w_0=W_0,
            center=POSITION_WAIST,
            colors_generator=phase_shift_color_generator)

        second_lens_outgoing_waves_shifted = generate_wavefronts_start_to_end_flat(start_point=POSITION_LENS_2,
                                                                                   end_point=POSITION_CAMERA,
                                                                                   wavelength=WAVELENGTH,
                                                                                   width=HEIGHT_CAMERA,
                                                                                   tracker=TRACKER_TIME,
                                                                                   colors_generator=phase_shift_color_generator)
        self.play(FadeOut(title_2, shift=dy * UP),
                  title_3.animate.move_to([title_3.get_center()[0], y_0, 0]),
                  title_4.animate.move_to([title_4.get_center()[0], y_1, 0]),
                  FadeIn(title_5, shift=dy * UP),
                  titles_square.animate.set_width(title_4.width + 0.1).move_to([title_4.get_center()[0], y_1, 0])
                  )
        self.next_slide()
        self.updated_object_animation(laser_waves, FadeIn)
        self.play(gaussian_beam_waves_unperturbed.animate.become(gaussian_beam_waves_phase_shifted),
                  second_lens_outgoing_waves_unperturbed.animate.become(second_lens_outgoing_waves_shifted))
        self.remove(gaussian_beam_waves_unperturbed, second_lens_outgoing_waves_unperturbed)
        self.add(gaussian_beam_waves_phase_shifted, second_lens_outgoing_waves_shifted)
        self.next_slide()
        self.start_loop()
        self.play(TRACKER_TIME.animate.increment_value(1), run_time=2, rate_func=linear)
        self.end_loop()
        left_side_group = VGroup(incoming_waves, sample, sample_outgoing_unperturbed_waves,
                                 sample_outgoing_perturbed_waves_1, sample_outgoing_perturbed_waves_2,
                                 gaussian_beam_waves_phase_shifted, gaussian_beam_waves_perturbed_1,
                                 gaussian_beam_waves_perturbed_2, laser_waves, lens_1)
        self.updated_object_animation([left_side_group, phase_image], FadeOut)

        complex_amplitude_graph_group.move_to(POSITION_LENS_1 + RIGHT).scale(2)
        dot_complex_amplitude.scale(0.5)
        TRACKER_SCANNING_CAMERA.set_value(0)

        self.updated_object_animation([complex_amplitude_graph_group, scanning_dot_2], FadeIn)
        self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), TRACKER_SCANNING_SAMPLE.animate.set_value(1),
                  run_time=4)
        TRACKER_SCANNING_CAMERA.set_value(0), TRACKER_SCANNING_SAMPLE.set_value(0)
        circ_complex_amplitude.clear_updaters()
        dot_complex_amplitude.clear_updaters()
        line_amplitude_perturbation.clear_updaters()
        scanning_dot_2.move_to(POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP)
        complex_amplitude_graph_group -= line_complex_amplitude
        complex_amplitude_graph_group -= line_amplitude_perturbation
        self.next_slide()
        self.play(line_complex_amplitude.animate.become(Line(start=ax_complex_amplitude.c2p(0, 0),
                                                             end=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
                                                             color=COLOR_PHASE_SHIFT_AMPLITUDE,
                                                             z_index=ax_complex_amplitude.z_index + 1)),
                  dot_complex_amplitude.animate.move_to(ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE)))
        line_amplitude_perturbation = Line(start=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
                                           end=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
                                           color=COLOR_PERTURBED_AMPLITUDE,
                                           z_index=line_complex_amplitude.z_index + 1)
        complex_amplitude_graph_group += line_complex_amplitude
        complex_amplitude_graph_group += line_amplitude_perturbation

        dot_complex_amplitude.add_updater(lambda m: m.move_to(
            ax_complex_amplitude.c2p(
                AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                    6 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
                AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                    6 * PI * TRACKER_SCANNING_CAMERA.get_value())) + 1))
        ))
        line_amplitude_perturbation.add_updater(lambda l: l.become(
            Line(start=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
                 end=ax_complex_amplitude.c2p(
                     AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                         6 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
                     AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                         6 * PI * TRACKER_SCANNING_CAMERA.get_value())) + 1)),
                 color=COLOR_PERTURBED_AMPLITUDE,
                 z_index=ax_complex_amplitude.z_index + 1)))
        self.next_slide()
        self.add(line_amplitude_perturbation)
        self.play(TRACKER_SCANNING_CAMERA.animate.set_value(1), run_time=4)
        self.next_slide()
        self.updated_object_animation([complex_amplitude_graph_group, scanning_dot_2], FadeOut)
        TRACKER_SCANNING_CAMERA.set_value(0)
        camera_scanner_group -= scanning_dot_2
        camera_scanner_group.move_to(POSITION_LENS_1 - 0.2 * UP).scale(1.7)
        phase_contrast_function = ax_2.plot(lambda x: 0.3 + 0.1 * np.sin(8 * np.pi * x) - 0.1 * np.cos(3 * np.pi * x),
                                            color=COLOR_INTENSITIES)
        camera_scanner_group -= constant_intensity_function
        scanning_dot_x_axis_2.scale(0.5)
        scanning_dot_2.move_to(POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP)
        self.updated_object_animation([camera_scanner_group, scanning_dot_2], FadeIn)
        self.next_slide()
        self.play(Create(phase_contrast_function), TRACKER_SCANNING_CAMERA.animate.set_value(1), run_time=2)
        self.next_slide()
        camera_scanner_group += phase_contrast_function
        self.updated_object_animation([camera_scanner_group, scanning_dot_2], FadeOut)
        self.updated_object_animation(left_side_group, FadeIn)
        self.next_slide()

        Dt_e = 3
        Dt_l = 1
        alpha = np.arcsin(Dt_l * WAVELENGTH_LASER / (Dt_e * WAVELENGTH))

        rotated_laser_waves = generate_wavefronts_start_to_end_gaussian(
            start_point=POSITION_WAIST + LENGTH_LASER_BEAM * UP * np.cos(alpha) - LENGTH_LASER_BEAM * RIGHT * np.sin(
                alpha),
            end_point=POSITION_WAIST - (
                    LENGTH_LASER_BEAM * UP * np.cos(alpha) - LENGTH_LASER_BEAM * RIGHT * np.sin(alpha)),
            tracker=TRACKER_TIME_LASER,
            wavelength=WAVELENGTH_LASER,
            x_R=X_R_LASER,
            w_0=W_0_LASER,
            center=POSITION_WAIST,
            colors_generator=lambda t: RED)
        self.play(FadeOut(title_3, shift=dy * UP),
                  title_4.animate.move_to([title_4.get_center()[0], y_0, 0]),
                  title_5.animate.move_to([title_5.get_center()[0], y_1, 0]),
                  titles_square.animate.set_width(title_5.width + 0.1).move_to([title_5.get_center()[0], y_1, 0])
                  )
        self.play(laser_waves.animate.become(rotated_laser_waves), run_time=2)
        self.remove(laser_waves)
        self.add(rotated_laser_waves)
        self.next_slide()
        self.start_loop()

        self.play(TRACKER_TIME.animate.increment_value(Dt_e),
                  TRACKER_TIME_LASER.animate.increment_value(Dt_l), run_time=2, rate_func=linear)
        self.end_loop()
        self.play(self.camera.frame.animate.scale(ZOOM_RATIO).move_to(POSITION_WAIST - 0.4 * RIGHT))
        self.next_slide()
        self.start_loop()
        self.play(TRACKER_TIME.animate.increment_value(Dt_e),
                  TRACKER_TIME_LASER.animate.increment_value(Dt_l), run_time=2, rate_func=linear)
        self.end_loop()

        self.updated_object_animation([lens_1, sample_outgoing_unperturbed_waves,
                                       sample_outgoing_perturbed_waves_1, sample_outgoing_perturbed_waves_2], FadeOut)

        complex_amplitude_graph_group.scale(ZOOM_RATIO).move_to(
            self.camera.frame_center - self.camera.frame_width / 4 * RIGHT)
        lines_original_width = line_complex_amplitude.stroke_width
        line_complex_amplitude.set_stroke(width=lines_original_width * ZOOM_RATIO)
        line_amplitude_perturbation.set_stroke(width=lines_original_width * ZOOM_RATIO)
        circ_complex_amplitude.set_stroke(width=lines_original_width * ZOOM_RATIO)
        ax_complex_amplitude.set_stroke(width=lines_original_width * ZOOM_RATIO)
        dot_complex_amplitude.scale(1)
        graph_background = Rectangle(width=complex_amplitude_graph_group.width + 0.1,
                                     height=complex_amplitude_graph_group.height + 0.1,
                                     fill_opacity=1, fill_color=BLACK, stroke_width=0.2).move_to(
            complex_amplitude_graph_group.get_center())
        waist_scanning_dot = Dot(color=COLOR_SCANNING_DOT, radius=0.05).move_to(POSITION_WAIST)
        self.updated_object_animation([complex_amplitude_graph_group, graph_background, waist_scanning_dot], FadeIn)

        modulation_rate = 2 * PI
        dot_complex_amplitude.add_updater(lambda m: m.move_to(
            ax_complex_amplitude.c2p(AMPLITUDE_SIZE * np.sin(-np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate)),
                                     AMPLITUDE_SIZE * np.cos(
                                         np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate)))))
        line_complex_amplitude.add_updater(lambda l: l.become(
            Line(start=ax_complex_amplitude.c2p(0, 0),
                 end=ax_complex_amplitude.c2p(
                     AMPLITUDE_SIZE * np.sin(-np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate)),
                     AMPLITUDE_SIZE * np.cos(np.cos(TRACKER_TIME_LASER.get_value() * modulation_rate))),
                 stroke_width=lines_original_width * ZOOM_RATIO,
                 color=COLOR_PHASE_SHIFT_AMPLITUDE,
                 z_index=line_complex_amplitude.z_index + 1)))
        self.next_slide()
        self.start_loop()
        self.play(TRACKER_TIME.animate.increment_value(6),
                  TRACKER_TIME_LASER.animate.increment_value(2), run_time=4, rate_func=linear)
        self.end_loop()
        self.updated_object_animation(self.mobjects, FadeOut)
        self.play(self.camera.frame.animate.scale(1 / ZOOM_RATIO).move_to(ORIGIN))
        complex_amplitude_graph_group.scale(1 / ZOOM_RATIO).move_to([-3.5, 0, 0])
        dot_complex_amplitude.move_to(ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE))
        line_complex_amplitude.become(
                      Line(start=ax_complex_amplitude.c2p(0, 0),
                           end=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE),
                           stroke_width=lines_original_width,
                           color=COLOR_PHASE_SHIFT_AMPLITUDE,
                           z_index=ax_complex_amplitude.z_index + 1))
        line_amplitude_perturbation.set_stroke(width=lines_original_width)
        circ_complex_amplitude.set_stroke(width=lines_original_width)
        ax_complex_amplitude.set_stroke(width=lines_original_width)

        energy_spectrum_axes = Axes(x_range=[-1, 1, 0.25],
                                    y_range=[-1, 1, 0.25],
                                    x_length=5,
                                    y_length=5,
                                    tips=False).move_to([-complex_amplitude_graph_group.get_center()[0], 0, 0])

        labels_complex_amplitude = energy_spectrum_axes.get_axis_labels(
            Tex(r'$\omega,E$'), Tex(r'$\psi$'))

        DELTA_W = 0.2
        n = 4
        spectral_lines_generators = [lambda n=n: Line(start=energy_spectrum_axes.c2p(DELTA_W * n, 0),
                                                      end=energy_spectrum_axes.c2p(DELTA_W * n, special.jv(n,
                                                                                                           TRACKER_PHASE_MODULATION.get_value()) ** 2),
                                                      color=PURPLE_D,
                                                      stroke_width=5,
                                                      z_index=energy_spectrum_axes.z_index+1) for n in range(-4, 4)]
        spectral_lines = [always_redraw(spectral_lines_generator) for spectral_lines_generator in
                          spectral_lines_generators]
        line_complex_amplitude.clear_updaters()
        dot_complex_amplitude.clear_updaters()
        line_complex_amplitude.add_updater(lambda l: l.become(
            Line(start=ax_complex_amplitude.c2p(0, 0),
                 end=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value())),
                 stroke_width=lines_original_width * ZOOM_RATIO,
                 color=COLOR_PHASE_SHIFT_AMPLITUDE,
                 z_index=line_complex_amplitude.z_index + 1)))

        dot_complex_amplitude.add_updater(lambda l: l.move_to(ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value()))))

        self.updated_object_animation([complex_amplitude_graph_group, energy_spectrum_axes, labels_complex_amplitude], FadeIn)
        self.updated_object_animation(spectral_lines, FadeIn)
        self.next_slide()
        self.play(TRACKER_PHASE_MODULATION.animate.increment_value(2), run_time=5)
        self.updated_object_animation(list(set(spectral_lines).difference(spectral_lines[4])), lambda m: m.animate.set_color(GRAY))
        [spectral_line.clear_updaters() for spectral_line in spectral_lines]
        self.next_slide()
        line_amplitude_perturbation.clear_updaters()
        line_amplitude_perturbation.add_updater(lambda l: l.become(
                                    Line(start=ax_complex_amplitude.c2p(0, AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value())),
                                         end=ax_complex_amplitude.c2p(
                                                              AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                                                                  6 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
                                                              AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value()) +
                                                              AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                                                                  6 * PI * TRACKER_SCANNING_CAMERA.get_value())))),
                                         color=COLOR_PERTURBED_AMPLITUDE,
                                         z_index=line_complex_amplitude.z_index + 1
                                         ),

        ))
        dot_complex_amplitude.clear_updaters()
        dot_complex_amplitude.add_updater(lambda l: l.move_to(ax_complex_amplitude.c2p(
                                                              AMPLITUDE_SIZE * (np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                                                                  6 * PI * TRACKER_SCANNING_CAMERA.get_value())) - 1),
                                                              AMPLITUDE_SIZE * special.jv(0, TRACKER_PHASE_MODULATION.get_value()) +
                                                              AMPLITUDE_SIZE * (np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                                                                  6 * PI * TRACKER_SCANNING_CAMERA.get_value()))))))

        self.play(TRACKER_SCANNING_CAMERA.animate.increment_value(1), run_time=5)

        self.next_slide()
        self.updated_object_animation(self.mobjects, FadeOut)

        final_title = Tex("Questions?", color=WHITE).scale(1.5)
        self.play(Write(final_title))
        self.next_slide()
        self.wait(1)



    def updated_object_animation(self,
                                 objects: Union[Mobject, list[Mobject], VGroup],
                                 animation: Union[Callable, list[Callable]]):

        if isinstance(objects, (list, VGroup)):
            objects = list(objects)
            decomposed_objects = []
            for obj in objects:
                if isinstance(obj, (list, VGroup)):
                    objects.extend(obj)
                else:
                    decomposed_objects.append(obj)
        elif isinstance(objects, Mobject):
            decomposed_objects = [objects]
        else:
            raise TypeError("objects must be Mobject, list[Mobject] or VGroup")

        if isinstance(animation, Callable):
            animation = [animation for i in range(len(decomposed_objects))]

        object_updaters = [obj.get_updaters() for obj in decomposed_objects]
        [obj.clear_updaters() for obj in decomposed_objects]
        self.play(*[a(o) for a, o in zip(animation, decomposed_objects)])
        for i, obj in enumerate(decomposed_objects):
            for updater in object_updaters[i]:
                obj.add_updater(updater)

# m = Microscope()
# m.construct()

# manim -pql slides/scene.py Microscope
# manim-slides convert Microscope slides/presentation.html
