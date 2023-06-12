from manim import *
from manim_slides import Slide
import numpy as np
from typing import List, Union

TIME_TRACKER = ValueTracker(0)
SCANNING_TRACKER_SAMPLE = ValueTracker(0)
SCANNING_TRACKER_CAMERA = ValueTracker(0)
BEGINNING = - 8
FIRST_LENS_X = -2
SECOND_LENS_X = 4
INITIAL_VERTICAL_LENGTH = 1
FINAL_VERTICAL_LENGTH = 2
POSITION_CAMERA = np.array([SECOND_LENS_X + 2, 0, 0])
END = SECOND_LENS_X + 2
POSITION_WAIST = (FIRST_LENS_X + SECOND_LENS_X) / 3
W_0 = 0.2
X_R = (FIRST_LENS_X - SECOND_LENS_X) / 10
POSITION_SAMPLE = np.array([-5, 0, 0])
HEIGHT_SAMPLE = INITIAL_VERTICAL_LENGTH
WIDTH_SAMPLE = 0.5
HEIGHT_CAMERA = FINAL_VERTICAL_LENGTH
WIDTH_CAMERA = WIDTH_SAMPLE
POSITION_AXES_1 = np.array([-2.5, 2.5, 0])
POSITION_AXES_2 = np.array([2.5, 2.5, 0])
HEIGHT_SCANNING_AXES = 2.5
WIDTH_SCANNING_AXES = 2.5
WAVELENGTH = 0.5
AXES_RANGE = 1
AMPLITUDE_SIZE = 0.8
PHASE_SHIFT_AMPLITUDE = 0.2


def noise_function_1(x):
    return 0.1 * np.sin(3*x) + 0.2 * np.sin(2*x)

def noise_function_2(x):
    return 0.1 * np.sin(2*x) - 0.2 * np.sin(3*x)


# manim -pql slides/scene.py Microscope
# manim-slides convert Microscope slides/presentation.html

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
        Text(axis_x_label).scale(0.3), Text(axis_y_label).scale(0.3)
    )

    def scanning_dot_generator():
        scanning_dot = Dot(point=dot_start_point + tracker.get_value() * (dot_end_point - dot_start_point), color=RED)
        return scanning_dot

    scanning_dot = always_redraw(scanning_dot_generator)

    def scanning_dot_x_axis_generator():
        scanning_dot_x_axis_start_point = ax.c2p(0, 0)
        scanning_dot_x_axis_end_point = ax.c2p(AXES_RANGE, 0)
        scanning_dot_x_axis = Dot(point=scanning_dot_x_axis_start_point +
                                        tracker.get_value() * (scanning_dot_x_axis_end_point -
                                                               scanning_dot_x_axis_start_point), color=RED)
        return scanning_dot_x_axis

    scanning_dot_x_axis = always_redraw(scanning_dot_x_axis_generator)

    if function_to_plot is not None:
        amplitude_graph = ax.plot(function_to_plot, color=RED)
        return ax, labels, scanning_dot, scanning_dot_x_axis, amplitude_graph
    else:
        return ax, labels, scanning_dot, scanning_dot_x_axis


class Microscope(Slide):
    def construct(self):
        incoming_waves = generate_waves(start_point=[BEGINNING, 0, 0],
                                        end_point=POSITION_SAMPLE,
                                        wavelength=WAVELENGTH,
                                        width=HEIGHT_SAMPLE,
                                        tracker=TIME_TRACKER)

        sample_outgoing_waves = generate_waves(start_point=POSITION_SAMPLE,
                                               end_point=[FIRST_LENS_X, 0, 0],
                                               wavelength=WAVELENGTH,
                                               width=HEIGHT_SAMPLE,
                                               tracker=TIME_TRACKER)

        second_lens_outgoing_waves = generate_waves(start_point=[SECOND_LENS_X, 0, 0],
                                                    end_point=POSITION_CAMERA,
                                                    wavelength=WAVELENGTH,
                                                    width=HEIGHT_CAMERA,
                                                    tracker=TIME_TRACKER)

        lens_1 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=BLUE).move_to([FIRST_LENS_X, 0, 0])
        lens_2 = Ellipse(width=0.5, height=FINAL_VERTICAL_LENGTH + 0.5, color=BLUE).move_to([SECOND_LENS_X, 0, 0])
        sample = Rectangle(height=HEIGHT_SAMPLE, width=WIDTH_SAMPLE, color=BLUE).move_to(POSITION_SAMPLE)
        camera = Rectangle(height=HEIGHT_CAMERA, width=WIDTH_CAMERA, color=GRAY, fill_color=GRAY_A,
                           fill_opacity=0.3).move_to(POSITION_CAMERA)

        gaussian_beam_waves = [ArcBetweenPoints(
            start=[FIRST_LENS_X + i * WAVELENGTH,
                   -gaussian_beam_w_x(POSITION_WAIST - (FIRST_LENS_X + i * WAVELENGTH), W_0, X_R), 0],
            end=[FIRST_LENS_X + i * WAVELENGTH,
                 gaussian_beam_w_x(POSITION_WAIST - (FIRST_LENS_X + i * WAVELENGTH), W_0, X_R), 0],
            radius=-gaussian_beam_R_x(POSITION_WAIST - (FIRST_LENS_X + i * WAVELENGTH), X_R)) for i in range(12)]
        for i, arc in enumerate(gaussian_beam_waves):
            arc.add_updater(lambda m, i=i: m.become(ArcBetweenPoints(
                start=[FIRST_LENS_X + np.mod(TIME_TRACKER.get_value() + i * WAVELENGTH, SECOND_LENS_X - FIRST_LENS_X),
                       -gaussian_beam_w_x(
                           POSITION_WAIST - (FIRST_LENS_X + np.mod(TIME_TRACKER.get_value() + i * WAVELENGTH,
                                                                   SECOND_LENS_X - FIRST_LENS_X)),
                           W_0, X_R), 0],
                end=[FIRST_LENS_X + np.mod(TIME_TRACKER.get_value() + i * WAVELENGTH, SECOND_LENS_X - FIRST_LENS_X),
                     gaussian_beam_w_x(
                         POSITION_WAIST - (FIRST_LENS_X + np.mod(TIME_TRACKER.get_value() + i * WAVELENGTH,
                                                                 SECOND_LENS_X - FIRST_LENS_X)),
                         W_0, X_R), 0],
                radius=-gaussian_beam_R_x(POSITION_WAIST - (
                        FIRST_LENS_X + np.mod(TIME_TRACKER.get_value() + i * WAVELENGTH, SECOND_LENS_X - FIRST_LENS_X)),
                                          X_R)))
                            .set_opacity(there_and_back_with_pause(
                np.mod(TIME_TRACKER.get_value() + i * WAVELENGTH, SECOND_LENS_X - FIRST_LENS_X) / (
                        SECOND_LENS_X - FIRST_LENS_X))).set_fill(BLUE, opacity=0)
                            )
        #
        gaussian_beam_waves = VGroup(*gaussian_beam_waves)
        microscope_VGroup = VGroup(incoming_waves, sample, lens_1, gaussian_beam_waves, lens_2, sample_outgoing_waves,
                                   second_lens_outgoing_waves, camera)
        self.add(microscope_VGroup)
        self.start_loop()
        self.play(TIME_TRACKER.animate.set_value(1), run_time=2, rate_func=linear)
        self.end_loop()

        ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1 = generate_scanning_axes(
            dot_start_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP,
            dot_end_point=POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT - HEIGHT_SAMPLE / 2 * UP,
            axes_position=POSITION_AXES_1,
            tracker=SCANNING_TRACKER_SAMPLE,
            function_to_plot=lambda x: np.exp(-(6 * (x - 0.5)) ** 2),
            axis_x_label="Position",
            axis_y_label="Amplitude")

        self.play(Create(ax_1), Write(labels_1), run_time=2)
        self.next_slide()
        self.play(Create(scanning_dot_1), Create(scanning_dot_x_axis_1))
        self.next_slide()
        self.play(SCANNING_TRACKER_SAMPLE.animate.set_value(1), Create(amplitude_graph_1), run_time=2)

        ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2 = generate_scanning_axes(
            dot_start_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT + HEIGHT_CAMERA / 2 * UP,
            dot_end_point=POSITION_CAMERA - WIDTH_CAMERA / 2 * RIGHT - HEIGHT_CAMERA / 2 * UP,
            axes_position=POSITION_AXES_2,
            tracker=SCANNING_TRACKER_CAMERA,
            function_to_plot=lambda x: np.exp(-(6 * (x - 0.5)) ** 2),
            axis_x_label="Position",
            axis_y_label="Amplitude")

        intensity_plot = ax_2.plot(lambda x: np.exp(-(12 * (x - 0.5)) ** 2), color=GREEN)
        intensity_label = ax_2.get_graph_label(
            intensity_plot, "\\text{Intensity}", x_val=AXES_RANGE / 2, direction=RIGHT / 2
        ).scale(0.5)
        camera_scanner_group = VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2)
        self.next_slide()
        self.play(Create(ax_2), Write(labels_2), run_time=2)
        self.next_slide()
        self.play(Create(scanning_dot_2), Create(scanning_dot_x_axis_2))
        self.next_slide()
        self.play(SCANNING_TRACKER_CAMERA.animate.set_value(1), Create(amplitude_graph_2), run_time=2)
        self.next_slide()
        self.play(AnimationGroup(FadeIn(intensity_label), Create(intensity_plot)), run_time=2)
        self.next_slide()
        self.play(FadeOut(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2, amplitude_graph_2,
                                 ax_1, labels_1, scanning_dot_1, scanning_dot_x_axis_1, amplitude_graph_1,
                                 intensity_plot, intensity_label)))

        SCANNING_TRACKER_SAMPLE.set_value(0)

        ax_complex_amplitude = Axes(x_range=[-AXES_RANGE, AXES_RANGE, 0.25],
                                    y_range=[-AXES_RANGE, AXES_RANGE, 0.25],
                                    x_length=WIDTH_SCANNING_AXES,
                                    y_length=HEIGHT_SCANNING_AXES,
                                    tips=False).move_to(POSITION_AXES_1)
        labels_complex_amplitude = ax_complex_amplitude.get_axis_labels(
            Tex(r'$\text{Re}\left(\psi\right)$').scale(0.3), Tex(r'$\text{Im}\left(\psi\right)$').scale(0.3)
        )

        # circ_complex_amplitude = Circle(radius=np.linalg.norm(ax_complex_amplitude.c2p((AMPLITUDE_SIZE, 0)) - ax_complex_amplitude.c2p((AMPLITUDE_SIZE, 0))), color=WHITE).move_to(ax_complex_amplitude.c2p(0, 0))
        # circ_complex_amplitude.add_updater(lambda m: m.become()

        def circ_complex_amplitude_generator():
            return Circle(
                radius=np.linalg.norm(ax_complex_amplitude.c2p((AMPLITUDE_SIZE, 0)) - ax_complex_amplitude.c2p((0, 0))),
                color=WHITE).move_to(ax_complex_amplitude.c2p(0, 0))

        circ_complex_amplitude = always_redraw(circ_complex_amplitude_generator)

        def arrow_complex_amplitude_generator():
            arrow_complex_amplitude = Line(
                start=ax_complex_amplitude.c2p(0, 0),
                end=ax_complex_amplitude.c2p(AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                    6 * PI * SCANNING_TRACKER_SAMPLE.get_value())),
                                             AMPLITUDE_SIZE * np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                                                 6 * PI * SCANNING_TRACKER_SAMPLE.get_value()))),
                color=RED, z_index=ax_complex_amplitude.z_index + 1)
            return arrow_complex_amplitude

        line_complex_amplitude = always_redraw(arrow_complex_amplitude_generator)
        # The dot can not have an always_redraw updater because it is going to change color.
        dot_complex_amplitude = Dot(point=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0), color=RED)
        dot_complex_amplitude.add_updater(lambda m: m.move_to(
            ax_complex_amplitude.c2p(
                AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(6 * PI * SCANNING_TRACKER_SAMPLE.get_value())),
                AMPLITUDE_SIZE * np.sin(
                    PHASE_SHIFT_AMPLITUDE * np.sin(6 * PI * SCANNING_TRACKER_SAMPLE.get_value())))).set_z_index(
            line_complex_amplitude.z_index + 1))
        self.play(Create(ax_complex_amplitude), Create(labels_complex_amplitude), Create(circ_complex_amplitude))
        # Here it is separated because the dot has to be created after the axes, or it glitches..
        scanning_dot_1.move_to(POSITION_SAMPLE + WIDTH_SAMPLE / 2 * RIGHT + HEIGHT_SAMPLE / 2 * UP)
        self.play(Create(line_complex_amplitude), Create(dot_complex_amplitude), Create(scanning_dot_1))
        complex_amplitude_graph_group = VGroup(ax_complex_amplitude, labels_complex_amplitude)
        self.next_slide()
        self.play(SCANNING_TRACKER_SAMPLE.animate.set_value(1), run_time=4)
        self.next_slide()
        SCANNING_TRACKER_CAMERA.set_value(0)
        self.play(Create(VGroup(ax_2, labels_2, scanning_dot_2, scanning_dot_x_axis_2)))
        constant_function = ax_2.plot(lambda x: 0.25, color=RED)
        camera_scanner_group.add(constant_function)
        camera_scanner_group.remove(amplitude_graph_2)
        self.next_slide()
        self.play(SCANNING_TRACKER_CAMERA.animate.set_value(1), Create(constant_function), run_time=2)
        self.next_slide()
        # incoming_waves.clear_updaters()
        # gaussian_beam_waves.clear_updaters()
        # sample_outgoing_waves.clear_updaters()
        # second_lens_outgoing_waves.clear_updaters()
        self.play(FadeOut(microscope_VGroup), FadeOut(camera_scanner_group), FadeOut(scanning_dot_1))
        self.play(complex_amplitude_graph_group.animate.move_to([0, 0, 0]).scale(scale_factor=2.5),
                  dot_complex_amplitude.animate.set_fill(BLUE))  # ,
        self.next_slide()

        def line_amplitude_perturbation_generator():
            line_amplitude_perturbation = Line(
                start=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0),
                end=ax_complex_amplitude.c2p(
                    AMPLITUDE_SIZE * np.cos(PHASE_SHIFT_AMPLITUDE * np.sin(
                        6 * PI * SCANNING_TRACKER_SAMPLE.get_value())),
                    AMPLITUDE_SIZE * np.sin(PHASE_SHIFT_AMPLITUDE * np.sin(
                        6 * PI * SCANNING_TRACKER_SAMPLE.get_value()))),
                color=BLUE)
            return line_amplitude_perturbation

        line_amplitude_perturbation = always_redraw(line_amplitude_perturbation_generator)
        line_complex_amplitude.clear_updaters()
        line_complex_amplitude.add_updater(lambda m: m.become(Line(start=ax_complex_amplitude.c2p(0, 0),
                                                                   end=ax_complex_amplitude.c2p(AMPLITUDE_SIZE, 0),
                                                                   color=RED,
                                                                   z_index=ax_complex_amplitude.z_index + 1)),
                                           )
        tex = MathTex(r"\psi_{\text{out}}\left(x\right)="
                      r"\psi_{\text{unperturbed}}+i\psi_{\text{perturbation}}\left(x\right)").next_to(
            ax_complex_amplitude.get_bottom(), RIGHT + UP).scale(0.6)
        tex[0][8:20].set_color(RED)
        tex[0][22:].set_color(BLUE)
        self.add(tex, line_amplitude_perturbation)
        self.play(SCANNING_TRACKER_SAMPLE.animate.set_value(2), run_time=4)
        self.next_slide()
        self.play(complex_amplitude_graph_group.animate.move_to(POSITION_AXES_1).scale(scale_factor=1 / 2.5),
                  FadeOut(tex))


# manim -pql slides/scene.py Microscope
# manim-slides convert Microscope slides/presentation.html
