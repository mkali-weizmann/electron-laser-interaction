from manim import *
import numpy as np
from scene import generate_wavefronts_start_to_end_gaussian
from matplotlib import pyplot as plt


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)
SCALING_FACTOR = 1.7
ARC_RADIUS = 4 * SCALING_FACTOR
ARC_CENTER = 0.3 * SCALING_FACTOR
ARC_ANGLE = 0.3
CAVITIES_TILT_ANGLE = 0.3
UNCONCENTRICITY = 0.1 * SCALING_FACTOR
HALF_CAVITY_LENGTH = ARC_RADIUS - UNCONCENTRICITY
DUMMY_TRACKER = ValueTracker(0.65)
WAVELENGTH = 0.51 * SCALING_FACTOR
X_R = 0.9 * SCALING_FACTOR
W_0 = 0.13 * SCALING_FACTOR
SQUARE_SIDE_LENGTH = 1.5 * SCALING_FACTOR
DIAGRAM_RECTANGLE_WIDTH = SQUARE_SIDE_LENGTH / 2
DIAGRAM_RECTANGLE_HEIGHT = DIAGRAM_RECTANGLE_WIDTH * np.tan(CAVITIES_TILT_ANGLE)
PHOTONS_FREQUENCY = 2

DIAGRAM_SQUARE_UPPER_RIGHT_CORNER = np.array([DIAGRAM_RECTANGLE_WIDTH / 2, DIAGRAM_RECTANGLE_HEIGHT / 2, 0])
DIAGRAM_SQUARE_LOWER_LEFT_CORNER = np.array([-DIAGRAM_RECTANGLE_WIDTH / 2, -DIAGRAM_RECTANGLE_HEIGHT / 2, 0])
DIAGRAM_SQUARE_UPPER_LEFT_CORNER = np.array([-DIAGRAM_RECTANGLE_WIDTH / 2, DIAGRAM_RECTANGLE_HEIGHT / 2, 0])
DIAGRAM_SQUARE_LOWER_RIGHT_CORNER = np.array([DIAGRAM_RECTANGLE_WIDTH / 2, -DIAGRAM_RECTANGLE_HEIGHT / 2, 0])

WIDTH_DIFFERENCE = (SQUARE_SIDE_LENGTH - DIAGRAM_RECTANGLE_WIDTH) / 2
PHOTON_END_POINT_UPPER_RIGHT = DIAGRAM_SQUARE_UPPER_RIGHT_CORNER + np.array([WIDTH_DIFFERENCE, WIDTH_DIFFERENCE * np.tan(CAVITIES_TILT_ANGLE), 0])
PHOTON_END_POINT_LOWER_LEFT = DIAGRAM_SQUARE_LOWER_LEFT_CORNER + np.array([-WIDTH_DIFFERENCE, -WIDTH_DIFFERENCE * np.tan(CAVITIES_TILT_ANGLE), 0])
PHOTON_END_POINT_UPPER_LEFT = DIAGRAM_SQUARE_UPPER_LEFT_CORNER + np.array([-WIDTH_DIFFERENCE, WIDTH_DIFFERENCE * np.tan(CAVITIES_TILT_ANGLE), 0])
PHOTON_END_POINT_LOWER_RIGHT = DIAGRAM_SQUARE_LOWER_RIGHT_CORNER + np.array([WIDTH_DIFFERENCE, -WIDTH_DIFFERENCE * np.tan(CAVITIES_TILT_ANGLE), 0])




def photon_curve_function(start_point, end_point, amplitude=0.1, frequency=1):
    orthogonal_direction = np.cross(end_point - start_point, np.array([0, 0, 1]))
    orthogonal_direction[2] = 0
    orthogonal_direction = normalize_vector(orthogonal_direction)
    def curve_function(t):
        return start_point + (end_point - start_point) * t + amplitude * np.sin(2 * PI * frequency * t) * orthogonal_direction
    return curve_function

def mirror(center, radius, outwards_normal_angle, width, color=WHITE, stroke_width=8):
    inwards_normal = - np.array([np.cos(outwards_normal_angle), np.sin(outwards_normal_angle), 0])
    arc = Arc(radius=radius, start_angle=outwards_normal_angle - width / 2, angle=width, arc_center=center + radius * inwards_normal, color=color, stroke_width=stroke_width)
    return arc


class Cavity(Scene):
    def construct(self):
        # right_mirror = mirror(center=np.array([HALF_CAVITY_LENGTH, 0, 0]), radius=ARC_RADIUS, outwards_normal_angle=0, width=ARC_ANGLE, color=BLUE)
        # left_mirror = mirror(center=np.array([-HALF_CAVITY_LENGTH, 0, 0]), radius=ARC_RADIUS, outwards_normal_angle=PI, width=ARC_ANGLE, color=BLUE)
        right_mirror = mirror(center=np.array([HALF_CAVITY_LENGTH * np.cos(-CAVITIES_TILT_ANGLE), HALF_CAVITY_LENGTH * np.sin(-CAVITIES_TILT_ANGLE), 0]), radius=ARC_RADIUS, outwards_normal_angle=-CAVITIES_TILT_ANGLE, width=ARC_ANGLE, color=BLUE)
        left_mirror = mirror(center=np.array([-HALF_CAVITY_LENGTH * np.cos(-CAVITIES_TILT_ANGLE), -HALF_CAVITY_LENGTH * np.sin(-CAVITIES_TILT_ANGLE), 0]), radius=ARC_RADIUS, outwards_normal_angle=PI - CAVITIES_TILT_ANGLE, width=ARC_ANGLE, color=BLUE)

        upper_right_mirror = mirror(center=np.array([HALF_CAVITY_LENGTH * np.cos(CAVITIES_TILT_ANGLE), HALF_CAVITY_LENGTH * np.sin(CAVITIES_TILT_ANGLE), 0]), radius=ARC_RADIUS, outwards_normal_angle=CAVITIES_TILT_ANGLE, width=ARC_ANGLE, color=BLUE)
        lower_left_mirror = mirror(center=np.array([-HALF_CAVITY_LENGTH * np.cos(CAVITIES_TILT_ANGLE), -HALF_CAVITY_LENGTH * np.sin(CAVITIES_TILT_ANGLE), 0]), radius=ARC_RADIUS, outwards_normal_angle=PI + CAVITIES_TILT_ANGLE, width=ARC_ANGLE, color=BLUE)

        horizontal_waves = generate_wavefronts_start_to_end_gaussian(start_point=np.array([HALF_CAVITY_LENGTH * np.cos(-CAVITIES_TILT_ANGLE), HALF_CAVITY_LENGTH * np.sin(-CAVITIES_TILT_ANGLE), 0]),
                                                               end_point=np.array([-HALF_CAVITY_LENGTH * np.cos(-CAVITIES_TILT_ANGLE), -HALF_CAVITY_LENGTH * np.sin(-CAVITIES_TILT_ANGLE), 0]),
                                                               tracker=DUMMY_TRACKER,
                                                               wavelength=WAVELENGTH,
                                                               x_R=X_R,
                                                               w_0=W_0,
                                                               )
        tilted_waves = generate_wavefronts_start_to_end_gaussian(start_point=np.array([-HALF_CAVITY_LENGTH * np.cos(CAVITIES_TILT_ANGLE), -HALF_CAVITY_LENGTH * np.sin(CAVITIES_TILT_ANGLE), 0]),
                                                               end_point=np.array([HALF_CAVITY_LENGTH * np.cos(CAVITIES_TILT_ANGLE), HALF_CAVITY_LENGTH * np.sin(CAVITIES_TILT_ANGLE), 0]),
                                                               tracker=DUMMY_TRACKER,
                                                               wavelength=0.51,
                                                               x_R=0.9,
                                                               w_0=0.13,
                                                               )
        [wave.set_color(RED) for wave in horizontal_waves]
        [wave.set_color(GREEN) for wave in tilted_waves]
        photon_UR = ParametricFunction(photon_curve_function(start_point=DIAGRAM_SQUARE_UPPER_RIGHT_CORNER, end_point=PHOTON_END_POINT_UPPER_RIGHT, amplitude=0.1, frequency=PHOTONS_FREQUENCY), t_range=[0, 1], color=GREEN)
        photon_LL = ParametricFunction(photon_curve_function(start_point=DIAGRAM_SQUARE_LOWER_LEFT_CORNER, end_point=PHOTON_END_POINT_LOWER_LEFT, amplitude=0.1, frequency=PHOTONS_FREQUENCY), t_range=[0, 1], color=GREEN)
        photon_UL = ParametricFunction(photon_curve_function(start_point=DIAGRAM_SQUARE_UPPER_LEFT_CORNER, end_point=PHOTON_END_POINT_UPPER_LEFT, amplitude=0.1, frequency=PHOTONS_FREQUENCY), t_range=[0, 1], color=RED)
        photon_LR = ParametricFunction(photon_curve_function(start_point=DIAGRAM_SQUARE_LOWER_RIGHT_CORNER, end_point=PHOTON_END_POINT_LOWER_RIGHT, amplitude=0.1, frequency=PHOTONS_FREQUENCY), t_range=[0, 1], color=RED)

        square = Square(side_length=SQUARE_SIDE_LENGTH, fill_color=BLACK, fill_opacity=1)
        secondary_square = Square(side_length=SQUARE_SIDE_LENGTH)
        # svg = SVGMobject("slides/Photon-photon_scattering.SVG", color=WHITE)
        diagram_rectangle = Rectangle(color=PURPLE, height=DIAGRAM_RECTANGLE_HEIGHT, width=DIAGRAM_RECTANGLE_WIDTH)
        triangle_up = Triangle(color=PURPLE, fill_color=PURPLE, fill_opacity=1).scale(0.1).rotate(PI/2).move_to([0, DIAGRAM_RECTANGLE_HEIGHT/2, 0])
        triangle_down = Triangle(color=PURPLE, fill_color=PURPLE, fill_opacity=1).scale(0.1).rotate(3*PI/2).move_to([0, -DIAGRAM_RECTANGLE_HEIGHT/2, 0])
        triangle_left = Triangle(color=PURPLE, fill_color=PURPLE, fill_opacity=1).scale(0.1).rotate(PI).move_to([-DIAGRAM_RECTANGLE_WIDTH/2, 0, 0])
        triangle_right = Triangle(color=PURPLE, fill_color=PURPLE, fill_opacity=1).scale(0.1).rotate(0).move_to([DIAGRAM_RECTANGLE_WIDTH/2, 0, 0])

        self.add(left_mirror, right_mirror, upper_right_mirror, lower_left_mirror, horizontal_waves, tilted_waves, square, photon_UR, photon_LL, photon_UL, photon_LR, diagram_rectangle, triangle_up, triangle_down, triangle_left, triangle_right, secondary_square)



AX_X_LIM = 2

ax = Axes(
    x_range=[-AX_X_LIM, AX_X_LIM, 0.5],
    y_range=[0, 1.1, 0.25],
    tips=False)

# x_min must be > 0 because log is undefined at 0.
parabolic_graph = ax.plot(lambda x: x ** 2, x_range=[-AX_X_LIM, AX_X_LIM], use_smoothing=True, color=BLUE)
exponent_graph = ax.plot(lambda x: np.exp(-x ** 2), x_range=[-AX_X_LIM, AX_X_LIM], use_smoothing=True, color=RED)

heating_width = 0.8

parabolic_graph_heated = DashedVMobject(ax.plot(lambda x: x ** 2 * (1-np.exp(-(x/heating_width)**4)), x_range=[-AX_X_LIM, AX_X_LIM], use_smoothing=True, color=BLUE))
exponent_graph_heated = DashedVMobject(ax.plot(lambda x: np.exp(-x ** 4), x_range=[-AX_X_LIM, AX_X_LIM], use_smoothing=True, color=RED))

parabolic_legend = Text("Potential", color=RED).to_corner(UR).scale(0.6)
exponent_legend = Text("Mode", color=BLUE).scale(0.6).next_to(parabolic_legend, DOWN)

class Potential(Scene):
    def construct(self):
        self.add(ax, parabolic_graph, exponent_graph, parabolic_legend ,exponent_legend)

class PotentialHeated(Scene):
    def construct(self):
        self.add(ax, parabolic_graph_heated, exponent_graph_heated, parabolic_legend ,exponent_legend)
# %%
if __name__ == "__main__":
    from manim import *
    import numpy as np
    from matplotlib import pyplot as plt

    # fig, ax = plt.subplots(figsize=(7, 4))
    fig, ax = plt.subplots(figsize=(7, 4))
    AX_X_LIM = 2.5
    heating_width = 0.8
    x = np.linspace(-2, 4, 100)
    ax.plot(x, 0.5*x ** 2, '--', label="Potential - Low Power")
    ax.plot(x, np.exp(-x ** 2), '--', label="Mode - Low Power")
    ax.plot(x, 0.5*x ** 2 +0.1 * np.exp(-x**2), label="Potential - High Power")
    ax.plot(x, np.exp(-x ** 4), label="Mode - High Power")
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)

    plt.ylim(0, 1.1)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='right')
    plt.title("Potential and Mode, Low and High Power")
    plt.savefig("slides/media/images/potential_and_mode - assymetric.svg")
    plt.show()


