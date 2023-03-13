from microscope import *
import itertools

import pandas as pd

power = 9.56e16
l_1 = 1064e-9
l_2 = 532e-9
NA_1 = 0.05
# theta_polarization = np.pi/2

C_dummy = CavityNumericalPropagator(
    l_1=l_1,
    l_2=l_2,
    power_1=power,
    power_2=power,
    NA_1=NA_1,
)

NA_2 = C_dummy.NA_2

power_1 = [power]
power_2 = [power]
E_0 = [300]  # 10,
l_1 = [1064e-9]
l_2 = [532e-9]
alpha_cavity = [26]  #  np.linspace(0, 32, 33)  # [0, 5, 10, 15, 20, 25]
theta_polarization = [np.pi / 2]  # 0,
x = [0]  # , 0.3e-6, 0.6e-6
y = [0]  # , 1e-6
t = np.linspace(0, 5e-15, 50) #  [0, 5e-16, 1e-15, 1.5e-15]  # [0]  #
list_of_lists = [x, y, t, theta_polarization, E_0, l_1, l_2, alpha_cavity, power_1, power_2]
zipped_lists = itertools.product(*list_of_lists)
df = pd.DataFrame(zipped_lists,
                  columns=["x", "y", "t", "theta_polarization", "E_0", "l_1", "l_2", "alpha_cavity", "power_1", "power_2"])
df = df[df["l_2"] != df["l_1"]]
df['NA_1'] = NA_1
df['NA_2'] = NA_2
df["phi"] = 0
df["alexeys_prediction"] = 0
df["ratio"] = 0
df["phi_cross_term"] = 0
df["alexeys_prediction_cross_term"] = 0
df["ratio_cross_term"] = 0
df["phi_l_1"] = 0
df["alexeys_prediction_l_1"] = 0
df["ratio_l_1"] = 0
df["phi_l_2"] = 0
df["alexeys_prediction_l_2"] = 0
df["ratio_l_2"] = 0
df["phi_standing"] = 0
df["prediction_standing"] = 0
df["ratio_standing"] = 0
df["phi_l_1_standing"] = 0
df["prediction_l_1_standing"] = 0
df["ratio_l_1_standing"] = 0
df["prediction_l_2_standing"] = 0
df["phi_l_2_standing"] = 0
df["ratio_l_2_standing"] = 0


for i in range(len(df)):

    x = df.loc[i, "x"]
    y = df.loc[i, "y"]
    t = df.loc[i, "t"]
    w_1 = w_of_l(df.loc[i, "l_1"])
    w_2 = w_of_l(df.loc[i, "l_2"])
    power_1 = df.loc[i, "power_1"]
    power_2 = df.loc[i, "power_2"]
    theta_polarization = df.loc[i, "theta_polarization"]
    alpha_cavity = df.loc[i, "alpha_cavity"] * 2 * np.pi / 360
    E_0 = df.loc[i, "E_0"]
    l_1 = df.loc[i, "l_1"]
    l_2 = df.loc[i, "l_2"]
    NA_1 = df.loc[i, "NA_1"]
    NA_2 = df.loc[i, "NA_2"]

    coordinate_system = CoordinateSystem(axes=(np.array([df.loc[i, "x"]], dtype=np.float64), np.array([df.loc[i, "y"]], dtype=np.float64)))
    input_wave = WaveFunction(np.array([[1]]), coordinate_system, Joules_of_keV(df.loc[i, "E_0"]))

    C = CavityNumericalPropagator(
        l_1=l_1,
        l_2=l_2,
        power_1=power_1,
        power_2=power_2,
        NA_1=NA_1,
        NA_2=NA_2,
        ring_cavity=True,
        theta_polarization=theta_polarization,
        n_z=50000,
        alpha_cavity=alpha_cavity,
        t=np.array([df.loc[i, "t"]], dtype=np.float64),
        debug_mode=False,
    )

    C_l_1 = CavityNumericalPropagator(
        l_1=l_1,
        l_2=None,
        power_1=power_1,
        power_2=None,
        NA_1=NA_1,
        NA_2=None,
        ring_cavity=True,
        theta_polarization=theta_polarization,
        n_z=50000,
        alpha_cavity=alpha_cavity,
        t=np.array([df.loc[i, "t"]], dtype=np.float64),
        debug_mode=False,
    )

    C_l_2 = CavityNumericalPropagator(
        l_1=l_2,
        l_2=None,
        power_1=power_2,
        power_2=None,
        NA_1=NA_2,
        NA_2=None,
        ring_cavity=True,
        theta_polarization=theta_polarization,
        n_z=50000,
        alpha_cavity=alpha_cavity,
        t=np.array([df.loc[i, "t"]], dtype=np.float64),
        debug_mode=False,
    )

    C_standing = CavityNumericalPropagator(
        l_1=l_1,
        l_2=l_2,
        power_1=power_1,
        power_2=power_2,
        NA_1=NA_1,
        NA_2=NA_2,
        ring_cavity=False,
        theta_polarization=theta_polarization,
        n_z=50000,
        alpha_cavity=alpha_cavity,
        t=np.array([df.loc[i, "t"]], dtype=np.float64),
        debug_mode=False,
    )

    C_l_1_standing = CavityNumericalPropagator(
        l_1=l_1,
        l_2=None,
        power_1=power_1,
        power_2=None,
        NA_1=NA_1,
        NA_2=None,
        ring_cavity=False,
        theta_polarization=theta_polarization,
        n_z=50000,
        alpha_cavity=alpha_cavity,
        t=np.array([df.loc[i, "t"]], dtype=np.float64),
        debug_mode=False,
    )

    C_l_2_standing = CavityNumericalPropagator(
        l_1=l_2,
        l_2=None,
        power_1=power_2,
        power_2=None,
        NA_1=NA_2,
        NA_2=None,
        ring_cavity=False,
        theta_polarization=theta_polarization,
        n_z=50000,
        alpha_cavity=alpha_cavity,
        t=np.array([df.loc[i, "t"]], dtype=np.float64),
        debug_mode=False,
    )


    phi = C.phi(input_wave)[0, 0, 0]
    phi_l_1 = C_l_1.phi(input_wave)[0, 0, 0]
    phi_l_2 = C_l_2.phi(input_wave)[0, 0, 0]
    phi_standing = C_standing.phi(input_wave)[0, 0, 0]
    phi_l_1_standing = C_l_1_standing.phi(input_wave)[0, 0, 0]
    phi_l_2_standing = C_l_2_standing.phi(input_wave)[0, 0, 0]
    df.loc[i, "phi"] = phi
    df.loc[i, "phi_l_1"] = phi_l_1
    df.loc[i, "phi_l_2"] = phi_l_2
    df.loc[i, "phi_standing"] = phi_standing
    df.loc[i, "phi_l_1_standing"] = phi_l_1_standing
    df.loc[i, "phi_l_2_standing"] = phi_l_2_standing

    beta = input_wave.beta

    cross_term = -8 * np.sqrt(np.pi * power_1 * power_2 / (C.w_0_1 ** 2 + C.w_0_2 ** 2)) * FINE_STRUCTURE_CONST / \
                 (C_LIGHT * M_ELECTRON * beta * gamma_of_beta(beta) * w_1 * w_2 * np.cos(alpha_cavity)) * \
                 np.exp(-y**2 * (1 / C.w_0_1 ** 2 + 1 / C.w_0_2 ** 2)) * \
                 safe_exponent(-((w_1 - w_2) + beta*(w_1 + w_2) * np.sin(alpha_cavity))**2 /
                               (4 * (1 / C.w_0_1 ** 2 + 1 / C.w_0_2 ** 2) * (beta * C_LIGHT * np.cos(alpha_cavity)) ** 2)) * \
                 np.cos((w_1 - w_2) * t + x/(beta * C_LIGHT * np.cos(alpha_cavity)) * ((w_1 - w_2) * np.sin(alpha_cavity) + beta * (w_1 + w_2)))

    l_1_term = - np.exp(
        -2 * y ** 2 / C_l_1.w_0_1 ** 2) * C_l_1.power_1 * FINE_STRUCTURE_CONST * C_l_1.l_1 ** 2 / \
                                                      (np.sqrt(2) * C_LIGHT ** 3 * M_ELECTRON * np.pi ** (
                                                                  3 / 2) * C_l_1.w_0_1 * beta * gamma_of_beta(
                                                          beta) * np.abs(np.cos(alpha_cavity)))
    l_2_term = - np.exp(
        -2 * y ** 2 / C_l_2.w_0_1 ** 2) * C_l_2.power_1 * FINE_STRUCTURE_CONST * C_l_2.l_1 ** 2 / \
               (np.sqrt(2) * C_LIGHT ** 3 * M_ELECTRON * np.pi ** (
                                                                            3 / 2) * C_l_2.w_0_1 * beta * gamma_of_beta(
                                                                    beta) * np.abs(np.cos(alpha_cavity)))

    l_1_term_standing = - np.exp(- 2 * y ** 2 / C_l_1_standing.w_0_1 ** 2) * C_l_1_standing.power_1 * FINE_STRUCTURE_CONST * C_l_1_standing.l_1 ** 2 * np.sqrt(2) / \
                                                          (
                                                                  C_LIGHT ** 3 * M_ELECTRON * np.pi ** (3/2) * C_l_1_standing.w_0_1 * beta * gamma_of_beta(beta) * np.abs(np.cos(alpha_cavity))
                                                          ) * \
    (
        1 +
        (-4 +
         4*beta**2 +
         beta**2 * np.cos(2*(alpha_cavity - theta_polarization)) +
         2*beta**2 * np.cos(2*theta_polarization) +
         beta**2 * np.cos(2*(alpha_cavity+theta_polarization))) /
        (
                -4 + 4*beta**2 * np.sin(alpha_cavity)**2
         ) *

        np.exp(-1/2 * k_of_l(C_l_1_standing.l_1)**2 * C_l_1_standing.w_0_1**2 * np.tan(alpha_cavity)**2)
        * np.cos(2*k_of_l(C_l_1_standing.l_1) * x / np.cos(alpha_cavity))
    )

    l_2_term_standing = - np.exp(- 2 * y ** 2 / C_l_2_standing.w_0_1 ** 2) * C_l_2_standing.power_1 * FINE_STRUCTURE_CONST * C_l_2_standing.l_1 ** 2 * np.sqrt(
        2) / \
                        (
                                C_LIGHT ** 3 * M_ELECTRON * np.pi ** (
                                    3 / 2) * C_l_2_standing.w_0_1 * beta * gamma_of_beta(beta) * np.abs(
                            np.cos(alpha_cavity))
                        ) * \
                        (
                                1 +
                                (-4 +
                                 4 * beta ** 2 +
                                 beta ** 2 * np.cos(2 * (alpha_cavity - theta_polarization)) +
                                 2 * beta ** 2 * np.cos(2 * theta_polarization) +
                                 beta ** 2 * np.cos(2 * (alpha_cavity + theta_polarization))) /
                                (
                                        -4 + 4 * beta ** 2 * np.sin(alpha_cavity) ** 2
                                ) *

                                np.exp(-1 / 2 * k_of_l(C_l_2_standing.l_1) ** 2 * C_l_2_standing.w_0_1 ** 2 * np.tan(alpha_cavity) ** 2)
                                * np.cos(2 * k_of_l(C_l_2_standing.l_1) * x / np.cos(alpha_cavity))
                        )

    df.loc[i, "alexeys_prediction"] = cross_term + l_1_term + l_2_term
    df.loc[i, "alexeys_prediction_l_1"] = l_1_term
    df.loc[i, "alexeys_prediction_l_2"] = l_2_term
    df.loc[i, "alexeys_prediction_cross_term"] = cross_term
    df.loc[i, "phi_cross_term"] = phi - phi_l_1 - phi_l_2
    df.loc[i, "prediction_standing"] = l_1_term_standing + l_2_term_standing + cross_term
    df.loc[i, "prediction_l_1_standing"] = l_1_term_standing
    df.loc[i, "prediction_l_2_standing"] = l_2_term_standing

df['l_1'] *= 1e9
df['l_2'] *= 1e9
df["ratio"] = df["alexeys_prediction"] / df["phi"]
df["ratio_cross_term"] = df["alexeys_prediction_cross_term"] / df["phi_cross_term"]
df["ratio_l_1"] = df["alexeys_prediction_l_1"] / df["phi_l_1"]
df["ratio_l_2"] = df["alexeys_prediction_l_2"] / df["phi_l_2"]
df["ratio_standing"] = df["prediction_standing"] / df["phi_standing"]
df["ratio_l_1_standing"] = df["prediction_l_1_standing"] / df["phi_l_1_standing"]
df["ratio_l_2_standing"] = df["prediction_l_2_standing"] / df["phi_l_2_standing"]
# df_standing.to_csv(f"Figures\out\standing_wave - alpha_cavity - 10 - theta=0.csv")
# df_ring.to_csv(f"Figures\out\\ring_wave - alpha_cavity - 10 - theta=0.csv")
