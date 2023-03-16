from microscope import *
import pandas as pd


N_POINTS = 64  # Resolution of image
l_1 = 1064e-9
l_2s = [732e-9, 832e-9, 932e-9, 1032e-9]
E_0 = 10
E_0s = [E_0 for j in range(len(l_2s))]
# l_2s = [532e-9, 532e-9, 532e-9, 532e-9, 532e-9]
# E_0s = [400, 300, 200, 100, 50, 10]
E_0s = [Joules_of_keV(keV) for keV in E_0s]
NA_1 = 0.05
pixel_size = 1e-10
theta_polarization = 0
x, y, t = np.array([0]), np.array([0], dtype=np.float64), np.array([0], dtype=np.float64)

coordinate_system = CoordinateSystem(axes=(x, y))
df_standing = pd.DataFrame(
    columns=["E_0", "beta", "alpha", "analytical_power", "numerical_power", "analytical_mask", "numerical_mask"]
)
df_ring = pd.DataFrame(
    columns=["E_0", "beta", "alpha", "analytical_power", "numerical_power", "analytical_mask", "numerical_mask"]
)
df_1f = pd.DataFrame(
    columns=["E_0", "beta", "analytical_power", "numerical_power", "analytical_mask", "numerical_mask"]
)

for i in range(len(l_2s)):
    input_wave = WaveFunction(E_0s[i], np.array([[1]]), coordinate_system)

    cavity_2f_analytical = CavityAnalyticalPropagator(
        l_1=l_1,
        l_2=l_2s[i],
        power_1=-1,
        NA_1=NA_1,
        ring_cavity=False,
        theta_polarization=theta_polarization,
        input_wave_energy_for_power_finding=E_0s[i],
    )
    cavity_2f_analytical_ring = CavityAnalyticalPropagator(
        l_1=l_1,
        l_2=l_2s[i],
        power_1=-1,
        NA_1=NA_1,
        ring_cavity=True,
        theta_polarization=theta_polarization,
        input_wave_energy_for_power_finding=E_0s[i],
    )
    cavity_2f_numerical = CavityNumericalPropagator(
        l_1=l_1,
        l_2=l_2s[i],
        power_1=-1,
        NA_1=NA_1,
        ring_cavity=False,
        theta_polarization=theta_polarization,
        input_wave_energy_for_power_finding=E_0s[i],
        n_z=50000,
    )
    cavity_2f_numerical_ring = CavityNumericalPropagator(
        l_1=l_1,
        l_2=l_2s[i],
        power_1=-1,
        NA_1=NA_1,
        ring_cavity=True,
        theta_polarization=theta_polarization,
        input_wave_energy_for_power_finding=E_0s[i],
        n_z=50000,
    )

    cavity_1f_analytical = CavityAnalyticalPropagator(
        l_1=l_1,
        l_2=l_2s[i],
        power_1=-1,
        power_2=None,
        NA_1=NA_1,
        ring_cavity=False,
        theta_polarization=theta_polarization,
        input_wave_energy_for_power_finding=E_0s[i],
    )

    cavity_1f_numerical = CavityNumericalPropagator(
        l_1=l_1,
        l_2=l_2s[i],
        power_1=-1,
        power_2=None,
        NA_1=NA_1,
        ring_cavity=False,
        theta_polarization=theta_polarization,
        input_wave_energy_for_power_finding=E_0s[i],
        n_z=50000,
    )

    phase_and_amplitude_mask_a = cavity_2f_analytical.phase_and_amplitude_mask(input_wave)[0, 0]
    phase_and_amplitude_mask_a_ring = cavity_2f_analytical_ring.phase_and_amplitude_mask(input_wave)[0, 0]
    phase_and_amplitude_mask_n = cavity_2f_numerical.phase_and_amplitude_mask(input_wave)[0, 0]
    phase_and_amplitude_mask_n_ring = cavity_2f_numerical_ring.phase_and_amplitude_mask(input_wave)[0, 0]
    phase_and_amplitude_mask_a_1f = cavity_1f_analytical.phase_and_amplitude_mask(input_wave)[0, 0]
    phase_and_amplitude_mask_n_1f = cavity_1f_numerical.phase_and_amplitude_mask(input_wave)[0, 0]

    df_standing.loc[i] = [
        np.real(keV_of_Joules(E_0s[i])),
        beta_of_E(E_0s[i]),
        cavity_2f_analytical.beta_electron2alpha_cavity(beta_of_E(E_0s[i])),
        cavity_2f_analytical.power_1,
        cavity_2f_numerical.power_1,
        phase_and_amplitude_mask_a,
        phase_and_amplitude_mask_n,
    ]
    df_ring.loc[i] = [
        keV_of_Joules(E_0s[i]),
        beta_of_E(E_0s[i]),
        cavity_2f_analytical.beta_electron2alpha_cavity(beta_of_E(E_0s[i])),
        cavity_2f_analytical_ring.power_1,
        cavity_2f_numerical_ring.power_1,
        phase_and_amplitude_mask_a_ring,
        phase_and_amplitude_mask_n_ring,
    ]

    df_1f.loc[i] = [
        keV_of_Joules(E_0s[i]),
        beta_of_E(E_0s[i]),
        cavity_1f_analytical.power_1,
        cavity_1f_numerical.power_1,
        phase_and_amplitude_mask_a_1f,
        phase_and_amplitude_mask_n_1f,
    ]

df_standing["power_ratio"] = df_standing["analytical_power"] / df_standing["numerical_power"]
df_standing["mask_ratio"] = df_standing["analytical_mask"] / df_standing["numerical_mask"]
df_standing["analytical_abs"] = np.abs(df_standing["analytical_mask"])
df_standing["numerical_abs"] = np.abs(df_standing["numerical_mask"])
df_standing["abs_ratio"] = df_standing["analytical_abs"] / df_standing["numerical_abs"]
df_standing["analytical_phase"] = np.angle(df_standing["analytical_mask"])
df_standing["numerical_phase"] = np.angle(df_standing["numerical_mask"])
df_standing["phase_diff"] = df_standing["analytical_phase"] - df_standing["numerical_phase"]
df_ring["analytical_abs"] = np.abs(df_ring["analytical_mask"])
df_ring["numerical_abs"] = np.abs(df_ring["numerical_mask"])
df_ring["abs_ratio"] = df_ring["analytical_abs"] / df_ring["numerical_abs"]
df_ring["analytical_phase"] = np.angle(df_ring["analytical_mask"])
df_ring["numerical_phase"] = np.angle(df_ring["numerical_mask"])
df_ring["phase_diff"] = df_ring["analytical_phase"] - df_ring["numerical_phase"]
df_ring["mask_ratio"] = df_ring["analytical_mask"] / df_ring["numerical_mask"]
df_ring["power_ratio"] = df_ring["analytical_power"] / df_ring["numerical_power"]
df_1f["power_ratio"] = df_1f["analytical_power"] / df_1f["numerical_power"]
df_1f["mask_ratio"] = df_1f["analytical_mask"] / df_1f["numerical_mask"]

df_standing = df_standing[["E_0", "beta", "alpha", "power_ratio", "abs_ratio"]]
df_ring = df_ring[["E_0", "beta", "alpha", "power_ratio", "abs_ratio"]]
df_1f = df_1f[["E_0", "beta", "power_ratio", "mask_ratio"]]

df_standing = df_standing.astype(np.float64)
df_ring = df_ring.astype(np.float64)
df_1f[["E_0", "beta", "power_ratio"]] = df_1f[["E_0", "beta", "power_ratio"]].astype(np.float64)

df_standing.to_csv(f"Figures\out\standing_wave - alpha - 10 - theta=0.csv")
df_ring.to_csv(f"Figures\out\\ring_wave - alpha - 10 - theta=0.csv")

# %%



# plt.plot(np.real(df_standing['E_0']), np.real(df_standing['power_ratio']), label='power_ratio')
# plt.plot(np.real(df_standing['E_0']), np.real(df_standing['abs_ratio']), label='abs_ratio')
# plt.legend()
# plt.savefig('Figures\out\standing_wave - E_0 - no correction.png')
# plt.show()
#
#
# plt.plot(np.real(df_ring['E_0']), np.real(df_ring['power_ratio']), label='power_ratio')
# plt.plot(np.real(df_ring['E_0']), np.real(df_ring['abs_ratio']), label='abs_ratio')
# plt.legend()
# plt.savefig('Figures\out\\running_wave - E_0.png')
# plt.show()
