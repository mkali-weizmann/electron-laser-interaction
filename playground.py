from microscope import *
from tqdm import tqdm
print_parameters = True
defocus_nm = 0.0000000000e+00
Cs_mm = 0.0000000000e+00
n_electrons = 20
power_1 = 1.8648123635e+04
focal_length_mm = 3.0000000000e+00
n_z = None
for alpha_cavity_deviation_degrees in tqdm([0, 1, 2, 3, 4]):
    for resolution in tqdm([1024, 512]):
        for E_0 in tqdm([300, 80, 100, 200, 250]):
            for NA_1 in tqdm([0.1, 0.2, 0.02, 0.05, 0.08, 0.15]):
                for second_laser in tqdm([True, False]):
                    for ring_cavity in tqdm([True, False]):
                        for polarization_pies in tqdm(list(np.linspace(0, 1/2, 5, endpoint=True))):

                            first_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=True)
                            if second_laser:
                                power_2 = -1
                            else:
                                power_2 = None
                            l_1 = 1064e-9
                            l_2 = 532e-9

                            # DELETE THIS LINE, IT'S HERE ONLY FOR THE PROPOSAL DEFENSE:
                            if second_laser:
                                ring_cavity = True
                            else:
                                ring_cavity = False

                            input_wave_full = WaveFunction(E_0=Joules_of_keV(E_0), mrc_file_path=r'data\static data\apof_in_ice.mrc')
                            input_wave = WaveFunction(E_0=input_wave_full.E_0,
                                                      psi=input_wave_full.psi[100:100 + resolution, 100:100 + resolution],
                                                      coordinates=CoordinateSystem(dxdydz=input_wave_full.coordinates.dxdydz,
                                                                                   n_points=(resolution, resolution)))

                            # dummy_sample = SamplePropagator(dummy_potential=f'letters_{N_POINTS}',
                            #                                 coordinates_for_dummy_potential=CoordinateSystem(axes=(input_coordinate_system.x_axis,
                            #                                                                                        input_coordinate_system.y_axis,
                            #                                                                                        np.linspace(-5e-10, 5e-10, 2)
                            #                                                                                        )))

                            cavity = CavityNumericalPropagator(l_1=l_1, l_2=l_2, power_1=power_1, power_2=power_2, NA_1=NA_1,
                                                               ring_cavity=ring_cavity,
                                                               alpha_cavity_deviation=alpha_cavity_deviation_degrees / 360 * 2 * np.pi,
                                                               theta_polarization=polarization_pies * np.pi,
                                                               n_z=n_z, ignore_past_files=False, print_progress=False)
                            second_lens = LensPropagator(focal_length=focal_length_mm * 1e-3, fft_shift=False)
                            aberration_propagator = AberrationsPropagator(Cs=Cs_mm * 1e-3, defocus=defocus_nm * 1e-9, astigmatism_parameter=0,
                                                                          astigmatism_orientation=0)
                            M = Microscope([first_lens, cavity, second_lens, aberration_propagator], n_electrons_per_square_angstrom=n_electrons)
                            pic = M.take_a_picture(input_wave)