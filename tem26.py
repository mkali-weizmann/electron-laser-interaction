#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:39:43 2020

@author: petar

Helper functions for TEM simulation
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.random import uniform, normal, poisson
from Bio import PDB
from pandas import read_csv
import os
import os.path
import time
from scipy import special
from scipy.spatial.transform import Rotation
from scipy import ndimage
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal

global diagnostics_on
global silent_mode
diagnostics_on = False # default
silent_mode = False # default

#%% high-level dictionaries
def make_laser(*argv, **kwargs):
    '''
    Parameters
    ----------
    *argv : dict
        existing laser dict, otherwise empty
    **kwargs :
        'power' : laser power [W], default None
        'peak phase deg' : peak phase desired at focus [deg], default None
        'long_offset' : wrt camera axis [A], default 0
        'trans_offset' : wrt camera axis [A], default 0
        'z_offset' : wrt back focal plane [A], default 0
        'xy_angle' : azimuthal angle [deg], default 0
        'xz_angle' : out-of-plane tilt [deg], default 0
        'pol_angle' : laser polarization angle (rel to e-beam axis) [deg], default 90
        'mode' : 'node' or 'antinode' (whichever is at long. mode center),
                 default 'antinode'
        'NA' : numerical aperture, default 0.04
        'wlen' : laser wavelength [A], default 1.064e4     
        'true_zernike_mode' : replace lpp with true zenrike pp [bool], default False
        'zernike_cut_on' : cut-on frequency for zernike pp [A^-1]
        'crossed_lpps' : use crossed LPPs (90 deg offset) [bool], default False
        'crossed_lpp_phases_deg' : peak phase of each LPP [deg], default [45, 45]
        'crossed_lpp_phis' : longitudinal mode phase of each lpp [deg]
    Returns
    -------
    laser : dict
        updated or new depending on argv
    '''
    if len(argv) == 0:
        laser = {
            'power': None, # laser power [W]
            'peak_phase_deg': None, # peak phase desired at focus [deg]
            'long_offset': 0, # along laser axis [A]
            'trans_offset': 0, # perp to laser axis [A]
            'z_offset': 0, # wrt back focal plane [A]
            'xy_angle': 0, # azimuthal angle [deg]
            'xz_angle': 0, # out-of-plane tilt [deg]
            'pol_angle': 90, # laser polarization angle (rel to e-beam axis) [deg]
            'mode': 'antinode', # 'node' or 'antinode' (at longitudinal mode center)
            'NA': 0.04, # numerical aperture
            'wlen': 1.064e4, # optical wavelength [A]
            'true_zernike_mode': False, # replace lpp with true zernike pp [bool]
            'zernike_cut_on': 0.01, # cut on frequency for zernike pp [A^-1]
            'crossed_lpps': False, # use crossed LPPs (90 deg offset) [bool]
            'crossed_lpp_phases_deg': [45, 45], # peak phase of each LPP [deg]
            'crossed_lpp_phis': None # longitudinal mode phase of each lpp [deg]
            }
    else:
        laser = argv[0]
    for key, value in kwargs.items():
        laser.update({key: value})   
    laser = get_deriv_params('laser', laser)
    return laser

def make_camera(*argv, **kwargs):
    '''
    Parameters
    ----------
    *argv : dict
        existing camera dict, otherwise empty
    **kwargs :
        'pixel_size' : camera pixel size [A], default 0.97
        'rows' : camera rows, default 3838
        'cols' : camera columns, default 3710
        'dqe_coeffs' : coefficients for cubic polynomial fit to camera dqe (list)
        'camera_dqe_on' :  enable camera dqe envelope [bool], default False
    Returns
    -------
    camera : dict
        updated or new depending on argv
    '''
    if len(argv) == 0:
        camera = {
            'pixel_size': 0.97, # [A]
            'rows': 3838,
            'cols': 3710,
            'dqe_coeffs': [0.6, 0.247764387, -0.78184914, 0.094084754], # poly coeffs
            'camera_dqe_on': False
            }
    else:
        camera = argv[0]
    for key, value in kwargs.items():
        camera.update({key: value})
    camera = get_deriv_params('camera', camera)
    return camera

def make_scope(*argv, **kwargs):
    '''
    Parameters
    ----------
    *argv : dict
        existing scope dict, otherwise empty
    **kwargs :
        'dose' : total dose [e- per A^2],
            default 10
        'HT' : high tension [V],
            default 300e3
        'energy_spread_std' : from source [eV],
            default 0.8
        'beam_semiangle' : beam semi-angle (std dev) [rad],
            default 0
        'Cs' : spherical aberration coefficent [A],
            default 4.8e7
        'Cc' : chromatic aberration coefficient [A],
            default 7.2e7
        'focal_length' : focal length [A], default 20e7
        'accel_voltage_std' : accel V std dev rel to accel V,
            default 0.07e-6
        'OL_current_std' : objective lens current std dev rel to current,
        default 0
        'coherence_envelopes_on' : enable coherence envelopes [bool],
            default True
        'enable_cross_term' : in partial coherence calc [bool],
            default True
        'enable_phase_plate_coherence': in partial coherence calc [bool],
            default True
    Returns
    -------
    scope : dict
        updated or new depending on argv
    '''
    if len(argv) == 0:
        scope = {
            'dose': 10, # [e- per A^2]
            'HT': 300e3, # [V]
            'energy_spread_std': 0.8, # [eV]
            'beam_semiangle': 0, # [rad]
            'Cs': 4.8e7, # [A]
            'Cc': 7.2e7, # [A]
            'focal_length': 20e7, # [A]
            'accel_voltage_std': 0.07e-6, # 0.07 ppm
            'OL_current_std': 0.1, # 0.1 ppm
            'coherence_envelopes_on': True,
            'enable_cross_term': True,
            'enable_phase_plate_coherence': True
            }
    else:
        scope = argv[0]
    for key, value in kwargs.items():
        scope.update({key: value})
    scope = get_deriv_params('scope', scope)
    return scope

def make_sample(*argv, **kwargs):
    '''
    Parameters
    ----------
    *argv : dict
        existing sample dict, otherwise empty
    **kwargs :
        'name' : sample name,
            default 'test_plane'
        'circle_diam' : diameter [A],
            default 5
        'circle_lattice_const' : lattice constant (for test_circle_array) [A],
            default 20
        'test_phase' : std if test_plane, constant if test_circle [rad],
            default 0.1
        'thickness' : total sample thickness [A],
            default 200
        'xyz_loc' : wrt origin of object space [A],
            default [0,0,0]
        'xyz_angles_deg' : rotations along each axis [deg],
            default [0,0,0]
        'slice_thickness' : multislice calc. slice thickness [A],
            default 2
        'viration' : thermal vibration std. [A],
            default 0
        'solvated' : add solvent or neglect? (proteins only) [bool],
            default True
        'lattice_protein' : protein to use for lattice sample,
            default 'proteasome_ta'
        'lattice_box_size' : optional lattice spatial extent along xyz [A],
            default None
        'random_orientations' : lattice proteins randomly oriented? [bool],
            default True
        'random_positions' : lattice proteins randomly offset (xyz)? [bool],
            default True
        'random_position_magnitude' : amt. to vary pos. from lattice site [A],
            default 5
        'interp_positioning' : interp. lattice protein positions? [bool],
            default True
        'box_pad_size' : pad size for protein box calculation [A],
            default 10
        'section_thickness' : thickness of protein section [A],
            default None
        'section_z' : mean z of protein section [A],
            default 0
        'rezero_section' : adjust section in object plane? [bool or 'z'],
            default 'z'
    Returns
    -------
    sample : dict
        updated or new depending on argv
    '''
    if len(argv) == 0:
        sample = {
            'name': 'test_plane', # sample name
            'circle_diam': 5, # diameter (for test_circles) [A]
            'circle_lattice_const': 20, # lattice constant (for test_circle_array) [A]
            'test_phase': 0.1, # std if test_plane, constant if test_circle [rad]
            'thickness': 200, # total sample thickness [A]
            'xyz_loc': [0,0,0], # wrt origin of object space [A]
            'xyz_angles_deg': [0,0,0], # rotations along each axis [deg]
            'slice_thickness': 2, # multislice calc. slice thickness [A]
            'vibration': 0, # thermal vibration std. [A]
            'solvated': True, # add solvent or neglect? (proteins only) [bool]
            'lattice_protein': 'proteasome_ta', # protein to use for lattice sample
            'lattice_box_size': None, # optional lattice spatial extent along xyz [A]
            'lattice_is_2d': False, # force lattice to have only 1 protein along z
            'random_orientations': True, # lattice proteins randomly oriented? [bool]
            'random_positions': True, # lattice proteins randomly offset (xyz)? [bool]
            'random_position_magnitude': 5, # amt. to vary pos. from lattice site [A]
            'interp_positioning': True, # interp. lattice protein positions? [bool]
            'box_pad_size': 10, # pad size for protein box calculation [A]
            'section_thickness': None, # thickness of protein section [A]
            'section_z': 0, # mean z of protein section [A]
            'rezero_section': 'z', # adjust section in object plane? [bool or 'z']
            }
    else:
        sample = argv[0]
    for key, value in kwargs.items():
        sample.update({key: value})
    sample = get_deriv_params('sample', sample)
    return sample

def specify_img_features(*argv, **kwargs):
    '''
    Parameters
    ----------
    *argv : dict
        existing sample dict, otherwise empty
    **kwargs :
        'mean_z' : average defocus position [A] (positive = under-focus),
            default 0
        'astig_z' : astig half-offset [A] (extrema are +/- astig_z from avg),
            default 0
        'astig_angle' : angle of major axis [deg],
            default 0
    Returns
    -------
    img : dict
        updated or new depending on argv
    '''
    if len(argv) == 0:
        img = {
            'mean_z': 0, # average defocus position [A]
            'astig_z': 0, # astig half-offset [A] (extrema are +/- astig_z from avg)
            'astig_angle': 0 # angle of major axis [deg]
            }
    else:
        img = argv[0]
    for key, value in kwargs.items():
        img.update({key: value})
    return img

def specify_sim_features(*argv, **kwargs):
    '''
    Parameters
    ----------
    *argv : dict
        existing sim_features dict, otherwise empty
    **kwargs :
        'load_exit_wave' : location of exit wave to load [str],
            default ''
        'save_folder_name' : loc of folder to save results to [str],
            default (curr path)
        'save_exit_wave_TF' : save exit wave? [bool],
            default False
        'exit_wave_file_name' : exit wave file name [str],
            default 'exit_wave'
        'save_results_TF' : save results? [bool],
            default False
        'output_file_name' : output file name [str],
            default 'output'
        'track_time_TF' : track time? [bool],
            default False
        'diagnostic_mode_TF' : enable diagnostic mode? [bool],
            default False
        'silent_mode_TF' : enable silent mode? [bool],
            default False
    Returns
    -------
    sim_features : dict
        updated or new depending on argv
    '''
    if len(argv) == 0:
        sim_features = {
            'load_exit_wave': '',
            'save_folder_name': os.getcwd(),
            'save_exit_wave_TF': False,
            'exit_wave_file_name': 'exit_wave',
            'save_results_TF': False,
            'output_file_name': 'output',
            'track_time_TF': False,
            'diagnostic_mode_TF': False,
            'silent_mode_TF': False
            }
    else:
        sim_features = argv[0]
    for key, value in kwargs.items():
        sim_features.update({key: value})
    return sim_features
default_sim_features = specify_sim_features()

def specify_plot_features(*argv, **kwargs):
    '''
    Parameters
    ----------
    *argv : dict
        existing plot_features dict, otherwise empty
    **kwargs :
        'save_plots_TF' : save plots? [bool], default False
        'save_folder_name' : location of folder to save results to [str], default ''
        'plot_image_pdf_TF' : plot probability density? [bool], default True
        'plot_noisy_image_TF' : plot noisy image? [bool], default False
            'cmap_limits' : set colormap limits for noisy image, default None
        'plot_downsampled_image_TF' : plot downsampled image? [bool], default False
            'downsample_factor' : for downsampled image, default 5
        'plot_lpp_TF' : plot laser phase plate? [bool], default False
            'lpp_zoom_factor' : for lpp image, default 5
        'plot_bfp_TF' : plot back focal plane fields? [bool], deafult False
            'bfp_zoom_factor' : for bfp image, default 3
        'plot_asd_log10_TF' : plot log10 of amplitude spectrum? [bool], default False
            'asd_quantiles' : set quantiles for asd plot
        'plot_ctf_TF' : plot contrast transfer function? [bool], default False
            'astig_avg_TF': do astigmatic radial average? [bool], default True
            'ctf_bins' : number of bins for radial profile plot
        
    Returns
    -------
    plot_features : dict
        updated or new depending on argv
    '''
    if len(argv) == 0:
        plot_features = {
            'save_plots_TF': False,
            'save_folder_name': '',
            'plot_image_pdf_TF': True,
            'plot_noisy_image_TF': False, 'cmap_limits': None,
            'plot_downsampled_image_TF': False, 'downsample_factor': 5,
            'plot_lpp_TF': False, 'lpp_zoom_factor': 5,
            'plot_bfp_TF': False, 'bfp_zoom_factor': 3,
            'plot_asd_log10_TF': False, 'asd_quantiles': [0, 1],
            'plot_ctf_TF': False, 'astig_avg_TF': False, 'ctf_bins': 50
            }
    else:
        plot_features = argv[0]
    for key, value in kwargs.items():
        plot_features.update({key: value})
    return plot_features
default_plot_features = specify_plot_features()

#%% ultra-high-level functions
def do_microscopy(img, sample, laser, camera, scope,
                  sim_features=default_sim_features):
    # toggle silent mode - disable text during execution
    if sim_features['silent_mode_TF']:
        toggle_silent_mode('on')
    else:
        toggle_silent_mode('off')
    if not silent_mode:
        print('===========================================================')
        print('BEGINNING MICROSCOPY SESSION')
    
    time_stamps = stamp_time('start')
    
    # toggle diagnostics - extra figures and text during execution
    if sim_features['diagnostic_mode_TF']:
        toggle_diagnostics('on')
    else:
        toggle_diagnostics('off')
    
    # make directory in which results will be saved
    if not os.path.isdir(sim_features['save_folder_name']):
        if sim_features['save_exit_wave_TF'] or sim_features['save_results_TF']:
            os.mkdir(sim_features['save_folder_name'])

    if not sim_features['load_exit_wave']:        
        # compute input wave
        psi_in = 1
        
        # compute exit wave
        if sim_features['save_exit_wave_TF']:
            exit_wave_op = get_exit_wave_op(sample, psi_in, scope, camera)
        else:
            exit_wave_op = get_exit_wave_op(sample, psi_in, scope, camera,
                                            return_potential=True)
        time_stamps = stamp_time('computed_exit_wave', time_stamps)
        
        if sim_features['save_exit_wave_TF']:
            np.savez(os.path.join(sim_features['save_folder_name'],
                                  sim_features['exit_wave_file_name'] + '.npz'),
                     sample=sample,
                     psi_in=psi_in,
                     scope=scope,
                     camera=camera,
                     exit_wave_op=exit_wave_op,
                     tempy_version = os.path.basename(__file__))
            print('...Saved exit wave to: ' +
                  os.path.join(sim_features['save_folder_name'],
                               sim_features['exit_wave_file_name'] + '.npz'))
            time_stamps = stamp_time('saved_exit_wave', time_stamps)
    else:
        # load the exit wave
        loaded_exit_wave = np.load(sim_features['load_exit_wave'], allow_pickle=True)
        if not silent_mode:
            print('...Loaded exit wave from: ' + sim_features['load_exit_wave'])
        
        # totally-overwritten variables
        sample = loaded_exit_wave['sample'].item()
        camera = loaded_exit_wave['camera'].item()
        psi_in = loaded_exit_wave['psi_in']
        exit_wave_op = loaded_exit_wave['exit_wave_op'].item()
        
        # partially-overwritten variable
        temp_scope = loaded_exit_wave['scope'].item()
        scope['wlen'] = temp_scope['wlen']
        scope['HT'] = temp_scope['HT']
        scope = get_deriv_params('scope', scope) # need to recalculate these
        
        time_stamps = stamp_time('loaded_exit_wave', time_stamps)
    
    (scope, laser) = get_transfer_func(img, scope, camera, laser)
    time_stamps = stamp_time('computed_transfer_funcs', time_stamps)
    
    psi_ip = TEM(exit_wave_op['psi'], laser, scope, camera) # send psi through TEM
    img_ip_pdf = calc_image_pdf(psi_ip) # calculate probability density
    time_stamps = stamp_time('sent_psi_through_TEM', time_stamps)
    
    # prepare output dictionary
    results = {'sample': sample,
               'laser': laser,
               'camera': camera,
               'scope': scope,
               'exit_wave_op': exit_wave_op,
               'img_ip_pdf': img_ip_pdf,
               'sim_features': sim_features,
               'img': img,
               'time_stamps': time_stamps,
               'tempy_version': os.path.basename(__file__)
               }
    if sim_features['save_results_TF']:
        np.savez(os.path.join(sim_features['save_folder_name'],
                              sim_features['output_file_name'] + '.npz'),
                 results=results)
        print('...Saved everything to: ' +
              os.path.join(sim_features['save_folder_name'],
                           sim_features['output_file_name'] + '.npz'))
    
    if not silent_mode:
        print('MICROSCOPY SESSION COMPLETE.')
        print('===========================================================')
    return results

def plot_results(results, plot_features=default_plot_features):
    print('Starting plotter...')
    if not os.path.isdir(plot_features['save_folder_name']):
        if plot_features['save_plots_TF']:
            os.mkdir(plot_features['save_folder_name'])
    
    if plot_features['plot_bfp_TF']:
        plot_bfp(results['laser'], results['camera'], results['scope'],
                 plot_features)
        save_plot(plot_features, 'bfp_summary')
    if plot_features['plot_lpp_TF']:
        plot_lpp(results['laser'], results['camera'], results['scope'],
                 plot_features)
        save_plot(plot_features, 'lpp_summary')
    if plot_features['plot_image_pdf_TF']:
        plot_image(results['img_ip_pdf'], plot_features)
        save_plot(plot_features, 'image_pdf')
    if plot_features['plot_noisy_image_TF']:
        img_noisy = pdf_to_image(results['img_ip_pdf'],
                                 results['scope']['dose'],
                                 results['camera']['pixel_size'],
                                 noise_flag=True)
        plot_image(img_noisy, plot_features)
        save_plot(plot_features, 'noisy_image')
    if plot_features['plot_downsampled_image_TF']:
        img_noisy = pdf_to_image(results['img_ip_pdf'],
                                 results['scope']['dose'],
                                 results['camera']['pixel_size'],
                                 noise_flag=True)
        ds_img_noisy = downsample_image(img_noisy,
                                        plot_features['downsample_factor'])
        plot_image(ds_img_noisy, plot_features)
        save_plot(plot_features, 'noisy_image_downsampled')
    if plot_features['plot_ctf_TF']:
        plot_ctf(results['img'], results['laser'], results['camera'],
                 results['scope'], plot_features)
        save_plot(plot_features, 'ctf')
    print('...Plotting complete.')

#%% constants and derivative parameters
const = {
    'alpha': 0.0072973525693, # fine structure constant
    'c': 299792458, # speed of light [m/s]
    'hbar': 6.62607004e-34/(2*np.pi), # reduced planck's constant [kg*m^2/s]
    'e_mass': 0.511e6, # electron mass [eV/c^2]
    'e_compton': 2.42631023867e-2, # compton wavelength [A]
    'n_avogadro': 6.02214076e23, # avogadro's number [mol^-1]
    'a0': 0.529, # bohr radius [A]
    'e_charge_VA': 14.4, # electron charge [V*A]
    'e_charge_SI': 1.602176634e-19, # elementary charge [C]
    'J_per_eV': 1.602176634e-19 # joules per electron volt (unit conversion)
    }

# hardcoded dictionary - add new elements as needed
z_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8,
          'NA': 11, 'MG': 12, 'P': 15, 'S': 16,
          'FE': 26, 'AU': 79} # atomic numbers

# van der waals radii (-1: unknown), en.wikipedia.org/wiki/Van_der_Waals_radius
vdw_dict = {'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52,
            'NA': 2.27, 'MG': 1.73, 'P': 1.8, 'S': 1.8,
            'FE': -1, 'AU': 1.66}
atom_library = list(vdw_dict.keys()) # list of all atom names

##############################################################################
# ADD NEW PROTEINS BY EDITING protein_library AND ADDING AN elif STATEMENT TO
# get_struct_file_loc
protein_library = ['hemoglobin',
                   'proteasome_human',
                   'proteasome_ta',
                   'apoferritin',
                   'ribosome',
                   'myoglobin']
def get_struct_file_loc(sample_name):
    if sample_name == 'hemoglobin':
        struct_file_loc = '1bz0_hemoglobin.pdb'
    elif sample_name == 'proteasome_human':
        struct_file_loc = '6rgq_proteasome_human.pdb'
    elif sample_name == 'proteasome_ta':
        struct_file_loc = '3j9i_proteasome_ta.pdb'
    elif sample_name == 'apoferritin':
        struct_file_loc = '6z6u_apoferritin.pdb'
    elif sample_name == 'ribosome':
        struct_file_loc = '4v6x_ribosome.cif'
    elif sample_name == 'myoglobin':
        struct_file_loc = '3rgk_myoglobin.pdb'
    return struct_file_loc
##############################################################################

def get_deriv_params(flag,dic):
    if flag == 'scope':
        dic['gamma_lorentz'] = 1 + dic['HT']/const['e_mass'] # lorentz factor 
        dic['beta_lattice'] = np.sqrt(1 - 1/dic['gamma_lorentz']**2)
        dic['wlen'] = const['e_compton']/\
            (dic['gamma_lorentz']*dic['beta_lattice']) # wavelength [A]
        dic['sigma_e'] = 2*np.pi/(dic['wlen']*dic['HT']) *\
            ((const['e_mass']*const['J_per_eV'] + const['e_charge_SI']*dic['HT']) /\
             (2*const['e_mass']*const['J_per_eV'] + const['e_charge_SI']*dic['HT']))
    elif flag == 'camera':
        dic['nyquist'] = 1/(2*dic['pixel_size']) # [A^-1]
    elif flag == 'laser':
        dic['w0'] = dic['wlen']/(np.pi*dic['NA']) # waist radius [A]
        dic['zR'] = dic['w0']/dic['NA'] # rayleigh range [A]
    elif flag == 'sample':
        if dic['name'] == 'test_plane':
            dic['class'] = 'test'
        elif dic['name'] == 'test_circle':
            dic['class'] = 'test'
        elif dic['name'] == 'test_circle_array':
            dic['class'] = 'test'
        elif dic['name'] == 'random_carbon':
            dic['class'] = 'random'
            dic['min_atom_sep'] = 1.4 # [A]
            dic['atom_density_gcm3'] = 2.2 # [g/cm^3]
            dic['atom_element'] = 'C'
        elif dic['name'].split('_')[0] == 'atom':
            dic['class'] = 'atom'
            dic['atom_element'] = dic['name'].split('_')[1]
            if dic['atom_element'] not in atom_library:
                raise Exception('atoms currently supported:\n' +
                                print(', '.join(str(x) for x in atom_library)))
        elif dic['name'] in protein_library:
            dic['class'] = 'protein'
        elif dic['name'] == 'protein_lattice':
            dic['class'] = 'protein_lattice'
        else:
            raise Exception('currently supported sample names (as string):\n' +
                            'test_plane\n' +
                            'test_circle\n' +
                            'test_circle_array\n' +
                            'random_carbon\n' +
                            'protein_lattice\n' + 
                            '\n'.join(protein_library))
            
        if dic['class'] == 'protein':
            dic['struct_file_loc'] = get_struct_file_loc(dic['name'])
        elif dic['class'] == 'protein_lattice':
            dic['struct_file_loc'] = get_struct_file_loc(dic['lattice_protein'])
        
    else:
        raise Exception('acceptable flags:\n:' +
                        'scope\n' + 'camera\n' + 'laser\n' + 'sample\n')
    return dic

#%% back focal plane coordinate system
def make_bfp_coords_S(camera, negative_flag=False):
    # S is the back focal plane coordinate in spatial frequency units [A^-1]
    # determine spacing based on camera dimensions
    ds_row = 1/(camera['rows']*camera['pixel_size']) # [A^-1]
    ds_col = 1/(camera['cols']*camera['pixel_size']) # [A^-1]
    s_ax_row = np.arange(-camera['nyquist'], camera['nyquist'], ds_row) # [A^-1]
    s_ax_col = np.arange(-camera['nyquist'], camera['nyquist'], ds_col) # [A^-1]
    
    # weird accounting due to np.arange endpoint handling
    if len(s_ax_row) == camera['rows'] + 1:
        s_ax_row = s_ax_row[:-1]
    if len(s_ax_col) == camera['cols'] + 1:
        s_ax_col = s_ax_col[:-1]
    
    [S_c, S_r] = np.meshgrid(s_ax_col, s_ax_row) # [A^-1]
    S_rad = (S_c**2 + S_r**2)**(1/2) # [A^-1]
    if negative_flag: # return (-s_x, -s_y) instead of (s_x, s_y)
        return {'x': -S_c, 'y': S_r, 'r': S_rad} # xy more intuitive than rc
    else:
        return {'x': S_c, 'y': -S_r, 'r': S_rad} # xy more intuitive than rc
    
def make_laser_coords_L(S, scope, laser):
    # L is the dimensionless coordinate of the laser plane
    # calculate physical coordinates in the bfp
    Sx_phys = S['x']*scope['wlen']*scope['focal_length'] # [A]
    Sy_phys = S['y']*scope['wlen']*scope['focal_length'] # [A]
    
    # rotate and translate
    Rx_rot = Sx_phys*np.cos(np.deg2rad(laser['xy_angle'])) +\
        Sy_phys*np.sin(np.deg2rad(laser['xy_angle'])) # [A]
    Ry_rot = -Sx_phys*np.sin(np.deg2rad(laser['xy_angle'])) +\
        Sy_phys*np.cos(np.deg2rad(laser['xy_angle'])) # [A]
    Rx_trans = Rx_rot - laser['long_offset']
    Ry_trans = Ry_rot - laser['trans_offset']
    
    # make dimensionless coordinate of the laser plane
    Lx = Rx_trans/laser['zR'] # [dimensionless]
    Ly = Ry_trans/laser['w0'] # [dimensionless]
    return {'x': Lx, 'y': Ly}

def make_azimuthal_coord(S):
    # everybody's favorite azimuthal angle definition
    phi = np.arctan2(S['y'], S['x']) # azimuthal coordinate [rad]
    return phi

def make_defocus_coord(S, img):
    # output is the total defocus coordinate f(s) in [A]
    phi = make_azimuthal_coord(S) # [rad]
    f_0 = img['mean_z'] # mean defocus [A]
    f_a = img['astig_z'] # astigmatism magnitude [A]
    return f_0 + f_a*np.cos(2*(phi -\
                               np.deg2rad(img['astig_angle']))) # total defocus [A]

#%% transfer function calculations
def get_gamma(S, scope, img):
    f_tot = make_defocus_coord(S, img) # total defocus (mean + astigmatism) [A]
    return np.pi*(-f_tot*scope['wlen']*S['r']**2 +\
                  0.5*scope['Cs']*scope['wlen']**3*S['r']**4)

def get_gamma_gradient(S, scope, img):
    phi = make_azimuthal_coord(S) # [rad]
    # gradient has dimensions [rad/(A^-1)]
    dgamma_dSx = 2*np.pi*((scope['Cs']*scope['wlen']**3*S['r']**3 -\
                           img['mean_z']*scope['wlen']*S['r'])*np.cos(phi) -\
                          img['astig_z']*scope['wlen']*S['r']*\
                              np.cos(phi - 2*np.deg2rad(img['astig_angle'])))
    dgamma_dSy = 2*np.pi*((scope['Cs']*scope['wlen']**3*S['r']**3 -\
                           img['mean_z']*scope['wlen']*S['r'])*np.sin(phi) +\
                          img['astig_z']*scope['wlen']*S['r']*\
                              np.sin(phi - 2*np.deg2rad(img['astig_angle'])))
    return(dgamma_dSx, dgamma_dSy)

def get_relativistic_epsilon(HT):
    return (1 + HT/const['e_mass'])/(1 + HT/(2*const['e_mass']))

def get_defocus_spread(scope):
    epsilon = get_relativistic_epsilon(scope['HT']) # relativistic correction
    return scope['Cc']*np.sqrt((epsilon*scope['energy_spread_std']/scope['HT'])**2 +\
                               (epsilon*scope['accel_voltage_std'])**2 +\
                                   4*(scope['OL_current_std'])**2) # [A]

def get_dqe_envelope(S, camera):
    env_dqe_raw = np.polynomial.polynomial.polyval(S['r']/camera['nyquist'],
                                                   camera['dqe_coeffs'])
    env_dqe = env_dqe_raw/env_dqe_raw.max() # normalize
    env_dqe[S['r'] > camera['nyquist']] = 0
    return env_dqe

def get_transfer_func_mag_and_phase(chi, S, L, img, laser, camera, scope):
    # STANDARD DEVIATIONS of defocus [A] and beam tilt [A^-1]
    sigma_f = get_defocus_spread(scope) # [A]   
    sigma_s = scope['beam_semiangle']/scope['wlen'] # [rad] -> [A^-1]
    
    # get gamma_lattice gradient
    (dgamma_dSx, dgamma_dSy) = get_gamma_gradient(S, scope, img)
    
    # get eta gradient
    if scope['enable_phase_plate_coherence']:
        if laser['true_zernike_mode']: # true zernike phase plate
            # numerically compute gradient
            (deta_dSx, deta_dSy) = get_numerical_phase_plate_gradient(laser,
                                                                      camera,
                                                                      scope)
        else: # laser phase plate
            if laser['crossed_lpps']: # two laser phase plates crossed at 90 deg
                deta_dSx = np.zeros_like(dgamma_dSx)
                deta_dSy = np.zeros_like(dgamma_dSy)
                
                # handle laser mode input (phase shifts w.r.t. antinode-aligned)
                if laser['crossed_lpp_phis'] is not None:
                    curr_phis = [laser['crossed_lpp_phis'][0],
                                 laser['crossed_lpp_phis'][1]] # [deg]
                else: # single phi for both lasers
                    laser_phi = get_laser_phi(laser['mode'])
                    curr_phis = [laser_phi, laser_phi] # [deg]
                
                for las_ind in [0, 1]:
                    curr_laser = laser.copy() # copy laser dictionary
                    
                    # set polarization angle
                    if curr_laser['pol_angle'] == 'rra':
                        curr_laser['pol_angle'] = get_reversal_angle(scope['beta_lattice'])
                        print('Laser #{} polarization set to '.format(las_ind + 1) +
                              'relativistic reversal angle: ' +
                              '{:.2f} deg'.format(curr_laser['pol_angle']))
                    
                    # adjust settings unique to this laser
                    curr_laser['xy_angle'] = 90*las_ind
                    curr_laser.update({'peak_phase_deg':
                                       laser['crossed_lpp_phases_deg'][las_ind]}) # [deg]
                    curr_eta0 = get_eta0_from_peak_phase_deg(curr_laser, L, scope,
                                                             curr_phis[las_ind]) # [rad]
                    curr_laser.update({'eta0': curr_eta0})
                    
                    # create custom laser coordinate system
                    curr_L = make_laser_coords_L(S, scope,
                                                 curr_laser) # [dimensionless]
                    
                    # calculate gradient of this laser
                    (curr_dedx, curr_dedy) = get_laser_phase_gradient(curr_L,
                                                                      curr_laser,
                                                                      scope)
                    # add gradient to total eta gradient
                    deta_dSx += curr_dedx
                    deta_dSy += curr_dedy
            else: # single laser phase plate
                # analytically compute gradient
                (deta_dSx, deta_dSy) = get_laser_phase_gradient(L, laser, scope)
    else:
        deta_dSx = 0
        deta_dSy = 0
    
    # helpful taylor expansion terms
    V1_x = dgamma_dSx + deta_dSx
    V1_y = dgamma_dSy + deta_dSy
    V2 = -np.pi*scope['wlen']*S['r']**2
    V3_x = -2*np.pi*scope['wlen']*S['x']
    V3_y = -2*np.pi*scope['wlen']*S['y']
    
    # helpful shorthand
    if scope['enable_cross_term']:
        V1_dot_V3 = V1_x*V3_x + V1_y*V3_y
        E = (sigma_s*sigma_f)**2*(V3_x**2 + V3_y**2)
    else:
        V1_dot_V3 = 0
        E = 0
    
    A = np.exp(-sigma_s**2*(V1_x**2 + V1_y**2)/2 - sigma_f**2*\
               (V2**2 - (sigma_s**2*V1_dot_V3)**2)/(2*(1 + E)))/np.sqrt(1 + E)
    xi = -chi + (sigma_s*sigma_f)**2*V2*V1_dot_V3/(1 + E)
    return (A, xi)

def get_ctf_from_A_xi(A, xi, nA, nxi, S):
    A0 = A[np.where(S['r'] == S['r'].min())] # A(0)
    xi0 = xi[np.where(S['r'] == S['r'].min())] # xi(0)
    return -1j*A0*(A*np.exp(1j*(xi - xi0)) - nA*np.exp(-1j*(nxi - xi0)))/2

def get_atf_from_A_xi(A, xi, nA, nxi, S):
    A0 = A[np.where(S['r'] == S['r'].min())] # A(0)
    xi0 = xi[np.where(S['r'] == S['r'].min())] # xi(0)
    return A0*(A*np.exp(1j*(xi - xi0)) + nA*np.exp(-1j*(nxi - xi0)))/2

def get_transfer_func(img, scope, camera, laser, ctf_mode=False,
                      return_ctf_components=False):
    # accounts for partial coherence rigorously

    # set up coordinates
    S = make_bfp_coords_S(camera) # spatial frequency coordinate [A^-1]
    
    # objective and laser phase
    gamma = get_gamma(S, scope, img)
    (laser, L) = get_laser_phase(laser, scope, camera, const, return_L=True)
    chi = gamma + laser['eta'] # total phase in bfp [rad]
    
    # get transfer function magnitude and phase
    if scope['coherence_envelopes_on']:
        (A, xi) = get_transfer_func_mag_and_phase(chi, S, L, img, laser,
                                                  camera, scope)
    else:
        A = np.ones_like(S['r'])
        xi = -chi
    H_bfp = A*np.exp(1j*xi) # transfer function
    
    # calculate ctf
    if ctf_mode:
        nS = make_bfp_coords_S(camera, negative_flag=True) # make (-s_x, -s_y) coord
        ngamma = get_gamma(nS, scope, img) # gamma_lattice(-s)
        nlaser = laser.copy() # make copy of laser for get_laser_phase call below
        (nlaser, nL) = get_laser_phase(nlaser, scope, camera, const,
                                       return_L=True, negative_flag=True) # eta(-s)
        nchi = ngamma + nlaser['eta'] # chi(-s) = gamma_lattice(-s) + eta(-s)
        if scope['coherence_envelopes_on']:
            # get A(-s), xi(-s)
            (nA, nxi) = get_transfer_func_mag_and_phase(nchi, nS, nL, img,
                                                        nlaser, camera, scope)
        else:
            nA = np.ones_like(S['r'])
            nxi = -nchi
        
        # contrast & amplitude transfer functions
        ctf = get_ctf_from_A_xi(A, xi, nA, nxi, S)
        atf = get_atf_from_A_xi(A, xi, nA, nxi, S)
    
    # apply camera dqe
    if camera['camera_dqe_on']:
        env_dqe = get_dqe_envelope(S, camera)
    else:
        env_dqe = 1
    H_bfp *= env_dqe

    scope.update({'gamma_lattice': gamma})
    scope.update({'H_bfp': H_bfp})
    if ctf_mode:
        if return_ctf_components:
            return (A, xi, nA, nxi)
        else:
            return (ctf, atf)
    else:
        return (scope, laser)

#%% laser phase plate calculation
def get_laser_phase(laser, scope, camera, const, return_L=False,
                    return_eta=False, S=None, negative_flag=False):
    if S is None: # make back focal plane coordinates
        if negative_flag: # useful for ctf calculation
            S = make_bfp_coords_S(camera, negative_flag=True) # [A^-1]
        else:
            S = make_bfp_coords_S(camera) # [A^-1]
            if diagnostics_on:
                if not silent_mode:
                    print('DIAGNOSTICS: Laser plane pixel spacing is ' +
                          '{:.2f} nm'.format(scope['wlen']*scope['focal_length']/\
                                             (10*camera['rows']*camera['pixel_size'])) +
                              ' along rows and '
                              '{:.2f} nm'.format(scope['wlen']*scope['focal_length']/\
                                                 (10*camera['rows']*\
                                                  camera['pixel_size'])) +
                                  ' along columns.')
                    print('The above may be false if custom S was provided.')
        # prepare array to store total phase from (both) lasers
        eta = np.zeros_like((make_bfp_coords_S(camera))['r'])
    else:
        eta = np.zeros_like(S['r'])
    
    if laser['true_zernike_mode']: # make circular phase plate
        if laser['peak_phase_deg'] is None:
            laser['peak_phase_deg'] = 90
            print('laser[\'peak_phase_deg\'] not specified: set to 90 deg')
        eta_max = np.deg2rad(laser['peak_phase_deg'])
        eta = np.where(S['r'] < laser['zernike_cut_on'], eta_max, 0)
        L = make_laser_coords_L(S, scope, laser) # [dimensionless]
    
    else: # make laser standing wave phase plate        
        if laser['crossed_lpps']: # two LPPs at 90 degrees
            peak_phases_deg = laser['crossed_lpp_phases_deg'] # 2 elements, not 1!
            eta0 = []
            
            # handle laser mode input (phase shifts w.r.t. antinode-aligned)
            if laser['crossed_lpp_phis'] is not None:
                curr_phis = [laser['crossed_lpp_phis'][0],
                             laser['crossed_lpp_phis'][1]] # [deg]
            else: # single phi for both lasers
                laser_phi = get_laser_phi(laser['mode'])
                curr_phis = [laser_phi, laser_phi] # [deg]
            
            for las_ind in [0, 1]:
                curr_laser = laser.copy() # duplicate laser structure temporarily
                
                # set polarization angle
                if laser['pol_angle'] == 'rra':
                    curr_laser['pol_angle'] = get_reversal_angle(scope['beta_lattice'])
                    if diagnostics_on:
                        print('Laser #{} polarization set to '.format(las_ind + 1) +
                              'relativistic reversal angle: ' +
                              '{:.2f} deg'.format(curr_laser['pol_angle']))
                
                # set xy angle and peak phase
                curr_laser['xy_angle'] = 90*las_ind # one at 0, one at 90
                curr_laser.update({'peak_phase_deg': peak_phases_deg[las_ind]}) # [deg]
                
                # make laser coordinate system (must be remade for each laser)
                L = make_laser_coords_L(S, scope, curr_laser) # [dimensionless]
                
                # get eta0
                curr_eta0 = get_eta0_from_peak_phase_deg(curr_laser, L, scope,
                                                         curr_phis[las_ind])
                eta0.append(curr_eta0) # this will be added to the laser dict
                
                # calculate current laser phase
                curr_eta = get_eta(curr_eta0, L, scope, curr_laser,
                                   curr_phis[las_ind])
                eta += curr_eta # add current laser to total phase
        
        else: # single laser phase plate
            # make laser coordinate system
            L = make_laser_coords_L(S, scope, laser) # [dimensionless]
            
            # handle (possible) reversal angle input
            if laser['pol_angle'] == 'rra':
                laser['pol_angle'] = get_reversal_angle(scope['beta_lattice'])
                if diagnostics_on:
                    print('Laser polarization set to relativistic reversal angle: ' +
                          '{:.2f} deg'.format(laser['pol_angle']))
            
            # handle laser mode input
            laser_phi = get_laser_phi(laser['mode'])
                    
            # if peak phase was specified, use it to get eta0 and laser power
            if laser['peak_phase_deg'] is not None: # get laser power [W]
                if laser['peak_phase_deg'] < 0:
                    raise Exception('peak laser phase cannot be negative')
                else:
                    eta0 = get_eta0_from_peak_phase_deg(laser, L, scope,
                                                        laser_phi)
                    laser.update({'eta0': eta0})
                    laser = get_laser_power_from_eta0(laser,
                                                      scope['HT'],
                                                      const)
                # laser phase profile [rad]
                eta = get_eta(eta0, L, scope, laser, laser_phi) # [rad]
                
            # if power is specified instead of peak phase, get eta0
            elif laser['power'] > 0:            
                # max phase shift (note: this is computed in SI units)
                eta0 = get_eta0_from_laser_power(laser,
                                                 scope['HT'], const) # [rad]
                
                # laser phase profile [rad]
                eta = get_eta(eta0, L, scope, laser, laser_phi) # [rad]
                laser.update({'peak_phase_deg': np.rad2deg(eta.max())}) # [deg]
                laser.update({'eta0': eta0}) # [rad]
            
            # otherwise, there is no phase plate
            else:
                eta0 = 0
                L = {'x': np.ones_like(eta), 'y': np.ones_like(eta)} # a formality

    laser.update({'eta': eta, 'eta0': eta0})
    if diagnostics_on:
        if not silent_mode:
            if laser['eta'].std() != 0:
                print('DIAGNOSTICS: Peak laser phase is '
                      '{:.2f} deg.'.format(np.rad2deg(eta_max)))
            else:
                print('DIAGNOSTICS: Laser is off.')
    if return_L:
        return (laser, L)
    elif return_eta:
        return (eta, eta0)
    else:
        return laser

def get_eta(eta0, L, scope, laser, laser_phi):
    # import laser coordinates individually to make life a little easier
    Lx = L['x']
    Ly = L['y']
    
    # calculate phase due to laser standing wave
    return eta0/2*np.exp(-2*Ly**2/(1+Lx**2))/np.sqrt(1+Lx**2)*\
        (1 + (1-2*scope['beta_lattice']**2*(np.cos(np.deg2rad(laser['pol_angle'])))**2)*\
         np.exp(-np.deg2rad(laser['xz_angle'])**2*(2/(laser['NA'])**2)*(1+Lx**2))*\
             (1+Lx**2)**(-1/4)*np.cos(2*Lx*Ly**2/(1+Lx**2) +\
                                      4*Lx/(laser['NA']**2) -\
                                          1.5*np.arctan(Lx) -\
                                              np.deg2rad(laser_phi)))

def get_laser_phi(mode):
    # phi is a phase term applied to shift the laser phase between node and antinode
    if mode == 'antinode':
        return 0
    elif mode == 'node':
        return np.rad2deg(np.pi)
    else:
        raise Exception('laser mode should be set to \'node\' or \'antinode\'')

def get_eta0_from_power(laser, e_energy, const):
    scope = get_deriv_params('scope', {'HT': e_energy}) # get scope['wlen']
    
    # max phase shift (note: this is computed in SI units)
    eta0 = np.sqrt(2/np.pi**3)*const['alpha']*laser['power']*scope['wlen']*1e-10*\
        laser['wlen']*1e-10*laser['NA']/(const['hbar']*const['c']**2)
    return eta0

def get_eta0_from_peak_phase_deg(laser, L, scope, laser_phi):
    # eta0 may not be max if there are tilts and such
    eta0_test = np.deg2rad(laser['peak_phase_deg']) # [rad]
    peak_phase_deg_test = np.rad2deg(get_eta(eta0_test, L, scope,
                                             laser, laser_phi).max()) # [deg]
    eta0 = eta0_test*laser['peak_phase_deg']/peak_phase_deg_test # [rad]
    return eta0

def get_eta0_from_laser_power(laser, e_energy, const):
    scope = get_deriv_params('scope', {'HT': e_energy}) # get scope['wlen']
    
    # max phase shift (note: this is computed in SI units)
    eta0 = np.sqrt(2/np.pi**3)*const['alpha']*laser['power']*scope['wlen']*1e-10*\
        laser['wlen']*1e-10*laser['NA']/(const['hbar']*const['c']**2)
    return eta0 # [rad]

def get_laser_power_from_eta0(laser, e_energy, const,
                              return_power=False, eta0=None):
    if eta0 is None:
        eta0_deg = np.rad2deg(laser['eta0'])
    else:
        eta0_deg = np.rad2deg(eta0)
    scope = get_deriv_params('scope', {'HT': e_energy}) # get scope['wlen']
    
    # power based on peta0
    power = np.deg2rad(eta0_deg)*const['hbar']*const['c']**2/\
        (np.sqrt(2/np.pi**3)*const['alpha']*scope['wlen']*1e-10*\
         laser['wlen']*1e-10*laser['NA'])
    
    if return_power:
        return power
    else:
        laser.update({'power': power})
        return laser

def get_laser_phase_gradient(L, laser, scope):
    # partial derivatives for transformation between L and S coordinates
    # L is dimensionless, S has units [A^-1]
    partial_scale_factor = scope['wlen']*scope['focal_length']
    dLx_dSx = np.cos(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['zR']
    dLx_dSy = np.sin(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['zR']
    dLy_dSx = -np.sin(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['w0']
    dLy_dSy = np.cos(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['w0']
    
    laser_phi = get_laser_phi(laser['mode'])
    
    # laser phase gradient
    deta_dLx = laser['eta0']/(2*laser['NA']**2*(1 + L['x']**2)**(11/4))*\
        np.exp(-2*L['y']**2/(1 + L['x']**2))*\
        (laser['NA']**2*L['x']*(1 + L['x']**2)**(1/4)*\
         (4*L['y']**2 - L['x']**2 - 1) +\
             np.exp(-2*np.deg2rad(laser['xz_angle'])**2*\
                    (1 + L['x']**2)/(laser['NA']**2))*\
                 (L['x']*(-4*np.deg2rad(laser['xz_angle'])**2*\
                          (1 + L['x']**2)**2 +\
                          laser['NA']**2*(4*L['y']**2 - 1.5*(1 + L['x']**2)) +\
                              scope['beta_lattice']**2*\
                                  (8*np.deg2rad(laser['xz_angle'])**2*\
                                   (1 + L['x']**2)**2 +\
                                   laser['NA']**2*\
                                       (3*(1 + L['x']**2) - 8*L['y']**2))*\
                                      np.cos(np.deg2rad(laser['pol_angle']))**2)*\
                  np.cos(laser_phi - 4*L['x']/(laser['NA']**2) -\
                         2*L['x']*L['y']**2/(1 + L['x']**2) +\
                             1.5*np.arctan(L['x'])) +\
                      (laser['NA']**2*(0.75 - L['y']**2 +\
                                       L['x']**2*(0.75 + L['y']**2)) -\
                       2*(1 + L['x']**2)*2)*\
                          ((2*scope['beta_lattice']*\
                            np.cos(np.deg2rad(laser['pol_angle'])))**2 - 2)*\
                              np.sin(laser_phi - 4*L['x']/(laser['NA']**2) -\
                                     2*L['x']*L['y']**2/(1 + L['x']**2) +\
                                         1.5*np.arctan(L['x']))))
             
    deta_dLy = 2*laser['eta0']*L['y']/((1 + L['x']**2)**(7/4))*\
        np.exp(-2*np.deg2rad(laser['xz_angle'])**2*\
               (1 + L['x']**2)/(laser['NA']**2) -\
               2*L['y']**2/(1 + L['x']**2))*\
            ((scope['beta_lattice']**2*\
              (1 + np.cos(2*np.deg2rad(laser['pol_angle']))) - 1)*\
             (np.cos(laser_phi - 4*L['x']/(laser['NA']**2) - 2*L['x']*\
                     L['y']**2/(1 + L['x']**2) + 1.5*np.arctan(L['x'])) -\
              L['x']*np.sin(laser_phi - 4*L['x']/(laser['NA']**2) -\
                            2*L['x']*L['y']**2/(1 + L['x']**2) +\
                                1.5*np.arctan(L['x']))) -\
                 np.exp(2*np.deg2rad(laser['xz_angle'])**2*\
                        (1 + L['x']**2)/(laser['NA']**2))*\
                     (1 + L['x']**2)**(1/4))

    # transformed laser phase gradient
    deta_dSx = deta_dLx*dLx_dSx + deta_dLy*dLy_dSx # [rad/(A^-1)]
    deta_dSy = deta_dLx*dLx_dSy + deta_dLy*dLy_dSy # [rad/(A^-1)]
    return (deta_dSx, deta_dSy)

def get_laser_phase_gradient_old(L, laser, scope):
    # partial derivatives for transformation between L and S coordinates
    # L is dimensionless, S has units [A^-1]
    partial_scale_factor = scope['wlen']*scope['focal_length']
    dLx_dSx = np.cos(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['zR']
    dLx_dSy = np.sin(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['zR']
    dLy_dSx = -np.sin(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['w0']
    dLy_dSy = np.cos(np.deg2rad(laser['xy_angle']))*\
        partial_scale_factor/laser['w0']
    
    # laser phase gradient
    deta_dLx = laser['eta0']/(2*(1 + L['x']**2)**(5/2))*np.exp(-2*L['y']**2/\
                                                            (1 + L['x']**2))*\
        (L['x']*(-1 - L['x']**2 + 4*L['y']**2) + 1/(laser['NA']**2*\
                                                    (1 + L['x']**2)**(1/4))*\
         (laser['NA']**2*L['x']*(-1.5 - 1.5*L['x']**2 + 4*L['y']**2)*\
          np.cos(4*L['x']/(laser['NA']**2) + 2*L['x']*L['y']**2/(1 + L['x']**2) -\
                 1.5*np.arctan(L['x'])) +\
              (-4*(1 + L['x']**2)**2 +\
               laser['NA']**2*\
                   (1.5 - 2*L['y']**2 + L['x']**2*(1.5 + 2*L['y']**2)))*\
                  np.sin(4*L['x']/(laser['NA']**2) + 2*L['x']*L['y']**2/\
                         (1 + L['x']**2) - 1.5*np.arctan(L['x']))
            ))
    deta_dLy = -2*laser['eta0']*L['y']/((1 + L['x']**2)**(7/4))*\
        np.exp(-2*L['y']**2/(1 + L['x']**2))*\
            ((1 + L['x']**2)**(1/4) + np.cos(4*L['x']/(laser['NA']**2) +\
                                             2*L['x']*L['y']**2/(1 + L['x']**2) -\
                                                 1.5*np.arctan(L['x'])) +\
             L['x']*np.sin(4*L['x']/(laser['NA']**2) + 2*L['x']*L['y']**2/\
                           (1 + L['x']**2) - 1.5*np.arctan(L['x'])))
    
    # transformed laser phase gradient
    deta_dSx = deta_dLx*dLx_dSx + deta_dLy*dLy_dSx # [rad/(A^-1)]
    deta_dSy = deta_dLx*dLx_dSy + deta_dLy*dLy_dSy # [rad/(A^-1)]
    return (deta_dSx, deta_dSy)

def get_numerical_phase_plate_gradient(laser, camera, scope):
    # general numerical gradient calculation (for e.g. zernike phase plate)
    # units are [rad/(A^-1)]
    (deta_dSr, deta_dSc) = np.gradient(laser['eta'],
                                       1/(camera['rows']*camera['pixel_size']),
                                       1/(camera['cols']*camera['pixel_size']))
    deta_dSx = deta_dSc
    deta_dSy = -deta_dSr # flip is consistent with +y pointing up
    return (deta_dSx, deta_dSy)

def get_reversal_angle(beta):
    return np.rad2deg(np.arccos(np.sqrt(1/2)/beta)) # [deg]

#%% sample potential calculation
def read_structure(sample):
    if sample['struct_file_loc'][-4:] == '.pdb':
        print('Reading PDB file: {}'.format(sample['struct_file_loc']))
        parser = PDB.PDBParser(QUIET=True)
    elif sample['struct_file_loc'][-4:] == '.cif':
        print('Reading CIF file: {}'.format(sample['struct_file_loc']))
        parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('struct', sample['struct_file_loc'])

    for i, atom in enumerate(structure.get_atoms()):
        if i==0:
            xyz_coords = np.array(atom.get_coord())
            atom_data = {'element': [atom.element]}
        else:
            xyz_coords = np.vstack((xyz_coords, np.array(atom.get_coord())))
            atom_data['element'].append(atom.element)

    xyz = tuple(xyz_coords[:,i] for i in range(3))
    rcz = get_rcz_from_xyz(xyz)
    atom_data.update({'coords': get_atom_coords_from_rcz(rcz)})
    atom_data.update({'id': get_id_from_element(atom_data['element'])})
    atom_data = move_atom_data_to_origin(atom_data)
    atom_data = rotate_atom_data(atom_data, sample['xyz_angles_deg'])
    atom_data = move_atom_data_3D(atom_data, sample['xyz_loc'])
    
    print('...structure file read.')
    return atom_data

def get_id_from_element(atom_elements):    
    try:
        id_array = np.array([z_dict[x] for x in atom_elements])
    except KeyError:
        raise Exception('missing element in z_dict\n' +
                        'structure contains: {}'.format(np.unique(atom_elements)))
    return id_array

def move_atom_data_to_origin(atom_data, move_z_only=False):
    rcz = get_rcz_from_atom_coords(atom_data['coords'])
    if move_z_only:
        shifted_rcz = (val - np.mean(val)*np.floor(ind/2)
                       for ind, val in enumerate(rcz))
    else:
        shifted_rcz = (val - np.mean(val) for val in rcz)
    atom_data.update({'coords': get_atom_coords_from_rcz(shifted_rcz)})
    return atom_data

def rotate_atom_data(atom_data, xyz_angles_deg):
    # coordinate transformation to xyz
    rcz = get_rcz_from_atom_coords(atom_data['coords'])
    xyz = get_xyz_from_rcz(rcz)
    # rotation
    rot = Rotation.from_euler('xyz', xyz_angles_deg, degrees=True)
    xyz_coords_rotated = rot.apply(get_atom_coords_from_rcz(xyz))
    xyz_rotated = get_rcz_from_atom_coords(xyz_coords_rotated)
    # coordinate transformation back to rcz
    rcz_rotated = get_rcz_from_xyz(xyz_rotated)
    atom_data.update({'coords': get_atom_coords_from_rcz(rcz_rotated)})
    return atom_data

def move_atom_data_3D(atom_data, xyz_loc):
    # coordinate transformation to xyz
    rcz = get_rcz_from_atom_coords(atom_data['coords'])
    xyz = get_xyz_from_rcz(rcz)
    # translation
    xyz_shifted = (pos + xyz_loc[i] for i, pos in enumerate(xyz))
    # coordinate transformation back to rcz
    rcz_shifted = get_rcz_from_xyz(xyz_shifted)
    atom_data.update({'coords': get_atom_coords_from_rcz(rcz_shifted)})
    return atom_data

def get_xyz_from_rcz(rcz):
    (r, c, z) = rcz
    return tuple([c, -r, z])

def get_rcz_from_xyz(xyz):
    (x, y, z) = xyz
    return tuple([-y, x, z])

def get_rcz_from_atom_coords(atom_coords):
    return tuple([atom_coords[:,i] for i in range(3)])

def get_atom_coords_from_rcz(rcz):
    (r, c, z) = rcz
    for ind in range(len(r)):
        if ind == 0:
            atom_coords = np.array([r[0], c[0], z[0]])
        else:
            atom_coords = np.vstack((atom_coords,
                                     np.array([r[ind], c[ind], z[ind]])))
    return atom_coords

def trim_slices_lazy(shape, slices):
    result = []
    for i in range(len(slices)):
        if(type(slices[i]) == type(slice(0))):
            sst = slices[i].start
            sen = slices[i].stop

            result.append(slice(max(sst, 0), min(sen, shape[i])))
        else:
            result.append(slices[i])
    return tuple(result)

def get_slices_lazy(shape, slices):
    result = []
    for i in range(len(slices)):
        if(type(slices[i]) == type(slice(0))):
            sst = slices[i].start
            sen = slices[i].stop

            l = sen - sst

            result.append(slice(max(-sst, 0), l - max(sen - shape[i], 0)))
    return tuple(result)

def get_potential_from_atom_data(atom_data, vol_dims_rcz, pix_sizes,
                                 pot_pix=11, ss_factor=8, edge_flag=False):  
    print('Getting potential of sample atoms...')
    # get potentials for the different atom types in atom_data
    unique_ids = np.unique(atom_data['id'])
    atom_pots = [get_atomic_potential(uid, pot_pix, pix_sizes[0], ss_factor)
                 for uid in unique_ids]
    pots_dict = dict(zip(unique_ids, atom_pots))
    
    if edge_flag: # pad vol_dims_rcz for large samples (e.g. random_carbon)
        extra_pix = pot_pix
        vol_dims_rcz = vol_dims_rcz + np.array([2*pix_sizes[0]*extra_pix,
                                                2*pix_sizes[0]*extra_pix, 0]) # [A]
    
    # make large field of view
    (vol, axes) = make_volume_with_axes(vol_dims_rcz, pix_sizes) # axes in [A]
    
    # fill with projected atomic potentials
    for atom_ind, coords in enumerate(atom_data['coords']):
        # potential of current element
        curr_pot = pots_dict[atom_data['id'][atom_ind]]
        
        # indices at which to paste current potential
        center_inds = [(np.abs(axes[i] - coords[i])).argmin() for i in range(3)]
        mini_inds = [[int(center_inds[i] - np.floor(pot_pix/2)),
                      int(center_inds[i] + np.floor(pot_pix/2)) + 1]
                     for i in range(2)]

        # add potential into large fov
        slices = tuple(slice(mini_inds[i][0], mini_inds[i][1])
                       for i in range(2)) + (center_inds[2],)
        
        curr_pot = curr_pot[get_slices_lazy(vol.shape, slices)]
        slices = trim_slices_lazy(vol.shape, slices)
        vol[slices] = vol[slices] + curr_pot

    # prepare output
    if edge_flag:
        vol = vol[extra_pix:-extra_pix, extra_pix:-extra_pix]
    potential = {'v': vol,
                 'axes': axes,
                 'pix_sizes': pix_sizes}
    print('...Atomic potential computed.')
    return potential

def get_atomic_potential(atom_id, n_grid_pix, pix_size, ss_factor):
    # get atom parameters
    ap = get_atomic_potential_params(atom_id)

    # make super-sampled grid
    mini_ax = np.linspace(-pix_size*np.floor(n_grid_pix/2),
                      pix_size*np.floor(n_grid_pix/2), n_grid_pix)
    [mini_c, mini_r] = np.meshgrid(mini_ax, mini_ax) # [A] regular sampling
    delta_sup = (np.arange(1, 2*ss_factor, 2) - ss_factor)*pix_size/(2*ss_factor)
    sup_ax = [pix + delta_sup for pix in mini_ax]
    sup_ax = np.concatenate(sup_ax)
    [sup_c, sup_r] = np.meshgrid(sup_ax, sup_ax) # [A] super-sampled
    sup_rad = (sup_r**2 + sup_c**2)**(1/2) # [A] radial coordinate
    
    # Kirkland eq C.20
    pot_const = 2*np.pi**2*const['a0']*const['e_charge_VA']
    sup_pot = (2*ap[0]*special.kn(0, 2*np.pi*sup_rad*np.sqrt(ap[1])) +
               2*ap[2]*special.kn(0, 2*np.pi*sup_rad*np.sqrt(ap[3])) +
               2*ap[4]*special.kn(0, 2*np.pi*sup_rad*np.sqrt(ap[5])) +
               ap[6]/ap[7]*np.exp(-np.pi**2*sup_rad**2/ap[7]) +
               ap[8]/ap[9]*np.exp(-np.pi**2*sup_rad**2/ap[9]) +
               ap[10]/ap[11]*np.exp(-np.pi**2*sup_rad**2/ap[11])) * pot_const
    
    # integrate super-sampled squares
    mini_pot = downsample_image(sup_pot, 8)
    mini_pot /= ss_factor**2
    
    # cut off radius of potential to remove edge discontinuities
    dc = np.round(np.sqrt(n_grid_pix))
    ci = np.floor(n_grid_pix/2)
    li = n_grid_pix - 1
    xy_cutoff = [(ci-dc,0), (ci+dc,0), (ci-dc,li), (ci+dc,li),
                 (0,ci-dc), (0,ci+dc), (li,ci-dc), (li,ci+dc)]
    pot_min = []
    for (x, y) in xy_cutoff:
        pot_min.append(mini_pot[int(x), int(y)])
    pot_min = np.max(pot_min) # baseline value

    pot = mini_pot - pot_min # subtract baseline
    pot[pot < 0] = 0 # potential is non-negative
    return pot

def apply_thermal_gaussian(potential, vib_std):
    print('...Applied {} A thermal vibration '.format(vib_std) +
          'to atomic potential as Gaussian blur.')
    vib_std_pix = [vib_std/potential['pix_sizes'][int(i/2)] for i in range(3)]
    potential.update({'v': ndimage.gaussian_filter(potential['v'], vib_std_pix)})
    return potential

def get_shang_sigworth_solvent(atom_data, potential,
                               r_asymptote=7.5, r_probe=1.7):
    print('Creating solvent (Shang & Sigworth continuum model)...')
    # evaluate distance r [argument of eq 1, Shang & Sigworth JSB 2012] and polarity
    rad_dist = r_asymptote*np.ones_like(potential['v']) # r will be stored here
    id_mat = np.zeros_like(potential['v']) # atom id will be stored here
    
    mask_mat = np.ones_like(potential['v'])

    [mesh_c, mesh_r, mesh_z] = np.meshgrid(potential['axes'][1],
                                           potential['axes'][0],
                                           potential['axes'][2]) # [A]
    for atom_ind, coords in enumerate(atom_data['coords']):
        # get atom location & box around it
        center_inds = [(np.abs(potential['axes'][i] - coords[i])).argmin()
                       for i in range(3)]
        rad_inds = [[int(center_inds[i] -\
                         np.floor(r_asymptote/potential['pix_sizes'][int(i/2)])),
                     int(center_inds[i] +\
                         np.floor(r_asymptote/potential['pix_sizes'][int(i/2)])) + 1]
                    for i in range(3)]
        
        vdw_rad = vdw_dict[atom_data['element'][atom_ind]] # van der waals radius
        if vdw_rad > 0: # skip atoms with unknown vdw radius?
            slices = tuple(slice(rad_inds[i][0], rad_inds[i][1]) for i in range(3))
            curr_r = np.sqrt((mesh_r[slices] - coords[0])**2 +\
                             (mesh_c[slices] - coords[1])**2 +\
                             (mesh_z[slices] - coords[2])**2) - vdw_rad
            curr_r = curr_r[get_slices_lazy(rad_dist.shape, slices)]
            slices = trim_slices_lazy(rad_dist.shape, slices)
            update_dist = rad_dist[slices] > curr_r

            outside_mask = curr_r < r_probe
            mask_mat[slices] = np.where(outside_mask, 0, mask_mat[slices])
            
            # update distance to nearest atom, its id, and the binary mask
            rad_dist[slices] = np.where(update_dist, curr_r, rad_dist[slices])
            id_mat[slices] = np.where(update_dist, atom_data['id'][atom_ind],
                                      id_mat[slices])
        if np.mod(atom_ind, 10000) == 0:
            print('......Analyzed ' +
                  '{}/{} atoms so far...'.format(atom_ind, len(atom_data['coords'])))
    print('......Analyzed all {} sample atoms.'.format(len(atom_data['coords'])))
    print('...Solvent initialized.')
    
    # shrink binary mask
    print('...Beginning masking...')

    boundary_inds = get_boundary_inds(mask_mat)

    boundary_inds_T = np.transpose(boundary_inds)
    r_shrink_test = [] # just gathering some statistics to test this out
    for ii, ind in enumerate(boundary_inds_T):
        nearest_id = int(id_mat[tuple(ind)]) # id of nearest atom
        if(nearest_id == 0):
            continue
        nearest_elem = list(z_dict)[list(z_dict.values()).index(nearest_id)] # element

        # determine r_shrink for current point
        r_shrink = vdw_dict[nearest_elem] + rad_dist[tuple(ind)]
        r_shrink_test.append(r_shrink) # just gathering some stats to test this out
        
        # find indices within r_shrink of the current point and set them to 0 in mask
        rad_inds = [[int(center_inds[i] -\
                         np.floor(r_shrink/potential['pix_sizes'][int(i/2)])),
                     int(center_inds[i] +\
                         np.floor(r_shrink/potential['pix_sizes'][int(i/2)])) + 1]
                    for i in range(3)]
        slices = tuple(slice(rad_inds[i][0], rad_inds[i][1]) for i in range(3))
        box_axes = [np.arange(-np.floor(r_shrink/potential['pix_sizes'][int(i/2)]),
                              np.floor(r_shrink/potential['pix_sizes'][int(i/2)]) + 1,
                              1, dtype=int) for i in range(3)]
        [box_c, box_r, box_z] = np.meshgrid(box_axes[1], box_axes[0], box_axes[2])
        box_rad = np.sqrt(box_r**2 + box_c**2 + box_z**2)
        inside_radius = box_rad < r_shrink
        mask_mat[slices] = np.where(inside_radius, 1, mask_mat[slices])
        if np.mod(ii, 10000) == 0:
            print('......Analyzed ' +
                  '{}/{} points so far...'.format(ii, len(boundary_inds_T)))
    print('......Analyzed all {} points.'.format(len(boundary_inds_T)))
    print('...Masking complete.')
    if diagnostics_on:
        if not silent_mode:
            print('...DIAGNOSTICS: r_shrink has a mean value of '
                  '{:.2f} A.'.format(np.mean(r_shrink_test)))
    
    # calculate density function [eq 1, Shang & Sigworth JSB 2012]
    print('...Calculating final solvent density...')
    # pre-calculate density functions (to reduce computational load)
    r_axis = np.arange(-1*(np.max(list(vdw_dict.values())) + 1),
                       np.max(rad_dist) + 1, 0.01)
    nonpolar_density_ss = get_shang_sigworth_density(r_axis, 'nonpolar')
    nonpolar_mat = np.interp(rad_dist, r_axis, nonpolar_density_ss)
    polar_density_ss = get_shang_sigworth_density(r_axis, 'polar')
    polar_mat = np.interp(rad_dist, r_axis, polar_density_ss)
    
    # evaluate density due to rad_dist at each point in volume
    density_ss = np.where(id_mat == 6,
                          nonpolar_mat,
                          polar_mat)
    print('...Solvent ready.')
    return density_ss*mask_mat # eq 3, Shang & Sigworth JSB 2012

def get_shang_sigworth_density(rad_dist, p_np):
    # values from tbl 1, Shang & Sigworth JSB 2012
    if p_np == 'polar':
        ssp = {'a2': 0.2, 'a3': -0.15, 'r1': 0.5, 'r2': 1.7, 'r3': 1.7,
                'sig1': 1, 'sig2': 1.77, 'sig3': 1.06}
    elif p_np == 'nonpolar':
        ssp = {'a2': 0.15, 'a3': -0.12, 'r1': 1.0, 'r2': 2.2, 'r3': 3.6,
                'sig1': 1, 'sig2': 1.77, 'sig3': 0.85}
     # eq 1, Shang & Sigworth JSB 2012
    return 0.5 + 0.5*special.erf((rad_dist - ssp['r1'])/(np.sqrt(2)*ssp['sig1'])) +\
        ssp['a2']*np.exp(-(rad_dist - ssp['r2'])**2/(2*ssp['sig2']**2)) +\
            ssp['a3']*np.exp(-(rad_dist - ssp['r3'])**2/(2*ssp['sig3']**2))

def get_boundary_inds(bin_mask):
    print('......Getting boundary indices...')
    full_coords_list = []
    for ax_ind in range(3):
        diff_vec = np.diff(bin_mask, axis=ax_ind)
        coords_init = diff_vec.nonzero()
        # only the current axis needs adjustment
        vals_init = diff_vec[coords_init]
        ax_new = np.where(vals_init < 0, coords_init[ax_ind],
                          coords_init[ax_ind]+1) # update axis at vals_init=+1
        coords_new = np.array(coords_init)
        coords_new[ax_ind] = ax_new # replace with updated axis
        full_coords_list.append(coords_new)
        print('.........Axis {}/3 ready...'.format(ax_ind+1))
    all_coords = np.hstack(tuple(full_coords_list[i] for i in range(3)))
    return tuple(all_coords[i] for i in range(3))

def get_atomic_potential_params(atom_id):
    if atom_id == -1 | atom_id > 102:
        raise Exception('Unsupported atom_id, should be between 0 and 102')
    else:
        with open('fparams.csv') as csvfile:
            params_list = read_csv(csvfile, header=None).to_numpy()
        params = np.array(params_list[atom_id,:]).astype(np.float)
    return params

def make_volume_with_axes(vol_dims, pix_sizes, val=0):
    axes = [np.arange(-vol_dims[a]/2, vol_dims[a]/2, pix_sizes[int(a/2)])
            for a in range(3)]
    vol = val*np.ones((len(axes[0]), len(axes[1]), len(axes[2])))
    return (vol, axes)

def make_random_atoms(vol_dims_rcz, min_atom_separation,
                      atom_density_gcm3, atom_element):
    atom_id = get_id_from_element(atom_element)[0] # id is atomic number
    atomic_weight = 2*atom_id # approximation!! reasonable for C, O
    subst_vol = np.prod(vol_dims_rcz) # [A^3]
    n_atoms = int(subst_vol*atom_density_gcm3*1e-24*const['n_avogadro']/atomic_weight)
    
    print('Making random atoms...')
    atom1 = get_rand_pos(vol_dims_rcz)
    atom2 = get_rand_pos(vol_dims_rcz)
    while np.sum((atom1 - atom2)**2) < min_atom_separation**2:
        print('...False start, trying again...')
        atom2 = get_rand_pos(vol_dims_rcz)
    atom_coords = np.vstack((atom1, atom2))
    
    atom_count = 2
    while atom_count < n_atoms:
        r_try = get_rand_pos(vol_dims_rcz)
        if np.any(get_all_distances(r_try, atom_coords) < min_atom_separation):
            continue
        else:
            atom_coords = np.vstack((atom_coords, r_try))
            atom_count += 1
            if np.mod(atom_count, 10000) == 0:
                print('...Made {}/{} atoms so far...'.format(atom_count, n_atoms))
    print('...Made all {}/{} atoms.'.format(len(atom_coords), n_atoms))  
    
    atom_data = {'coords': atom_coords,
                 'element': [atom_element]*len(atom_coords),
                 'id': [atom_id]*len(atom_coords)}
    return atom_data

def get_rand_pos(box_xyz_dims):
    return np.array([uniform(0, box_xyz_dims[0]),
                     uniform(0, box_xyz_dims[1]),
                     uniform(0, box_xyz_dims[2])])

def get_all_distances(r0,r_list):
    return np.sqrt(np.sum((r0 - r_list)**2, axis=1))

def calc_3d_pair_corr(atom_coords, r_max, dr, vol_dims_rcz):
    print('Calculating pair correlation...')
    print('...This may take a few minutes...')
    r_axis = np.arange(0, r_max+dr, dr) + dr/2
    g_full = np.zeros(len(r_axis))
    (r_coords, c_coords, z_coords) = get_rcz_from_atom_coords(atom_coords)
    
    int_bool = np.logical_and.reduce((np.asarray(r_coords) > r_max,
                                      np.asarray(c_coords) > r_max,
                                      np.asarray(z_coords) > r_max,
                                      np.asarray(r_coords) < vol_dims_rcz[0] - r_max,
                                      np.asarray(c_coords) < vol_dims_rcz[1] - r_max,
                                      np.asarray(z_coords) < vol_dims_rcz[2] - r_max))
    
    if sum(int_bool) < 1:
        raise RuntimeError('no atoms far enough from edge, try reducing r_max')
    interior_indices = [i for i, val in enumerate(int_bool) if val]
    
    for ind in interior_indices:
        dist_array = get_all_distances(atom_coords[ind], atom_coords)
        dist_array = np.delete(dist_array, [ind])
        index_array = np.array(list(map(int, dist_array/dr)))
        index_array[index_array > r_max/dr] = -1
        
        # count number of distinct entries
        c = dict()
        for i in index_array:
            c[i] = c.get(i, 0) + 1
        
        # add entries to full matrix
        for found_ind in list(c)[1:]:
            g_full[found_ind] += c[found_ind]
    
    number_density = len(atom_coords)/np.prod(vol_dims_rcz) # [number/A^3]
    g_out = g_full/(number_density*4*np.pi*r_axis**2*dr*len(interior_indices))
    
    print('...Finished!')
    return(r_axis[:-1], g_out[:-1])

def check_atom_coords(atom_coords, min_atom_separation):
    print('Checking for bad atoms may take a while...')
    (r_coords, c_coords, z_coords) = get_rcz_from_atom_coords(atom_coords)
    too_close = []
    for atom_id in range(len(r_coords)):
        if atom_id in too_close:
            continue
        else:
            # find nearby atoms
            close_atoms_r = set(np.asarray(abs(r_coords - r_coords[atom_id]) <=
                                           min_atom_separation).nonzero()[0].tolist())
            close_atoms_c = set(np.asarray(abs(c_coords - c_coords[atom_id]) <=
                                           min_atom_separation).nonzero()[0].tolist())
            close_atoms_z = set(np.asarray(abs(z_coords - z_coords[atom_id]) <=
                                           min_atom_separation).nonzero()[0].tolist())
            close_atoms_3d = set.intersection(close_atoms_r,
                                              close_atoms_c,
                                              close_atoms_z)
            close_atoms_3d.remove(atom_id)
            
            for test_id in close_atoms_3d:
                test_dist = np.sqrt(sum((np.array([r_coords[atom_id],
                                                   c_coords[atom_id],
                                                   z_coords[atom_id]]) -
                                         np.array([r_coords[test_id],
                                                   c_coords[test_id],
                                                   z_coords[test_id]]))**2))
                if test_dist < min_atom_separation:
                    too_close.append(test_id)
    print('...Finished! ' +
          'There are {} atoms too close together.'.format(len(too_close)))
    return too_close

#%% exit wave calculation
def get_exit_wave_op(sample, psi_in, scope, camera,
                     axial_pix_size=1, return_potential=True, atom_data=None):
    pix_sizes = [camera['pixel_size'], axial_pix_size] # [A]
    volume_dims = [camera['rows']*pix_sizes[0], camera['cols']*pix_sizes[0],
                   np.round(sample['thickness'])] # [A]
    
    if sample['class'] == 'test':
        if sample['name'] == 'test_plane':
            test_object = normal(0, sample['test_phase'],
                                 (camera['rows'], camera['cols']))
        elif sample['name'] == 'test_circle':
            circle_frac = sample['circle_diam']/(camera['rows']*camera['pixel_size'])
            phase_mask = make_aperture_mask(np.ones((camera['rows'],
                                                     camera['cols'])),
                                            circle_frac)
            test_object = sample['test_phase']*phase_mask
        elif sample['name'] == 'test_circle_array':
            lattice_const_pix = int(np.round(sample['circle_lattice_const']/\
                                             camera['pixel_size'])) # [pixels]
            circle_frac = sample['circle_diam']/(camera['rows']*camera['pixel_size'])
            circle_mask = make_aperture_mask(np.ones((camera['rows'],
                                                      camera['cols'])),
                                             circle_frac)
            phase_mask = np.zeros_like(circle_mask)
            row_max = int(np.round(camera['rows']/lattice_const_pix)/2)
            col_max = int(np.round(camera['cols']/lattice_const_pix)/2)
            for row in range(-row_max + 1, row_max):
                for col in range(-col_max + 1, col_max):
                    curr_circle = np.roll(np.roll(circle_mask,
                                                  lattice_const_pix*row,
                                                  axis=0),
                                          lattice_const_pix*col, axis=1)
                    phase_mask += curr_circle
            test_object = sample['test_phase']*phase_mask
        
        # prepare exit wave and make diagnostic plot
        exit_wave_op = {'psi': np.exp(1j*test_object)}
        if diagnostics_on:
            plt.figure()
            plt.suptitle('test object exit wave')
            plt.subplot(1,2,1)
            plt.imshow(np.abs(exit_wave_op['psi']))
            plt.title(r'$\left|\psi_{op}\right|$')
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(np.angle(exit_wave_op['psi']))
            plt.title(r'$\operatorname{arg}\left(\psi_{op}\right)$')
            plt.colorbar()
    elif sample['class'] == 'random':
        # make random atoms
        atom_data = make_random_atoms(volume_dims, sample['min_atom_sep'],
                                      sample['atom_density_gcm3'],
                                      sample['atom_element'])
        atom_data = move_atom_data_to_origin(atom_data)
        
        # calculate potential
        potential = get_potential_from_atom_data(atom_data, volume_dims, pix_sizes,
                                                 edge_flag=True)
        
        # apply thermal gaussian
        if sample['vibration'] > 0:
            potential = apply_thermal_gaussian(potential, sample['vibration'])
    elif sample['class'] == 'atom':
        # make atom data for single atom
        atom_data = {'coords': np.array([[0,0,0]]), # atom is at origin
                     'element': [sample['atom_element']],
                     'id': list(get_id_from_element([sample['atom_element']]))}
        
        # calculate potential - ensuring single slice
        potential = get_potential_from_atom_data(atom_data, volume_dims,
                                                 [pix_sizes[0],
                                                  sample['slice_thickness']])
        potential.update({'v': np.sum(potential['v'],
                                      axis=2)}) # flatten array along z
        
        # apply thermal gaussian
        if sample['vibration'] > 0:
            potential = apply_thermal_gaussian(potential, sample['vibration'])
    elif sample['class'] == 'protein':
        # import atoms from structure file
        atom_data = read_structure(sample)
        
        if sample['section_thickness'] is not None:
            if sample['rezero_section'] == True:
                rezero_flag = True
                z_only_flag = False
            elif sample['rezero_section'] == 'z':
                rezero_flag = True
                z_only_flag = True
            elif sample['rezero_section'] == False:
                rezero_flag = False
                z_only_flag = False
            atom_data = section_atom_data(atom_data,
                                          sample['section_z'],
                                          sample['section_thickness'],
                                          rezero_flag=rezero_flag,
                                          z_only_flag=z_only_flag)
        
        # calculate potential in small box (avoid storing large arrays)
        protein_box_dims = get_protein_box_dims(atom_data,
                                                pad_size=sample['box_pad_size']) # [A]
        if diagnostics_on:
            if not silent_mode:
                print('...DIAGNOSTICS: protein box has ' +
                      'side length {} A'.format(protein_box_dims[0]))
        if np.any(np.asarray(protein_box_dims) > np.asarray(volume_dims)):
            raise Exception('desired volume is too small for protein')
        potential = get_potential_from_atom_data(atom_data,
                                                 protein_box_dims,
                                                 pix_sizes)
        if sample['solvated']:
            solvent_density = get_shang_sigworth_solvent(atom_data, potential)
        else:
            print('Solvent disabled. Sample is in vacuum.')
            solvent_density = np.zeros_like(potential['v']) # no solvent
        
        # begin diagnostic plot
        if diagnostics_on:
            if not silent_mode:
                print('...DIAGNOSTICS: plotting sample potential and solvent density')
            plt.figure('exit_wave_calc')
            plt.subplot(2,3,1)
            plt.imshow(np.sum(potential['v'], axis=2))
            plt.colorbar()
            plt.title('potential')
            plt.subplot(2,3,2)
            plt.imshow(np.sum(solvent_density, axis=2))
            plt.colorbar()
            plt.title('solvent_density')
        
        # apply thermal gaussian
        if sample['vibration'] > 0:
            potential = apply_thermal_gaussian(potential, sample['vibration'])
            # continue diagnostic plot
            if diagnostics_on:
                plt.subplot(2,3,3)
                plt.imshow(np.sum(potential['v'], axis=2))
                plt.title('potential + vibration')
        
        # eq 4, Shang & Sigworth JSB 2012 - total potential
        potential['v'] = potential['v'] + 3.6*axial_pix_size*solvent_density
        
        # embed potential in larger box
        potential = embed_protein_box(potential, volume_dims)
    elif sample['class'] == 'protein_lattice':
        # get dimensions of lattice
        if sample['lattice_box_size'] is None:
            lattice_dims = volume_dims # [A]
        else:
            lattice_dims = sample['lattice_box_size'] # [A]
        lattice_dims_rcz = np.array([int(lattice_dims[i]/pix_sizes[int(i/2)])
                                     for i in range(3)]) # [pix]
        
        # get atom data
        if atom_data is None:
            atom_data = read_structure(sample)
        
        # get dimensions of protein
        protein_box_dims = get_protein_box_dims(atom_data,
                                                pad_size=sample['box_pad_size']) # [A]
        if sample['lattice_is_2d']:
            protein_box_dims[2] = sample['thickness']
        protein_box_dims_rcz = np.array([int(np.ceil(protein_box_dims[i]/\
                                                     pix_sizes[int(i/2)])) # [pix]
                                         for i in range(3)])
        if np.any(np.asarray(protein_box_dims) > np.asarray(volume_dims)):
            raise Exception('total volume is WAY too small for protein array')
        elif np.any(np.asarray(protein_box_dims) > np.asarray(lattice_dims)):
            raise Exception('desired lattice volume is too small for protein array')
        elif np.any(np.asarray(lattice_dims) > np.asarray(volume_dims)):
            raise Exception('lattice volume cannot exceed total volume')

        # get numer of proteins along each dimension
        protein_counts = [int(i) for i in
                          np.asarray(lattice_dims)/np.asarray(protein_box_dims)]
        # print(protein_counts)
        
        # determine locations where protein boxes will be placed
        slices = np.empty((len(lattice_dims), np.max(protein_counts)), dtype=slice)
        for ax_ind, N in enumerate(protein_counts):
            curr_d = protein_box_dims_rcz[ax_ind]
            if np.mod(N, 2) == 0: # even N
                start_ind_1 = (lattice_dims_rcz[ax_ind] - N*curr_d)/2
            else: # odd N
                start_ind_1 = (lattice_dims_rcz[ax_ind] - N*curr_d)/2
            start_inds = np.arange(start_ind_1, start_ind_1 + N*curr_d, curr_d)
            for box_ind, start_ind in enumerate(start_inds):
                slices[ax_ind, box_ind] = slice(int(start_ind),
                                                int(start_ind) + int(curr_d))
        all_box_locs = [] # complete list (to be populated below)
        for row_ind in range(protein_counts[0]):
            for col_ind in range(protein_counts[1]):
                for dep_ind in range(protein_counts[2]):
                    all_box_locs.append(tuple((slices[0, row_ind],
                                               slices[1, col_ind],
                                               slices[2, dep_ind])))
        print('Created {}/{} boxes.'.format(len(all_box_locs),
                                            np.prod(protein_counts)))

        if sample['solvated']:
            (latt_vol, latt_ax) = make_volume_with_axes(lattice_dims, pix_sizes,
                                                        val=3.6*axial_pix_size)
        else:
            (latt_vol, latt_ax) = make_volume_with_axes(lattice_dims, pix_sizes,
                                                        val=0)
        
        # calculate potential for each box
        if sample['interp_positioning']:
            # make a single potential to spline-interpolate
            print('Beginning stock potential calculation.')
            # calculate stock potential
            stock_potential = get_potential_from_atom_data(atom_data,
                                                           protein_box_dims,
                                                           pix_sizes)
            if sample['solvated']: # calculate solvent density
                solvent_density = get_shang_sigworth_solvent(atom_data,
                                                             stock_potential)
            else:
                print('Solvent disabled. Sample is in vacuum.')
                solvent_density = np.zeros_like(stock_potential['v']) # no solvent
            if sample['vibration'] > 0: # apply thermal gaussian
                stock_potential = apply_thermal_gaussian(stock_potential,
                                                         sample['vibration'])
            # eq 4, Shang & Sigworth JSB 2012 - total potential
            stock_potential['v'] = stock_potential['v'] +\
                3.6*axial_pix_size*solvent_density
            print('Stock potential ready.')
        for box_ind, slc in enumerate(all_box_locs):
            print('Embedding box {}/{}'.format(box_ind + 1, len(all_box_locs)))
            
            # get potential to put in current box
            if sample['random_orientations'] or sample['random_positions']:
                if sample['interp_positioning']: # do spline interp using scipy
                    curr_pot = stock_potential.copy()
                    face_vals = get_face_vals(stock_potential['v'])
                    if sample['random_orientations']:
                        curr_pot['v'] = ndimage.rotate(curr_pot['v'],
                                                       uniform(0, 360),
                                                       axes=(2, 0), reshape=False,
                                                       mode='constant',
                                                       cval=np.mean(face_vals))
                        curr_pot['v'] = ndimage.rotate(curr_pot['v'],
                                                       uniform(0, 360),
                                                       axes=(2, 1), reshape=False,
                                                       mode='constant',
                                                       cval=np.mean(face_vals))
                        curr_pot['v'] = ndimage.rotate(curr_pot['v'],
                                                       uniform(0, 360),
                                                       axes=(1, 0), reshape=False,
                                                       mode='constant',
                                                       cval=np.mean(face_vals))
                    if sample['random_positions']:
                        shift_max = sample['random_position_magnitude']
                        if sample['random_position_magnitude'] < 0:
                            magn = np.abs(shift_max)
                            print('random_position_magnitude cannot be ' +
                                  'negative, changed to {}'.format(magn))
                        curr_pot['v'] = ndimage.shift(curr_pot['v'],
                                                      uniform(-shift_max,
                                                              shift_max, (3)),
                                                      mode='constant',
                                                      cval=np.mean(face_vals))
                else: # calculate potential from scratch
                    # update sample coordinates
                    if sample['random_orientations']: # generate random orientation
                        sample.update({'xyz_angles_deg': uniform(0, 360, (1,3))})
                    if sample['random_positions']: # generate random position
                        sample.update({'xyz_loc': uniform(-100, 100, (1,3))})
                    
                    # update atom data
                    atom_data = read_structure(sample)
                    
                    # calculate potential
                    curr_pot['v'] = get_potential_from_atom_data(atom_data,
                                                                 protein_box_dims,
                                                                 pix_sizes)
                    if sample['solvated']: # calculate solvent density
                        solvent_density = get_shang_sigworth_solvent(atom_data,
                                                                     curr_pot)
                    else: # no solvent
                        print('Solvent disabled. Sample is in vacuum.')
                        solvent_density = np.zeros_like(curr_pot['v'])
                    
                    if sample['vibration'] > 0: # apply thermal gaussian
                        curr_pot = apply_thermal_gaussian(curr_pot,
                                                          sample['vibration'])
                    # eq 4, Shang & Sigworth JSB 2012 - total potential
                    curr_pot['v'] = curr_pot['v'] +\
                        3.6*axial_pix_size*solvent_density
            else:
                curr_pot = stock_potential.copy()
            
            # embed current potential into large volume
            latt_vol[slc] = 0 # remove background first (trust me)
            latt_vol[slc] = latt_vol[slc] + curr_pot['v']
            print('Box embedding complete.')
        
        # begin diagnostic plot
        if diagnostics_on:
            if not silent_mode:
                print('DIAGNOSTICS: plotting single site potential ' +
                      'and lattice potential')
            plt.figure('exit_wave_calc')
            plt.subplot(2,3,1)
            plt.imshow(np.sum(curr_pot['v'], axis=2))
            plt.title('single site potential')
            plt.subplot(2,3,2)
            plt.imshow(np.sum(latt_vol, axis=2))
            plt.title('lattice potential')
        
        # create potential dictionary
        potential = {'v': latt_vol,
                     'axes': latt_ax,
                     'pix_sizes': pix_sizes}
        
        # embed lattice in full volume
        if lattice_dims != volume_dims:
            potential = embed_protein_box(potential, volume_dims)
    
    if sample['class'] == 'test':
        if return_potential:
            print('Note: \'test\' samples do not have a potential.')
        return exit_wave_op
    else:
        if sample['class'] == 'atom':
            psi_op = np.exp(1j*scope['sigma_e']*potential['v'])
            psi_exit = psi_op.copy()
            
            # diagnostic plot
            if diagnostics_on:
                plt.figure()
                plt.suptitle('atom')
                plt.subplot(2,2,1)
                plt.imshow(potential['v'])
                plt.title('potential')
                plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow(np.abs(psi_op))
                plt.title(r'$\left|\psi_{op}\right|$')
                plt.colorbar()
                plt.subplot(2,2,4)
                plt.imshow(np.angle(psi_op))
                plt.title(r'$\operatorname{arg}\left(\psi_{op}\right)$')
                plt.colorbar()
                plt.tight_layout()
        else: # do multislice simulation
            pix_per_slice = sample['slice_thickness']/axial_pix_size
            if not(pix_per_slice.is_integer()):
                raise Exception('slice_thickness not divisible ' +
                                'by axial_pix_size')
            (psi_op,
             psi_exit,
             track_integral) = do_multislice(psi_in, potential, pix_per_slice,
                                             pix_sizes[1]*pix_per_slice,
                                             scope, camera)
        
            # more diagnostics
            if diagnostics_on:
                plt.subplot(2,3,4)
                plt.imshow(np.abs(psi_op))
                plt.colorbar()
                plt.title(r'$\left|\psi_{op}\right|$')
                plt.subplot(2,3,5)
                plt.imshow(np.angle(psi_op))
                plt.colorbar()
                plt.title(r'$\operatorname{arg}\left(\psi_{op}\right)$')
                plt.subplot(2,3,6)
                plt.plot(range(len(track_integral)), track_integral)
                plt.ylim([0, 2])
                plt.xlabel('slice')
                plt.title(r'$\left|\psi_{op}\right|^2$ integral')
                plt.tight_layout()
        
        # create exit wave dictionary
        exit_wave_op = {'psi': psi_op, # refocused wave (for later calcs)
                        'psi_ew': psi_exit, # exit wave (for reference only)
                        'atom_data': atom_data}
        
        if return_potential:
            exit_wave_op.update({'potential': potential})
        return exit_wave_op

def do_multislice(psi_in, potential, pix_per_slice,
                  slice_pix_size, scope, camera):
    print('Beginning multislice calculation...')
    n_slices = np.shape(potential['v'])[2]/pix_per_slice
    if not(n_slices.is_integer()):
        raise Exception('pix_per_slice must yield an integer number of slices')
    
    # make bfp coordinates, antaliasing mask, and calculate propagator
    S = make_bfp_coords_S(camera)
    antialias_mask = make_aperture_mask(S['r'], 0.5)
    # antialias_mask = np.ones_like(S['r'])
    p_fun = get_p_fun(scope['wlen'], slice_pix_size, S['r']**2, antialias_mask)
    
    # propagate through sample
    psi = psi_in*np.ones_like(S['r'])
    track_integral = [np.sum(calc_image_pdf(psi))]
    for slc in range(int(n_slices)):
        psi = propagate_slice(psi,
                              get_t_fun(potential, slc, pix_per_slice,
                                        scope['sigma_e']),
                              p_fun)
        track_integral.append(np.sum(calc_image_pdf(psi)))
        if np.mod(slc,10) == 0:
            print('...Slice {}/{} ready...'.format(slc+1, int(n_slices)))
    print('...All {} slices computed.'.format(int(n_slices)))
    
    psi_exit = psi.copy() # store the exit wave
    
    # return focus to middle of sample
    p_fun_back = get_p_fun(scope['wlen'], -1*slice_pix_size*(n_slices - 1)/2,
                            S['r']**2, antialias_mask)
    psi = propagate_slice(psi, 1, p_fun_back)
    track_integral.append(np.sum(calc_image_pdf(psi)))
    
    track_integral = track_integral/track_integral[0]
    print('...Exit wave calculated.\n' +
          '...Wavefunction integral had ' +
          'mean {:.3f} and std {:.3f}.'.format(np.mean(track_integral),
                                                np.std(track_integral)))
    return (psi, psi_exit, track_integral)

def propagate_slice(psi_in, t_slice, p_fun):
    # two equivalent formulations (numerical difference found to be ~1e-5)
    # return ifft2(fft2(psi_in) * p_fun) * t_slice # eq 6.87, Kirkland
    return ifft2(fft2(psi_in * t_slice,
                      workers=os.cpu_count() - 1) * p_fun,
                 workers=os.cpu_count() - 1) # eq 6.86, Kirkland, preferred

def get_t_fun(potential, slice_ind, pix_per_slice, sigma_e):
    int_pot = np.sum(potential['v'][:, :, slice(int(pix_per_slice*slice_ind),
                                                int(pix_per_slice*slice_ind +\
                                                    pix_per_slice))],
                     axis=2)
    return np.exp(1j*sigma_e*int_pot)

def get_p_fun(e_wlen, dz, s2, mask):
    # eq 6.65, Kirkland
    return fftshift(np.exp(-1j*np.pi*e_wlen*dz*s2) * mask)

def make_aperture_mask(r_grid, frac, force_circle=True):
    if force_circle: # makes circle regardless of aspect ratio
        (_, axes) = make_volume_with_axes([r_grid.shape[0],
                                           r_grid.shape[1], 1],
                                          [1, 1])
        [mesh_c, mesh_r] = np.meshgrid(axes[1], axes[0])
        rad = np.sqrt(mesh_r**2 + mesh_c**2)
        extreme_val = np.min([np.min(rad[0,:]),
                              np.min(rad[:,0])])
        mask = np.ones_like(rad)
        mask[rad > extreme_val*frac] = 0
    else: # makes oval if aspect ratio != 1
        extreme_val = np.min([np.min(r_grid[0,:]),
                              np.min(r_grid[:,0])])
        mask = np.ones_like(r_grid)
        mask[r_grid > extreme_val*frac] = 0
    return mask

def get_protein_box_dims(atom_data, pad_size=10):
    largest_dim = np.max(np.abs(atom_data['coords']))
    protein_dim = 2*(np.ceil(largest_dim) + pad_size) # [A]
    return 3*[int(protein_dim)]

def get_face_vals(cube):
    return np.hstack((cube[0,0,:], cube[0,:,0], cube[:,0,0],
                      cube[-1,-1,:], cube[-1,:,-1], cube[:,-1,-1]))

def embed_protein_box(potential, large_vol_dims):
    print('Embedding potential...')
    # make sure box is big enough
    face_vals = get_face_vals(potential['v'])
    if not(np.std(face_vals - np.mean(face_vals)) == 0):
        raise Exception('protein box too small, try increasing pad_size')
    
    # pad potential to large volume size
    goal_shape = np.array([int(large_vol_dims[i]/\
                               potential['pix_sizes'][int(i/2)])
                           for i in range(3)])
    pad_amount = (goal_shape - np.asarray(potential['v'].shape))/2
    padded_v = np.pad(potential['v'],
                      tuple(tuple([int(np.ceil(pad_amount[i])),
                                   int(np.floor(pad_amount[i]))])
                            for i in range(3)),
                      mode='constant', constant_values=np.mean(face_vals))
    
    potential.update({'small_v': potential['v']}) # store original for ref.
    potential.update({'v': padded_v})
    print('...Potential embedded.')
    return potential

def section_atom_data(atom_data, section_z, section_thickness,
                      rezero_flag=True, z_only_flag=True):
    # make an axial section through atom data
    # rezero_flag will move the sectioned atoms back to mean position (0,0,0)
    # z_only_flag will move only the mean z position back to 0
    lb = section_z - (section_thickness/2) # lower bound [A]
    ub = section_z + (section_thickness/2) # upper bound [A]
    mask = (atom_data['coords'][:, 2] < ub)*(atom_data['coords'][:, 2] > lb)
    atom_data.update({'coords': atom_data['coords'][mask],
                      'element': (np.asarray(atom_data['element'])[mask]).tolist(),
                      'id': atom_data['id'][mask]})
    if rezero_flag:
        atom_data = move_atom_data_to_origin(atom_data,
                                             move_z_only=z_only_flag)
    return atom_data

#%% microscope column
def propagate_op_to_bfp(psi_in):
    return fftshift(fft2(psi_in, workers=os.cpu_count() - 1))

def apply_transfer_func(psi_in, H):
    return psi_in*H

def propagate_fresnel(psi_in, fresnel_ft):
    psi_out_ft = fftshift(fft2(psi_in, workers=os.cpu_count() - 1))*fresnel_ft
    return ifft2(ifftshift(psi_out_ft), workers=os.cpu_count() - 1)

def propagate_bfp_field(psi_in, t_slice, p_fun):
    return fftshift(fft2(ifft2(ifftshift(psi_in*t_slice),
                               workers=os.cpu_count() - 1)*ifftshift(p_fun),
                         workers=os.cpu_count() - 1))

def propagate_bfp_to_ip(psi_bfp):
    return ifft2(ifftshift(psi_bfp), workers=os.cpu_count() - 1)

def calc_image_pdf(psi_ip):
    img_ip = np.abs(psi_ip)**2
    # return img_ip/img_ip.sum() # probability density (integral = 1)
    return img_ip

def TEM(exit_wave_op, laser, scope, camera):
    #transform exit wavefunction to back focal plane (bfp)
    exit_wave_bfp = propagate_op_to_bfp(exit_wave_op)
    
    # apply transfer function
    psi_bfp = apply_transfer_func(exit_wave_bfp, scope['H_bfp'])
    
    # laser phase plate may not be offset from bfp in this version
    if laser['z_offset'] != 0:
        raise Exception('This feature is deprecated in'
                        '{}.'.format(os.path.basename(__file__)))
    
    # transform back focal plane wavefunction to image plane (ip)
    psi_ip = propagate_bfp_to_ip(psi_bfp)

    return psi_ip

#%% image processing
def pdf_to_image(img_pdf, dose, pix_size, noise_flag):
    n_electrons_per_pixel = dose*pix_size**2 # [e/A^2 * A^2/pixel]
    if noise_flag:
        return poisson(n_electrons_per_pixel*img_pdf)
    else:
        return n_electrons_per_pixel*img_pdf

def calc_amplitude_spectrum(im):
    return np.abs(fftshift(fft2(im, workers=os.cpu_count() - 1)))

def downsample_image(im, downsample_factor, mean_flag=False):
    if np.mod(im.shape[0], downsample_factor) != 0:
        raise Exception('downsample_factor should be a divisor of image size')
    elif np.mod(im.shape[1], downsample_factor) != 0:
        raise Exception('downsample_factor should be a divisor of image size')
    if mean_flag:
        im /= downsample_factor**2 # return mean instead of sum of elements
    return np.add.reduceat(np.add.reduceat(im,
                                           np.arange(0, im.shape[0],
                                                     downsample_factor),
                                           axis=0),
                           np.arange(0, im.shape[1],
                                     downsample_factor),
                           axis=1)

def calc_radial_profile(im, r_map=None, n_bins=50, center=None):
    if r_map is None:
        [cols, rows] = np.indices(im.shape)
        if center is None:
            centers = [im.shape[0]/2, im.shape[1]/2]
        r_map = np.sqrt((rows - centers[0])**2 + (cols - centers[1])**2)
    r_axis = np.linspace(r_map.min(), r_map.max(), num=n_bins)
    dr = r_axis[1] - r_axis[0]
    val = lambda r : im[(r_map <= r + dr) & (r_map > r - dr)].mean()
    radial_profile = np.vectorize(val)(r_axis)
    return (radial_profile, r_axis)

#%% diagnostics, timekeeping, & miscellaneous
def toggle_diagnostics(*args):
    global diagnostics_on
    if args[0] == 'on':
        if diagnostics_on is not True:
            diagnostics_on = True
            if not silent_mode:
                print('Diagnostics enabled.')
    elif args[0] == 'off':
        if diagnostics_on is not False:
            diagnostics_on = False
            if not silent_mode:
                print('Diagnostics disabled.')
    else:
        raise Exception('toggle_diagnostics accepts args \'on\' and \'off\'.')

def toggle_silent_mode(*args):
    global silent_mode
    if args[0] == 'on':
        if silent_mode is not True:
            silent_mode = True
    elif args[0] == 'off':
        if silent_mode is not False:
            silent_mode = False
    else:
        raise Exception('toggle_silent_mode accepts args \'on\' and \'off\'.')

def stamp_time(event_str, time_stamps={}):
    if event_str == 'start':
        return {'start': int(time.time())}
    else:
        time_stamps.update({event_str: int(time.time()) - time_stamps['start']})
        return time_stamps

def get_sigma_from_fwhm(fwhm):
    # convert full width at half maximum to standard deviation
    return fwhm/(2*np.sqrt(2*np.log(2)))

def get_sigma_from_1overe(oneovere):
    # convert 1/e width to standard deviation
    return oneovere/np.sqrt(2)

#%% figure plotting functions
# image pdf plot
def plot_image(im, plot_features=default_plot_features):
    if plot_features['plot_asd_log10_TF']:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
        
        # image
        axes[0].title.set_text('Image')
        im0 = axes[0].imshow(im, cmap='gray')
    
        # amplitude spectrum
        asd = calc_amplitude_spectrum(im) # calculate amplitude spectrum
        log10_asd = np.log10(asd)
        color_min = np.quantile(log10_asd, plot_features['asd_quantiles'][0])
        color_max = np.quantile(log10_asd, plot_features['asd_quantiles'][1])
        axes[1].title.set_text(r'$\log_{10}|\mathcal{F}[Image]|$')
        im1 = axes[1].imshow(np.log10(asd), cmap='gray',
                             vmin=color_min, vmax=color_max)
        plt.colorbar(im0, cax=mal(axes[0]).append_axes('right',
                                                       size='5%',
                                                       pad=0.05))
        plt.colorbar(im1, cax=mal(axes[1]).append_axes('right',
                                                       size='5%',
                                                       pad=0.05))
    else:
        plt.figure(figsize=(12,7))
        plt.title('Image')
        plt.imshow(im, cmap='gray')
        plt.colorbar()

# back focal plane plot
def plot_bfp(laser, camera, scope, plot_features=default_plot_features):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

    # magnitude of H_bfp
    axes[0, 0].title.set_text(r'$\left|H_{bfp}\right|$')
    im0 = axes[0, 0].imshow(np.abs(scope['H_bfp']), cmap='gray',
                            extent=2*[-camera['nyquist'], camera['nyquist']])
    plt.colorbar(im0, cax=mal(axes[0, 0]).append_axes('right',
                                                      size='5%',
                                                      pad=0.05))
    # phase of H_bfp
    axes[0, 1].title.set_text(r'$\operatorname{arg}\left(H_{bfp}\right)$')
    im1 = axes[0, 1].imshow(np.angle(scope['H_bfp']), cmap='gray',
                            extent=2*[-camera['nyquist'], camera['nyquist']])
    cbar1 = plt.colorbar(im1,
                         cax=mal(axes[0, 1]).append_axes('right',
                                                         size='5%',
                                                         pad=0.05))
    cbar1.set_label('[rad]')
    cbar1.set_ticks([-np.pi, 0, np.pi])
    cbar1.set_ticklabels(['-$\pi$', '0', '+$\pi$'])

    # zoom of phase of H_bfp
    zf = plot_features['bfp_zoom_factor']
    if np.mod(np.round(zf), 2) == 0:
        zf = np.round(zf) + 1
    else:
        zf = np.round(zf)
    h_slc0 = slice(int(scope['H_bfp'].shape[0]*np.floor(zf/2)/zf),
                   -int(scope['H_bfp'].shape[0]*np.floor(zf/2)/zf))
    h_slc1 = slice(int(scope['H_bfp'].shape[1]*np.floor(zf/2)/zf),
                   -int(scope['H_bfp'].shape[1]*np.floor(zf/2)/zf))
    axes[0, 2].title.set_text(r'$\operatorname{arg}\left(H_{bfp}\right)$' +
                              ' - {}x Zoom'.format(zf))
    im2 = axes[0, 2].imshow(np.angle(scope['H_bfp'][h_slc0, h_slc1]),
                            cmap='gray', extent=2*[-camera['nyquist']/(2*zf),
                                                   camera['nyquist']/(2*zf)])
    cbar2 = plt.colorbar(im2, cax=mal(axes[0, 2]).append_axes('right',
                                                              size='5%',
                                                              pad=0.05))
    cbar2.set_label('[rad]')
    cbar2.set_ticks([-np.pi, 0, np.pi])
    cbar2.set_ticklabels(['-$\pi$', '0', '+$\pi$'])

    if laser['z_offset'] == 0 and laser['eta'].std() != 0: # laser is in bfp
        # gamma_lattice
        im3 = axes[1, 0].imshow(np.angle(np.exp(-1j*scope['gamma_lattice'])),
                                cmap='gray', extent=2*[-camera['nyquist'],
                                                       camera['nyquist']])
        axes[1, 0].title.set_text(r'$\gamma$')
        cbar3 = plt.colorbar(im3, cax=mal(axes[1, 0]).append_axes('right',
                                                                  size='5%',
                                                                  pad=0.05))
        cbar3.set_label('[rad]')
        cbar3.set_ticks([-np.pi, 0, np.pi])
        cbar3.set_ticklabels(['-$\pi$', '0', '+$\pi$'])
        
        # eta
        im4 = axes[1, 1].imshow(np.angle(np.exp(-1j*laser['eta'])),
                                cmap='gray', extent=2*[-camera['nyquist'],
                                                       camera['nyquist']])
        axes[1, 1].title.set_text(r'$\eta$')
        cbar4 = plt.colorbar(im4, cax=mal(axes[1, 1]).append_axes('right',
                                                                  size='5%',
                                                                  pad=0.05))
        cbar4.set_label('[rad]')
        cbar4.set_ticks([-np.pi, 0, np.pi])
        cbar4.set_ticklabels(['-$\pi$', '0', '+$\pi$'])
        
        # gamma_lattice + eta
        im5 = axes[1, 2].imshow(np.angle(np.exp(-1j*(scope['gamma_lattice'] +\
                                                     laser['eta']))),
                                cmap='gray', extent=2*[-camera['nyquist'],
                                                       camera['nyquist']])
        axes[1, 2].title.set_text(r'$\gamma + \eta$')
        cbar5 = plt.colorbar(im5, cax=mal(axes[1, 2]).append_axes('right',
                                                                  size='5%',
                                                                  pad=0.05))
        cbar5.set_label('[rad]')
        cbar5.set_ticks([-np.pi, 0, np.pi])
        cbar5.set_ticklabels(['-$\pi$', '0', '+$\pi$'])
        
    # finish figure
    plt.tight_layout()
    plt.show()

# laser phase plate plot
def plot_lpp(laser, camera, scope, plot_features=default_plot_features):
    if laser['eta'].std() == 0:
        print('Skipped plot_lpp because laser is off')
    else:
        # preliminary calculations
        S = make_bfp_coords_S(camera) # [A^-1]
        Sx_phys = S['x']*scope['wlen']*scope['focal_length']
        Sy_phys = S['y']*scope['wlen']*scope['focal_length']
        phase_full = np.rad2deg(np.angle(np.exp(-1j*laser['eta'])))
        zf = plot_features['lpp_zoom_factor']
        if np.mod(np.round(zf), 2) == 0:
            zf = np.round(zf) + 1
        else:
            zf = np.round(zf)
        p_slc0 = slice(int(phase_full.shape[0]*np.floor(zf/2)/zf),
                       -int(phase_full.shape[0]*np.floor(zf/2)/zf))
        p_slc1 = slice(int(phase_full.shape[1]*np.floor(zf/2)/zf),
                       -int(phase_full.shape[1]*np.floor(zf/2)/zf))
        
        # figure setup
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
        fig.suptitle('Laser Phase Profile ($\eta_{max}$ = ' +
                     '{:.2f}$^o$)'.format(np.rad2deg(laser['eta0'])))
        
        # laser phase
        axes[0].scatter(0, 0, marker='x', c='g')
        im0 = axes[0].imshow(phase_full, cmap='gist_heat_r',
                             vmin=0, vmax=-np.ceil(np.rad2deg(laser['eta0'])),
                             extent=2*[-camera['nyquist'], camera['nyquist']])
        axes[0].title.set_text('Whole BFP')
        axes[0].set_xlabel('[1/$\AA$]')
        cbar0 = plt.colorbar(im0, cax=mal(axes[0]).append_axes('right',
                                                               size='5%',
                                                               pad=0.05))
        cbar0.set_label('[deg]')
        cbar0.set_ticks([0, -int(round(np.rad2deg(laser['eta0'])))])
        cbar0.set_ticklabels(['0', str(int(round(np.rad2deg(laser['eta0']))))])
        
        # laser phase - zoomed
        axes[1].scatter(0,0,marker='x', c='g')
        im1 = axes[1].imshow(phase_full[p_slc0, p_slc1], cmap='gist_heat_r',
                             vmin=0, vmax=-np.ceil(np.rad2deg(laser['eta0'])),
                             extent=[Sx_phys.min()/(2*zf),
                                     Sx_phys.max()/(2*zf),
                                     Sy_phys.min()/(2*zf),
                                     Sy_phys.max()/(2*zf)])
        axes[1].title.set_text('{}x Zoom'.format(zf))
        axes[1].set_xlabel('[$\AA$]')
        cbar1 = plt.colorbar(im1, cax=mal(axes[1]).append_axes('right',
                                                               size='5%',
                                                               pad=0.05))
        cbar1.set_label('[deg]')
        cbar1.set_ticks([0, -int(round(np.rad2deg(laser['eta0'])))])
        cbar1.set_ticklabels(['0', str(int(round(np.rad2deg(laser['eta0']))))])
        
        # finish figure
        plt.tight_layout()
        plt.show()

# contrast transfer function plot
def plot_ctf(img, laser, camera, scope, plot_features=default_plot_features):
    # preliminary calculations
    S = make_bfp_coords_S(camera)
    (ctf, atf) = get_transfer_func(img, scope, camera,
                                   laser, ctf_mode=True)
    (A, xi, nA, nxi) = get_transfer_func(img, scope, camera,
                                         laser, ctf_mode=True,
                                         return_ctf_components=True)
    if plot_features['astig_avg_TF'] or img['astig_z'] != 0:
        f_tot = make_defocus_coord(S, img)
        (ctf_radial, _) = calc_radial_profile(np.real(ctf), r_map=f_tot*S['r'],
                                              n_bins=plot_features['ctf_bins'])
        (atf_radial, _) = calc_radial_profile(np.real(atf), r_map=f_tot*S['r'],
                                              n_bins=plot_features['ctf_bins'])
    else:
        (ctf_radial, _) = calc_radial_profile(np.real(ctf), r_map=S['r'],
                                              n_bins=plot_features['ctf_bins'])
        (atf_radial, _) = calc_radial_profile(np.real(atf), r_map=S['r'],
                                              n_bins=plot_features['ctf_bins'])
    radial_axis = np.linspace(S['r'].min(), S['r'].max(),
                              num=plot_features['ctf_bins'])
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

    # A (magnitude of H_bfp)
    axes[0, 0].title.set_text(r'$A = \left|H_{bfp}\right|$')
    im0 = axes[0, 0].imshow(A, cmap='gray', vmin=0, vmax=1,
                            extent=2*[-camera['nyquist'], camera['nyquist']])
    plt.colorbar(im0, cax=mal(axes[0, 0]).append_axes('right', size='5%',
                                                      pad=0.05))
    
    # xi (phase of H_bfp)
    axes[1, 0].title.set_text(r'$\xi = \operatorname{arg}\left(H_{bfp}\right)$')
    im1 = axes[1, 0].imshow(xi, cmap='gray',
                            extent=2*[-camera['nyquist'], camera['nyquist']])
    cbar1 = plt.colorbar(im1, cax=mal(axes[1, 0]).append_axes('right',
                                                              size='5%',
                                                              pad=0.05))
    cbar1.set_label('[rad]')
    cbar1.set_ticks([xi.min(), xi.max()])
    cbar1.set_ticklabels(['{:.1f}'.format(xi.min()), '{:.1f}'.format(xi.max())])

    # contrast transfer function
    axes[0, 1].title.set_text('CTF')
    im2 = axes[0, 1].imshow(np.real(ctf), cmap='gray', vmin=0, vmax=1,
                            extent=2*[-camera['nyquist'], camera['nyquist']])
    cbar2 = plt.colorbar(im2, cax=mal(axes[0, 1]).append_axes('right',
                                                              size='5%',
                                                              pad=0.05))
    cbar2.set_ticks([0, 1])
    cbar2.set_ticklabels(['0', '1'])

    # amplitude transfer function
    axes[1, 1].title.set_text('ATF')
    im3 = axes[1, 1].imshow(np.real(atf),
                            cmap='gray', vmin=0, vmax=1,
                            extent=2*[-camera['nyquist'], camera['nyquist']])
    cbar3 = plt.colorbar(im3, cax=mal(axes[1, 1]).append_axes('right',
                                                              size='5%',
                                                              pad=0.05))
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(['0', '1'])
    
    # contrast transfer function - radial plot
    axes[0, 2].title.set_text('CTF (radial)')
    axes[0, 2].plot(radial_axis, ctf_radial)
    
    # amplitude transfer function - radial plot
    axes[1, 2].title.set_text('ATF (radial)')
    axes[1, 2].plot(radial_axis, atf_radial)
    
    # finish figure
    plt.tight_layout()
    plt.show()

def save_plot(plot_features, save_name):
    if plot_features['save_plots_TF']:
        if not os.path.isdir(plot_features['save_folder_name']):
            os.mkdir(plot_features['save_folder_name'])
        plt.savefig(os.path.join(plot_features['save_folder_name'],
                                 save_name + '.png'), bbox_inches='tight')