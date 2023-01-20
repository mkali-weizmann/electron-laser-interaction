#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:24:13 2021

@author: petar
"""
# %%
import tem26 as tem
import numpy as np
from matplotlib import pyplot as plt

# %%
laser = tem.make_laser(peak_phase_deg=90, NA=0.04)
camera = tem.make_camera(rows=2048, cols=2048, pixel_size=1)
scope = tem.make_scope(
    HT=300e3,
    Cc=7.2e7,
    Cs=4.8e7,
    focal_length=20e7,
    energy_spread_std=tem.get_sigma_from_fwhm(0.8),
    coherence_envelopes_on=False,
    dose=100,
)
img = tem.specify_img_features(mean_z=15000, astig_z=0)

# get transfer function
(scope, laser) = tem.get_transfer_func(img, scope, camera, laser, ctf_mode=False, return_ctf_components=False)

# alternative way to get transfer function
(A, xi, _, _) = tem.get_transfer_func(img, scope, camera, laser, ctf_mode=True, return_ctf_components=True)
scope["H_bfp"] = A * np.exp(1j * xi)
laser = tem.get_laser_phase(laser, scope, camera, tem.const)

# samples
test = tem.make_sample(name="test_plane", test_phase=0.1)  # gaussian noise
protein = tem.make_sample(name="myoglobin", thickness=100, solvated=True, slice_thickness=10)  # myoglobin
# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for ind, sample in enumerate([test, protein]):
    psi_in = 1  # plane wave illumination
    ew = tem.get_exit_wave_op(sample, psi_in, scope, camera)  # exit wave
    psi = tem.TEM(ew["psi"], laser, scope, camera)  # image wavefunction
    pdf = tem.calc_image_pdf(psi)  # image probability density
    im = tem.pdf_to_image(pdf, scope["dose"], camera["pixel_size"], noise_flag=True)  # noisy image
    asd = tem.calc_amplitude_spectrum(im)  # amplitude spectrum

    ax[ind, 0].imshow(im[camera["rows"] - 50 : camera["rows"] + 50, camera["cols"] - 50 : camera["cols"] + 50])
    ax[ind, 0].set_title("im: {} (zoom)".format(sample["name"]))
    ax[ind, 1].imshow(np.log10(asd))
    ax[ind, 1].set_title("asd: {}".format(sample["name"]))

# built in functions and plotters
results = tem.do_microscopy(img, protein, laser, camera, scope)
tem.plot_results(results)
tem.plot_bfp(laser, camera, scope)
tem.plot_lpp(laser, camera, scope)
tem.plot_ctf(img, laser, camera, scope)
