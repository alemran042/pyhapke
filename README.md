# pyhapke
## Created by Jordan Ando

pyhapke is an implementation of Hapke's radiative transfer model (Hapke 1981), which is commonly used to model planetary surfaces. This model can be used to relate reflectance to single-scattering albedo of various materials. This package was created as part of the research for the author's PhD in Earth and Planetary Sciences, advised by Shuai Li, with the intent of simulating albedo variations in lunar polar ice. 

This package is built around the HapkeRTM class, which takes input parameters of incidence, emmittance, and phase angles, as well as phase, and can be used to compute either single scattering albedo or reflectance, given the other. Included constants consist primarily of relevant physical properties of lunar regolith and ice, as well as standard viewing geometries. Each verson of Hapke's model, for bidirectional reflectance, reflectance factor, and radiance factor, is included.

To initialize the pyhapke package, use

import pyhapke

pyhapke is designed to run with numpy and scipy as its primary dependencies, which are commonly found in other scientific Python applications, and can be easily installed through Anaconda or pip.

This software is licensed under the MIT license.
