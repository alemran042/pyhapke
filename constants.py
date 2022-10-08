import numpy as np

# CONSTANTS #################

e = 0
i = np.pi/6
g = np.pi/6

# Phase
P_REG = 0.8098 #Calculated from Legendre polynomials, b = -0.4 c = 0.25
# From Pascuzzo et al. 2022
(b,c) = (0.5,0.8)
P_ICE = (1-c) * (1 - b**2) / (1 - 2 * b * np.cos(g) + b**2) ** (3/2) + c * (1 - b**2) / (1 + 2 * b * np.cos(g) + b**2)**(3/2)

# Density (g/cm^3)
RHO_ICE = 0.917
RHO_REG = 1.5 # From Lunar Sourcebook

# Diameters

D_ICE = 100
D_REG = 100

PEAK_IDX = 128 # Index of 0.7 microns in the OC dataset