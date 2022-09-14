import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import geopandas as geopd
import rasterio as rio
import rasterio
import shapely
import pandas as pd
import scipy.optimize
import copy
import numpy.polynomial
import scipy

import shapely.geometry
import shapely.affinity
import rasterio.mask

from constants import *


class HapkeRTM:
    def __init__(self, i = np.pi/6, e = 0, g = np.pi/6, P = P_REG, wl = None):
        """Initialize an instance of the Hapke Radiative Transfer Model, which has tools to calculate ssa from reflectance, and vice versa

        Args:
            i (float, optional): Incidence angle in radians. Defaults to np.pi/6.
            e (float, optional): Emmitance angle in radians. Defaults to 0.
            g (float, optional): Phase angle in radiance. Defaults to np.pi/6.
            P (float or nd.array, optional): Phase function value at angle g.  Defaults to P_REG, regolith at 30 deg phase. Can also be input
            as an array with respect to wavelength
            wl (nd.array, optional): Wavelengths corresponding to each entry in P, if P is a nd.array. Defaults to None.
        """
        self.mu0 = np.cos(i)
        self.mu = np.cos(e)
        self.B = compute_B(g)
        self.P = P
        self.wl = wl

        pass

    def hapke_function(self, ssa):
        """Function R(omega), assuming other parameters are known. This is 
        the REFF version of the function (Hapke equation 10.4)

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: Reflectance
        """
        mu0 = self.mu0
        mu = self.mu
        B = self.B
        P = self.P

        # Compute H-funct
        H = compute_H2(ssa, mu)
        H0 = compute_H2(ssa, mu0)

        #R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)
        R = (ssa/4)  / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)

        return R

    def compute_ssa_from_R(self, R, method = 'lm'):
        """ Compute single scattering albedo given radiance and input. By default, uses Levenberg-Marquardt algorithm as part of 
        scipy.optimize.root. Hybr method (scipy default) not recommended, due to difficulty of finding simualted jacobian.

        Args:
            R (float): Reflectance
            method (str, optional): Root finding method from scipy.optimize.root. Defaults to 'lm'.
        """

        def obj_func(ssa, R):
            Rpred = self.hapke_function(ssa)
            return Rpred - R

        x0 = 0.5

        sol = scipy.optimize.root(fun = obj_func, x0 = 0.5, args = (R), method = method)

        return sol.x
        


