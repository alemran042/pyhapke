import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy

from .constants import *
from .hapkeFuncs import *

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

    def hapke_function_REFF(self, ssa):
        """Function R(omega), assuming other parameters are known. This is 
        the REFF version of the function (Hapke equation 10.4), computing the reflectance factor (reflectance coefficient)

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: Reflectance factor
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
    
    def hapke_function_RADF(self, ssa):
        """Function R(omega), assuming other parameters are known. This is 
        the RADF version of the function (Hapke equation 10.5), which computes I/F radiance.

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: I/F radiance
        """
        mu0 = self.mu0
        mu = self.mu
        B = self.B
        P = self.P

        # Compute H-funct
        H = compute_H2(ssa, mu)
        H0 = compute_H2(ssa, mu0)

        #R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)
        R = (ssa/4) * mu0/ (mu0 + mu) * ((1 + B) * P + H * H0 - 1)

        return R


    def hapke_function_BDRF(self, ssa):
        """Function R(omega), assuming other parameters are known. This is 
        the BDRF version of the function (Hapke equation 10.5), computing Bi-Directional Reflectance

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: Bidirectional reflectance
        """
        mu0 = self.mu0
        mu = self.mu
        B = self.B
        P = self.P

        # Compute H-funct
        H = compute_H2(ssa, mu)
        H0 = compute_H2(ssa, mu0)

        #R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)
        R = (ssa/4 / np.pi) / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)

    def compute_ssa_from_R(self, R, method = 'lm', model = "REFF"):
        """Compute single scattering albedo given radiance and input. By default, uses Levenberg-Marquardt algorithm as part of 
        scipy.optimize.root. Hybr method (scipy default) not recommended, due to difficulty of finding simualted jacobian.

        Args:
            R (float or nd.array): Reflectance
            method (str, optional): Root finding method from scipy.optimize.root. Defaults to 'lm'.
            model (function, optional): Version of Hapke's model to compute ssa from. Defaults to hapke_function_REFF.
        """

        if model == "REFF":
            def obj_func(ssa, R):
                Rpred = self.hapke_function_REFF(ssa)
                return Rpred - R
        elif model == "BDRF":
            def obj_func(ssa, R):
                Rpred = self.hapke_function_BDRF(ssa)
                return Rpred - R
        elif model == "RADF":
            def obj_func(ssa, R):
                Rpred = self.hapke_function_RADF(ssa)
                return Rpred - R

        if type(R) is np.ndarray:
            x0 = np.ones_like(R) * 0.5
        else:
            x0 = 0.5

        sol = scipy.optimize.root(fun = obj_func, x0 = x0, args = (R), method = method)

        return sol.x
        


