import sys, os
from csv import DictWriter

import numpy as np
import pandas as pd
import torch as T
from scipy.integrate import solve_ivp

from dataclasses import dataclass

from itertools import combinations_with_replacement
from itertools import chain

import torch_optimizer as optim_all

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
plt.style.use("seaborn-v0_8-whitegrid")

def build_true_model(x, t, params):
    """
    A function that gets the displacement, velocity and time as an input, and returns the true vector field output (velocity and acceleration)

    Parameters
    ----------
    x : numpy.ndarray 
        a 2D array containing the displacement in the first column and the velocity in the second one
    t : numpy.ndarray
        an 1D array containing the discrete time values 
    params : parameters dataclass
        the parameters of the run
    Returns
    -------
    numpy.ndarray
        a 2D array with the two vector field values, velocity as first column and acceleration as second, for the given input x and t
    """
    
    # current normal force:
    # N = N_base + N_dac_z + np.cos(45) * N_dac_x + np.cos(45) * N_dac_y + f(x,y)
    
    Ffricx = params.frx['friction_force_ratio']
    Ffricy = params.fry['friction_force_ratio']

    if params.frx['DR_flag']:
        Ffricx += params.frx["a"] * np.log((np.abs(x[2]) + params.frx["eps"]) / params.frx["V_star"]) \
                + params.frx["b"] * np.log(params.frx["c"] + params.frx["V_star"] / (np.abs(x[2]) + params.frx["eps"])) 
    if params.fry['DR_flag']:
        Ffricy += params.fry["a"] * np.log((np.abs(x[3]) + params.fry["eps"]) / params.fry["V_star"]) \
                + params.fry["b"] * np.log(params.fry["c"] + params.fry["V_star"] / (np.abs(x[3]) + params.fry["eps"])) 
    
    # [x, y, theta, vx, vy, omega]
    derivs = np.array([x[2],
                     x[3],
                     - (params.k1 + params.k2) / params.m1 * x[0] 
                     - (params.c1 + params.c2) / params.m1 * x[2] 
                     + params.k2 / params.m1 * x[1] 
                     + params.c2 / params.m1 * x[3]
                     - Ffricx / params.m1 * np.sign(x[2]) 
                     + params.F1 / params.m1 * np.cos(params.freq1 * t),
                     - params.k2 / params.m2 * x[1] 
                     - params.c2 / params.m2 * x[3] 
                     + params.k2 / params.m2 * x[0] 
                     + params.c2 / params.m2 * x[2] 
                     + params.F2 /params.m2 * np.cos(params.freq2 * (t + params.phi)) 
                     - Ffricy / params.m2 * np.sign(x[3])])

    if (np.abs(x[2]) <= 1e-5) and (np.abs(params.F1 * np.cos(params.freq1 * t) + params.c2 * x[3] + params.k2 * x[1] - (params.k1 + params.k2) * x[0]) <= np.abs(Ffricx)):
        derivs[[0, 2]] = 0.

    if (np.abs(x[3]) <= 1e-5) and (np.abs(params.c2 * x[2] + params.k2 * x[0] - params.k2 * x[1]) <= np.abs(Ffricy)):
        derivs[[1, 3]] = 0.
    
    return derivs


def generate_data(params):
    """
    A function that solves the system's equation and generates the ground truth data based on the defined run parameters

    Parameters
    ----------
    params : parameters dataclass
        the parameters of the run
    Returns
    -------
    numpy.ndarray
        an 1D array with the discrete time instances
    numpy.ndarray
        a 2D array with the ground truth data, displacements as first column and velocities as second
    """
    #Generate (noisy) measurements - Training Data
    ts = np.arange(0, params.timefinal, params.timestep)

    # Solve the equation
    sol = solve_ivp(
        lambda t, x: build_true_model(x, t, params), 
        t_span=[ts[0], ts[-1]], y0=params.x0, t_eval=ts
        )

    return ts, np.transpose(sol.y)

