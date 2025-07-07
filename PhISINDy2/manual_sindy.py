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





def apply_features(x, t, params, torch_flag=True):
    """
    Applies the feature candidates to the given data

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        a 2D array/tensor containing the displacement in the first column and the velocity in the second one
    t : numpy.ndarray or torch.Tensor
        an 1D array/tensor containing the discrete time values 
    params : parameters dataclass
        the parameters of the run
    torch_flag : bool
        if True, the type of x, t and return object is torch.Tensor, otherwise numpy.ndarray
    Returns
    -------
    numpy.ndarray or torch.Tensor
        a 2D array/tensor with the two vector field values, velocity as first column and acceleration as second, 
        applying the assumed features on the given input x and t
    
    """
    pol_indeces = list(chain.from_iterable(combinations_with_replacement(range(x.shape[1]), i) for i in range(params.poly_order + 1)))
    if torch_flag:
        return T.column_stack(
            (
                *[x[:, inds].prod(1) for inds in pol_indeces],
                *[T.cos(ph * t) for ph in params.cos_phases], # cosine features
                *[T.sign(x[:, 2]),] * params.y1_sgn_flag, # y1 signum feature
                *[T.sign(x[:, 3]),] * params.y2_sgn_flag, # y2 signum feature
                *[T.log((T.abs(x[:, 2]) + params.fr1["eps"]) / params.fr1["V_star"]),] * params.log_1_fr1, # natural log of abs vel
                *[T.log(params.fr1["c"] + params.fr1["V_star"] / (T.abs(x[:, 2]) + params.fr1["eps"])),] * params.log_2_fr1, # natural log of abs displ
                *[T.log((T.abs(x[:, 3]) + params.fr2["eps"]) / params.fr2["V_star"]),] * params.log_1_fr2, # natural log of abs vel
                *[T.log(params.fr2["c"] + params.fr2["V_star"] / (T.abs(x[:, 3]) + params.fr2["eps"])),] * params.log_2_fr2, # natural log of abs displ
            )
        )
    else:
        return np.column_stack(
            (
                *[x[:, inds].prod(1) for inds in pol_indeces],
                *[np.cos(ph * t) for ph in params.cos_phases], # cosine features
                *[np.sign(x[:, 2]),] * params.y1_sgn_flag, # y1 signum feature
                *[np.sign(x[:, 3]),] * params.y2_sgn_flag, # y2 signum feature
                *[np.log((np.abs(x[:, 2]) + params.fr1["eps"]) / params.fr1["V_star"]),] * params.log_1_fr1, # natural log of abs vel
                *[np.log(params.fr1["c"] + params.fr1["V_star"] / (np.abs(x[:, 2]) + params.fr1["eps"])),] * params.log_2_fr1, # natural log of abs displ
                *[np.log((np.abs(x[:, 3]) + params.fr2["eps"]) / params.fr2["V_star"]),] * params.log_1_fr2, # natural log of abs vel
                *[np.log(params.fr2["c"] + params.fr2["V_star"] / (np.abs(x[:, 3]) + params.fr2["eps"])),] * params.log_2_fr2, # natural log of abs displ
            )
        )



def get_feature_names(params):
    """"
    A function that stores the assumed features as strings

    Parameters
    ----------
    params : parameters dataclass
        the parameters of the run

    Returns
    -------
    list
        the list containing the assumed features as strings
    """
    # TODO: modify the name for our case, take index sample from apply_features
    return [*[f"cos({ph:.1f} t)" for ph in params.cos_phases], 
            *[f"sin({ph:.1f} t)" for ph in params.sin_phases], 
            *["sgn(x)",] * params.x_sgn_flag,
            *["sgn(y)",] * params.y_sgn_flag,
            *["1",] * (params.poly_order >= 0),
            *["x", "y"] * (params.poly_order >= 1),
            *["x^2", "xy", "y^2"] * (params.poly_order >= 2),
            *["x^3", "x^2y", "xy^2", "y^3"] * (params.poly_order >= 3),
            *["x^4", "x^3y", "x^2y^2", "xy^3", "y^4"] * (params.poly_order >= 4),]


class CoeffsDictionary(T.nn.Module):
    """
    A class for initializing, storing, and updating the ksi coefficients
    These coefficients are the linear weights of a neural network
    The class inherits from the torch.nn.Module
    """
    def __init__(self, n_combinations, n_eqs):

        super(CoeffsDictionary, self).__init__()
        self.linear = T.nn.Linear(n_combinations, n_eqs, bias=False)
        # Setting the weights to zeros
        self.linear.weight = T.nn.Parameter(0 * self.linear.weight.clone().detach())
    
    def forward(self, x):
        
        return self.linear(x)
    
def apply_known_physics(x, times, params):
    known_terms_1 = - (params.c1 + params.c2) / params.m1 * x[:, 2] - (params.k1 + params.k2) / params.m1 * x[:, 0] + params.k2 / params.m1 * x[:, 1] + params.c2 / params.m1 * x[:, 3] + params.F1 / params.m1 * T.cos(params.freq1 * times.squeeze(1))
    known_terms_2 = - params.c2 / params.m2 * x[:, 3] - params.k2 / params.m2 * x[:, 1] + params.k2 / params.m2 * x[:, 0] + params.c2 / params.m2 * x[:, 2]
    
    return T.column_stack((known_terms_1, known_terms_2))
                     
def apply_rk4_SparseId(x, coeffs, times, timesteps, params):
    """
    A function that applies the fourth order Runge-Kutta scheme to the given data in order to derive the ones in the following timestep
    During this process the approximate derivatives are used
   
    Parameters
    ----------
    x : torch.Tensor
        a 2D tensor containing the displacement in the first column and the velocity in the second one
    coeffs : CoeffsDictionary object
        the neural network with the sought coefficients as its weights
    times : torch.Tensor
        an 1D tensor containing the discrete time values
    timesteps : torch.Tensor
        an 1D tensor containing the discrete time intervals
    params : parameters dataclass
        the parameters of the run

    Returns
    -------
    torch.Tensor
        Predictions of both displacement and velocity for the next timesteps
    
    """
    
    d1 = apply_features(x, times, params)
    k1 = T.column_stack((x[:, 2:],
                         apply_known_physics(x, times, params) + coeffs(d1) * T.sign(x[:, 2:])
                        ))
    
    k1[:, [0, 2]] = T.where(((T.abs(x[:,2]).unsqueeze(1) <= 1e-3)), 0., k1[:, [0, 2]])
    k1[:, [1, 3]] = T.where(((T.abs(x[:,3]).unsqueeze(1) <= 1e-3)), 0., k1[:, [1, 3]])

    xtemp = x + 0.5 * timesteps * k1
    d2 = apply_features(xtemp, times + 0.5 * timesteps, params)
    k2 = T.column_stack((xtemp[:, 2:],
                         apply_known_physics(xtemp, times, params) + coeffs(d2) * T.sign(xtemp[:, 2:])
                        ))
    
    k2[:, [0, 2]] = T.where(((T.abs(xtemp[:,2]).unsqueeze(1) <= 1e-3)), 0., k2[:, [0, 2]])
    k2[:, [1, 3]] = T.where(((T.abs(xtemp[:,3]).unsqueeze(1) <= 1e-3)), 0., k2[:, [1, 3]])

    xtemp = x + 0.5 * timesteps * k1
    d3 = apply_features(xtemp, times + 0.5 * timesteps, params)
    k3 = T.column_stack((xtemp[:, 2:],
                         apply_known_physics(xtemp, times, params) + coeffs(d3) * T.sign(xtemp[:, 2:])
                        ))
    
    k3[:, [0, 2]] = T.where(((T.abs(xtemp[:,2]).unsqueeze(1) <= 1e-3)), 0., k3[:, [0, 2]])
    k3[:, [1, 3]] = T.where(((T.abs(xtemp[:,3]).unsqueeze(1) <= 1e-3)), 0., k3[:, [1, 3]])

    xtemp = x + timesteps * k3
    d4 = apply_features(xtemp, times + timesteps, params)
    k4 = T.column_stack((xtemp[:, 2:],
                         apply_known_physics(xtemp, times, params) + coeffs(d4) * T.sign(xtemp[:, 2:])
                        ))
    
    k4[:, [0, 2]] = T.where(((T.abs(xtemp[:,2]).unsqueeze(1) <= 1e-3)), 0., k4[:, [0, 2]])
    k4[:, [1, 3]] = T.where(((T.abs(xtemp[:,3]).unsqueeze(1) <= 1e-3)), 0., k4[:, [1, 3]])
    
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timesteps


def scale_torch(unscaled_tensor, params):
    """
    A function that applies a standard scaling to a torch tensor

    Parameters
    ----------
    unscaled_tensor ; torch.tensor
        a 2D tensor we wish to scale
    params : parameters dataclass
        the parameters of the run

    Returns
    -------
    torch.tensor
        the scaled tensor
    """
    return (unscaled_tensor - params.mus) / params.stds


def learn_sparse_model(coeffs, train_set, times, params, lr_reduction=10):
    """"
    A function that calculates which ksi coefficients lead to optimal prediction
    The updating of the coefficients is performed in a deep learning fashion

    Parameters
    ----------
    coeffs : CoeffsDictionary object
        the neural network with the sought coefficients as its weights
    train_set : torch.Tensor
        a 2D tensor containing the displacement in the first column and the velocity in the second one
    times : torch.Tensor
        an 1D tensor containing the discrete time values
    params : parameters dataclass
        the parameters of the run
    lr_reduction : int
        the value that the learning rate is divided by in each training batch

    Returns
    -------
    coeffs : CoeffsDictionary object
        the neural network with the updated/learnt coefficients as its weights
    loss_track : numpy.ndarray
        a 2D array containing the loss for each training batch (row), for each epoch (column)
    """
    # Define optimizer 

    opt_func = optim_all.RAdam(
        coeffs.parameters(), lr=params.lr, weight_decay=params.weightdecay
    )
   
    # Define loss function
    criteria = T.nn.MSELoss()
    # pre-allocate memory for loss_fuction
    loss_track = np.zeros((params.num_iter, params.num_epochs))

    # Training 
    for p in range(params.num_iter):
        for g in range(params.num_epochs):
            coeffs.train()

            opt_func.zero_grad()

            loss_new = T.autograd.Variable(T.tensor([0.0], requires_grad=True))
            weights = 2 ** (-0.5 * T.linspace(0, 0, 1))

            timesteps_i = T.tensor(np.diff(times, axis=0)).float()

            # One forward step predictions
            y_pred = apply_rk4_SparseId(train_set[:-1], coeffs, times[:-1], timesteps=timesteps_i, params=params)

            # One backward step predictions
            y_pred_back = apply_rk4_SparseId(train_set[1:], coeffs, times[1:], timesteps=-timesteps_i, params=params)
           
            if params.scaling:
                y_pred_scaled = scale_torch(y_pred, params)
                y_pred_back_scaled = scale_torch(y_pred_back, params)
                train_set_scaled = scale_torch(train_set, params)
                
                loss_new += criteria(y_pred_scaled, train_set_scaled[1:]) + weights[0] * criteria(
                    y_pred_back_scaled, train_set_scaled[:-1]
                )
            else:
                loss_new += criteria(y_pred, train_set[1:]) + weights[0] * criteria(
                    y_pred_back, train_set[:-1]
                )

            loss_track[p, g] += loss_new.item()
            loss_new.backward()
            opt_func.step()

            sys.stdout.write("\r [Iter %d/%d] [Epoch %d/%d] [Training loss: %.2e] [Learning rate: %.2e]" % (p + 1, params.num_iter, g + 1, params.num_epochs, loss_track[p, g], opt_func.param_groups[0]["lr"],))

        # Removing the coefficients smaller than tol and set gradients w.r.t. them to zero
        # so that they will not be updated in the iterations

        Ws = coeffs.linear.weight.detach().clone()
        Mask_Ws = (Ws.abs() > params.tol_coeffs).type(T.float)
        coeffs.linear.weight = T.nn.Parameter(Ws * Mask_Ws)

        coeffs.linear.weight.register_hook(lambda grad: grad.mul_(Mask_Ws))
        new_lr = opt_func.param_groups[0]["lr"] / lr_reduction
        opt_func = optim_all.RAdam(coeffs.parameters(), lr=new_lr, weight_decay=params.weightdecay)

    return coeffs, loss_track