from jax import Array
from typing import List, Callable, Any
from gibss.ser import SER
import jax.numpy as jnp
import numpy as np
from flax import struct

import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# simple monitor declares convergence when the 
class Monitor:
    def __init__(self, components, tol=1e-3):
        self.converged = False
        self.components = components
        self.tol = tol

    def monitor(self, new_components):
        diffs = [np.max(np.abs(c1.psi - c2.psi)) for c1, c2 in zip(self.components, new_components)]
        print(f'Max diffs: {diffs}')
        if np.max(diffs) < self.tol:
            self.converged = True
        self.components = new_components

    def report(self):
        print(f'Converged: {self.converged} at tolerance {self.tol}')

@struct.dataclass
class AdditiveModel:
    components: List[Any]
    monitor: Monitor
    iter: int

# Implement an additive model
def additive_model(psi_init: Array, components: List[Any], fit_functions: List[Callable], maxiter=100, monitor=None):
    """Additive mode

    Args:
        psi_init (Array): base value of the linear predictor
        fit_functions (List[Callable]): a list of fit functions, e.g. an SER with a signature (psi, old_component)
        maxiter (int, optional): number of iterations. Defaults to 100.
        monitor (_type_, optional): a function for monitoring convergence. Defaults to None.

    Returns:
        _type_: _description_
    """

    # initialize monitor if not provided
    monitor =  Monitor(components) if monitor is None else monitor

    # subsequent iterations: add and subtract
    psi = psi_init
    for i in range(maxiter-1):
        print(f'Iteration {i}')
        new_components = []
        for j, fun in enumerate(fit_functions):
            print(f'\tUpdating component {j}')
            psi = psi - components[j].psi
            new_components.append(fun(psi, components[j]))
            psi = psi + new_components[j].psi
        monitor.monitor(new_components)
        components = new_components
        if monitor.converged:
            break
     
    monitor.report()
    return AdditiveModel(components, monitor, i+1)
