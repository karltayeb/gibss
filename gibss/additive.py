from jax import Array
from typing import List, Callable, Any
import numpy as np
from dataclasses import dataclass
import jax
from functools import partial

class Monitor:
    def __init__(self, components, tol=1e-3):
        self.converged = False
        self.psi = [c.psi for c in components]
        self.tol = tol

    def monitor(self, new_components):
        new_psi = [c.psi for c in new_components]
        diffs = [np.max(np.abs(psi1 - psi2)) for psi1, psi2 in zip(self.psi, new_psi)]
        print(f'Max diffs: {diffs}')
        if np.max(diffs) < self.tol:
            self.converged = True
        self.psi = new_psi

    def report(self):
        print(f'Converged: {self.converged} at tolerance {self.tol}')


@partial(jax.tree_util.register_dataclass, data_fields=['components', 'monitor', 'iter'], meta_fields=[])
@dataclass
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
