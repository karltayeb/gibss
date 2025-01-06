from jax import Array
from typing import List, Callable, Any
import numpy as np
from dataclasses import dataclass
import jax
from functools import partial
from tqdm import tqdm

class Monitor:
    def __init__(self, components, tol=1e-3):
        self.converged = False
        self.psi = [c.psi for c in components]
        self.tol = tol

    def monitor(self, new_components):
        new_psi = [c.psi for c in new_components]
        diffs = np.array([np.max(np.abs(psi1 - psi2)) for psi1, psi2 in zip(self.psi, new_psi)])
        print(f'Max diffs: {diffs}')
        if np.max(diffs) < self.tol:
            self.converged = True
        self.psi = new_psi

    def report(self):
        print(f'Converged: {self.converged} at tolerance {self.tol}')


# @partial(jax.tree_util.register_dataclass, data_fields=['psi', 'components', 'fit_functions', 'monitor', 'iter'], meta_fields=[])
@dataclass
class AdditiveModel:
    psi: Array
    components: List[Any]
    fit_functions: List[Any]
    monitor: Monitor
    iter: int

# Implement an additive model
def update_additive_model(model: AdditiveModel, maxiter=100):
    """Additive mode

    Args:
        psi_init (Array): base value of the linear predictor
        fit_functions (List[Callable]): a list of fit functions, e.g. an SER with a signature (psi, old_component)
        maxiter (int, optional): number of iterations. Defaults to 100.
        monitor (_type_, optional): a function for monitoring convergence. Defaults to None.

    Returns:
        _type_: _description_
    """
    # subsequent iterations: add and subtract
    psi = model.psi
    components = model.components
    fit_functions = model.fit_functions
    monitor = model.monitor
    for i in tqdm(range(maxiter)):
        for j, fun in tqdm(enumerate(fit_functions), leave=False):
            psi = psi - components[j].psi
            # new_components.append(fun(psi, components[j]))
            components[j] = fun(psi, components[j])
            psi = psi + components[j].psi
        if monitor.converged:
            break
    return AdditiveModel(psi, components, fit_functions, monitor, model.iter + i + 1)
