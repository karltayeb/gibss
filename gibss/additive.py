from jax import Array
from typing import List, Callable, Any
import numpy as np
from dataclasses import dataclass
import jax
from functools import partial
from tqdm import tqdm
import jax.numpy as jnp

@dataclass
class AdditiveState:
    """
    This data class helps keep track of the GIBSS outer loop
    """
    diff: float
    tol: int
    converged: bool
    maxiter: int
    iter: int

@dataclass
class AdditiveModel:
    psi: Array
    components: List[Any]
    fit_functions: List[Any]
    state: AdditiveState


@partial(jax.tree_util.register_dataclass,
         data_fields=['psi', 'fit'], meta_fields=[])
@dataclass
class AdditiveComponent:
    psi: Array  # predictions
    fit: Any  # parameters


def make_fitfun(X: np.ndarray, y: np.ndarray, fun: Callable, initfun: Callable, kwargs: dict = dict()) -> Callable:
    """
    Produces a function for fitting an additive component using the provided data and settings. 
    The returned function has a signature compatible with the additive_model interface.

    Args:
        X (np.ndarray): A p x n matrix where p is the number of variables and n is the number of observations.
        y (np.ndarray): An n-dimensional vector of observations corresponding to the rows of X.
        fun (Callable): A function that takes `coef_init`, `X`, `y`, `psi`, and possibly other arguments specified in `kwargs`.
        initfun (Callable): A function with the signature `(X, y, psi, fit) -> Array` used to initialize `coef_init`. 
                            It must be able to handle `fit` as `None`.
        kwargs (dict): A dictionary of additional keyword arguments to pass to `fun`.

    Returns:
        Callable: A function `fitfun(psi: Array, fit: Union[None, Fit]) -> Fit`, used to fit the SER model.
    """
    @jax.jit
    def fitfun(psi, fit):
        coef_init = initfun(X, y, psi, fit)
        fit = fun(
            coef_init = coef_init,
            X = X, y=y, offset = psi,
            **kwargs
        )
        return fit
    return fitfun

def fit_additive_model(fitfuns, maxiter: int=10, tol: float=1e-3, keep_intermediate=False) -> AdditiveModel:
    model = AdditiveModel(0., None, fitfuns, None)
    return update_additive_model(model, maxiter, tol, keep_intermediate)

def update_additive_model(model: AdditiveModel, maxiter=100, tol=1e-3, keep_intermediate=False) -> AdditiveModel:
    # initialization
    psi = 0. if model.psi is None else model.psi
    components = [None for _ in model.fit_functions] if model.components is None else model.components
    fitfuns = model.fit_functions
    
    # for monitoring convergance
    maxiter = int(maxiter)
    tol = float(tol)
    L = len(fitfuns)
    diff = jnp.inf
    state = AdditiveState(diff, tol, False, maxiter, 0)
    intermediate = []
    for i in range(maxiter):
        psi_old = psi
        for l in range(L):
            if components[l] is not None:
                psi = psi - components[l].psi
            components[l] = fitfuns[l](psi, components[l])
            psi = psi + components[l].psi
        if keep_intermediate:
            intermediate.append(jax.tree.map(lambda x: x, components))
        diff = jnp.abs(psi - psi_old).max()
        state = AdditiveState(diff, tol, diff < tol, maxiter, i + 1)
        if diff < tol:
            break

    if keep_intermediate:
        return AdditiveModel(psi, tree_stack(intermediate), model.fit_functions, state)
    return AdditiveModel(psi, components, model.fit_functions, state)
