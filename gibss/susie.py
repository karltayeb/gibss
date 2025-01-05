from dataclasses import dataclass
from typing import Any, Callable, List
from functools import partial
import numpy as np
from scipy import sparse
from gibss.gibss import gibss
import jax
import jax.numpy as jnp
from jax import Array
from gibss.credible_sets import compute_cs, CS

@partial(jax.tree_util.register_dataclass,
         data_fields=['tol', 'converged', 'maxiter', 'iter'], meta_fields=[])
@dataclass
class AdditiveState:
    """
    This data class helps keep track of the GIBSS outer loop
    """
    tol: int
    converged: bool
    maxiter: int
    iter: int


def check_not_converged(val):
    """
    check convergence in the while loop for jax.lax.while
    """
    state, components = val
    return jax.lax.cond(
            state.converged | (state.iter >= state.maxiter),
            lambda: False, # stop the while loop
            lambda: True # stay in the while loop
    )

@partial(jax.tree_util.register_dataclass,
         data_fields=['psi', 'fit'], meta_fields=[])
@dataclass
class AdditiveComponent:
    psi: Array  # predictions
    fit: Any  # parameters

def ensure_dense_and_float(matrix):
    """
    X = np.random.binomial(1, np.ones((5, 5))*0.5)
    Xsp = sparse.csr_matrix(X)
    ensure_dense_and_float(Xsp)

    y = np.random.binomial(1, np.ones(5)*0.5)
    ensure_dense_and_float(y)
    """
    # Check if the input is a sparse matrix
    if sparse.issparse(matrix):
        # Convert sparse matrix to a dense array
        matrix = matrix.toarray()
        # Provide a message to the user
        print("Input is a sparse matrix. Converting to a dense array.")
    
    # Ensure the matrix is a numpy array (in case it's a list or other type)
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a sparse matrix or a numpy array.")
    
    # Ensure the matrix is of float type
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(float)
        print("Converting matrix to float type.")
    return matrix


def make_ser_fitfun(X: np.ndarray, y: np.ndarray, serfun: Callable, initfun: Callable, serkwargs: dict) -> Callable:
    @jax.jit
    def serfitfun(psi):
        coef_init = initfun(X, y, psi)
        return serfun(
            coef_init = coef_init,
            X = X, y=y, offset = psi,
            **serkwargs
        )

    @jax.jit
    def serfitfun2(psi, fit):
        coef_init = fit.fits.state.x
        return serfun(
            coef_init = coef_init,
            X = X, y=y, offset = psi,
            **serkwargs
        )
    return serfitfun, serfitfun2


def fit_susie_jax(X: np.ndarray, y: np.ndarray, L: int, serfun: Callable, initfun: Callable, serkwargs: dict, tol=1e-3, maxiter=10) -> Any:
    # initialization
    X = ensure_dense_and_float(X)
    y = ensure_dense_and_float(y)
    fitfun = make_ser_fitfun(X, y, serfun, initfun, serkwargs)
    psi_init = jnp.zeros_like(y).astype(float)

    # forward selection
    def f1(psi, x):
        """
        forward selection scan
        """
        serfit = fitfun(psi)
        psi2 = psi + serfit.psi
        return psi2, serfit
    psi, components = jax.lax.scan(f1, psi_init, np.arange(L))

    # run gibss
    def f2(psi, serfit):
        """
        subsequent iterations need to remove predictions first
        """
        psi2 = psi - serfit.psi
        serfit2 = fitfun(psi2)
        psi3 = psi2 + serfit2.psi
        return psi3, serfit2
    
    def update_sers(val):
        state, (psi1, components1) = val
        psi2, components2 = jax.lax.scan(f2, psi1, components1)

        # update the optimization state
        diff = jnp.abs(psi2 - psi1).max()
        state2 = AdditiveState(state.tol, diff < state.tol, state.maxiter, state.iter + 1)
        return state2, (psi2, components2)

    state = AdditiveState(tol, False, maxiter, 1) # forward initialization is 1 iter
    init_val = (state, (psi, components))
    final_state, (final_psi, final_components) = jax.lax.while_loop(check_not_converged, update_sers, init_val)
    return final_components, final_state


def tree_stack(trees):
    # https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
    
def fit_susie(X: np.ndarray, y: np.ndarray, serfun: Callable, initfun: Callable, serkwargs: dict, L: int=5, tol: float=1e-3, maxiter: int=10, warm_init: bool=False, cleanup: bool = False, keep_intermediate=False) -> Any:
    # get the right types
    X = ensure_dense_and_float(X)
    y = ensure_dense_and_float(y)
    L = int(L)
    maxiter = int(maxiter)
    tol = float(tol)

    # initialization
    fitfun, fitfun2 = make_ser_fitfun(X, y, serfun, initfun, serkwargs)
    psi_init = jnp.zeros_like(y).astype(float)
    psi = jnp.zeros_like(y)
    components = []
    intermediate = []
    # forard selection
    for l in range(L):
        component = fitfun(psi)
        psi = psi + component.psi
        components.append(component)
    state = AdditiveState(tol, False, maxiter, 1)
    if keep_intermediate:
        intermediate.append(tree_stack(components))

    # iterative stepwise selection
    i = 0
    diff = jnp.inf
    for i in range(1, maxiter):
        psi_old = psi
        for l in range(L):
            psi = psi - components[l].psi
            components[l] = fitfun2(psi, components[l]) if warm_init else fitfun(psi)
            psi = psi + components[l].psi
        if keep_intermediate:
            intermediate.append(tree_stack(components))
        # check convergence
        diff = jnp.abs(psi - psi_old).max()
        if diff < tol:
            break
    state = AdditiveState(tol, diff < tol, maxiter, i + 1)
    fit = tree_stack(components)

    if keep_intermediate:
        return tree_stack(intermediate), state

    # optionally, format resutlts into R-friendly format
    if cleanup:
        fit = cleanup_susie_fit(fit, state)
    return fit, state

def fit_additive(fitfuns: List[Callable], tol: float=1e-3, maxiter: int=10, keep_intermediate=False) -> Any:
    maxiter = int(maxiter)
    tol = float(tol)
    L = len(fitfuns)
    # initialization
    psi = 0.
    components = []
    intermediate = []
    # forard selection
    for l in range(L):
        component = fitfuns[l](psi, None)
        components.append(component)
        psi = psi + component.psi
    if keep_intermediate:
        intermediate.append(jax.tree.map(lambda x: x, components)) # makes a copy I think? Return all the same instance if we don't do this.
    state = AdditiveState(tol, False, maxiter, 1)

    # iterative stepwise selection
    i = 0
    diff = jnp.inf
    for i in range(1, maxiter):
        psi_old = psi
        for l in range(L):
            psi = psi - components[l].psi
            components[l] = fitfuns[l](psi, components[l])
            psi = psi + components[l].psi
        if keep_intermediate:
            intermediate.append(jax.tree.map(lambda x: x, components))
        diff = jnp.abs(psi - psi_old).max()
        state = AdditiveState(tol, diff < tol, maxiter, i + 1)
        if diff < tol:
            break

    if keep_intermediate:
        return tree_stack(intermediate), state

    return components, state

@dataclass
class SuSiERes:
    alpha: np.ndarray
    lbf: np.ndarray
    lbf_ser: np.ndarray
    prior_variance: np.ndarray
    effect: np.ndarray
    outer_loop_state: AdditiveState
    
    @property
    def cs(self):
        return [compute_cs(a) for a in self.alpha]

def cleanup_susie_fit(fit, state):
    return SuSiERes(
        np.array(fit.alpha),
        np.array(fit.fits.lbf),
        np.array(fit.lbf_ser),
        np.array(fit.prior_variance),
        np.array(fit.fits.beta),
        state
    )
