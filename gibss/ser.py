from typing import Any
from jax import Array
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial

@partial(jax.tree_util.register_dataclass,
         data_fields=['psi', 'alpha', 'lbf_ser', 'fits', 'prior_variance'], meta_fields=[])
@dataclass
class SER:
    psi: Array  # predictions
    alpha: Array  # posterior inclusion probability
    lbf_ser: float
    fits: Any  # parameters
    prior_variance: float

# @partial(jax.jit, static_argnames=['fitter', 'nullfitter'])
def ser(coef_init, X, y, offset, prior_variance, fitter, nullfitter):
    """Logistic SER"""
    # fit logistic regression for each column of X
    x0 = jnp.zeros_like(y)
    nullfit = nullfitter(y, offset)
    fits = fitter(coef_init, X, y, offset, prior_variance, nullfit)
    alpha = jnp.exp(fits.lbf - jax.scipy.special.logsumexp(fits.lbf))
    lbf_ser = jax.scipy.special.logsumexp(fits.lbf - jnp.log(X.shape[0]))
    # compute predictions
    # note the predictions do not include the contribution of the intercept
    psi = X.T @ (alpha * fits.beta)
    
    # return coef
    return SER(psi, alpha, lbf_ser, fits, prior_variance)