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


def _ser(fits, prior_variance, X):
    """Logistic SER"""
    alpha = jnp.exp(fits.lbf - jax.scipy.special.logsumexp(fits.lbf))
    lbf_ser = jax.scipy.special.logsumexp(fits.lbf - jnp.log(fits.lbf.size))
    # compute predictions
    # note the predictions do not include the contribution of the intercept
    psi = X.T @ (alpha * fits.beta)
    return SER(psi, alpha, lbf_ser, fits, prior_variance)


def ser(coef_init, X, y, offset, prior_variance, fitter, nullfitter):
    """Logistic SER"""
    nullfit = nullfitter(y, offset)
    fits = fitter(coef_init, X, y, offset, prior_variance, nullfit)
    return _ser(fits, prior_variance, X)