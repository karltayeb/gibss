from typing import Any
from jax import Array
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial

@partial(jax.tree_util.register_dataclass,
         data_fields=['psi', 'alpha', 'lbf_ser', 'prior_variance', 'fits'], meta_fields=[])
@dataclass
class SER:
    psi: Array  # predictions
    alpha: Array  # posterior inclusion probability
    lbf_ser: float
    prior_variance: float
    fits: Any  # parameters

def _ser(fits, X):
    """Logistic SER"""
    alpha = jnp.exp(fits.lbf - jax.scipy.special.logsumexp(fits.lbf))
    lbf_ser = jax.scipy.special.logsumexp(fits.lbf) - jnp.log(fits.lbf.size)
    # compute predictions
    # note the predictions do not include the contribution of the intercept
    psi = X.T @ (alpha * fits.beta)
    prior_variance = fits.prior_variance[0]
    return SER(psi, alpha, lbf_ser, prior_variance, fits)


def ser(coef_init, X, y, offset, fitter, nullfitter):
    """Logistic SER"""
    nullfit = nullfitter(y, offset)
    fits = fitter(coef_init, X, y, offset, nullfit)
    return _ser(fits, X)
