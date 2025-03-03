from gibss.newton import newton_factory
from functools import partial
from jax.scipy.special import logsumexp
import jax
from gibss.logistic import make_fixed_fitfun
from gibss.additive import AdditiveModel, update_additive_model
from jax.tree_util import Partial
from gibss.utils import tree_stack
import jax.numpy as jnp
from jax.scipy.stats import norm
from collections import namedtuple
import numpy as np


# only put a prior on the effect, the intercept
def nloglik1d(coef, x, y, offset, prior_variance):
    """Logistic log-likelihood"""
    psi = offset + x * coef[0]
    ll = (
        jnp.sum(y * psi - jnp.logaddexp(0, psi))
        + norm.logpdf(coef[0], jnp.zeros(1), jnp.sqrt(prior_variance)).sum()
    )
    return -ll


@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def fit1d(coef_init, x, y, offset, prior_variance):
    nll0 = nloglik1d(np.zeros(1), x, y, offset, prior_variance)
    f = lambda coef: nloglik1d(coef, x, y, offset, prior_variance) - nll0
    solver = newton_factory(f)
    return solver(coef_init)


@partial(jax.vmap, in_axes=(0, None))
def compute_lbf(fit, prior_variance):
    lbf = 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(fit.h[0, 0]) - fit.f
    return lbf


SER = namedtuple("SER", ["fits", "psi", "lbf", "lbf_ser", "alpha"])


@jax.jit
def logistic1d_ser(coef_init, X, y, offset, prior_variance):
    p, n = X.shape
    fits = fit1d(coef_init, X, y, offset, prior_variance)
    lbf = compute_lbf(fits, prior_variance)
    lbf_ser = logsumexp(lbf) - jnp.log(p)
    alpha = jnp.exp(lbf - logsumexp(lbf))
    psi = (fits.x[:, 0] * alpha) @ X
    return SER(fits=fits, psi=psi, lbf=lbf, lbf_ser=lbf_ser, alpha=alpha)


def make_logitic1d_laplace_ser(X, prior_variance):
    def f(psi, fit, y):
        if fit is None:
            p = X.shape[0]
            coef_init = jnp.zeros((p, 1))
        else:
            coef_init = fit.fits.x
        return logistic1d_ser(coef_init, X, y, psi, prior_variance)

    return f


Susie = namedtuple("SER", ["fixed_effect", "sers", "state"])


def fit_logistic1d_susie(X, y, L, prior_variance=1.0, maxiter=100, tol=1e-5):
    fitfun = Partial(make_logitic1d_laplace_ser(X, prior_variance), y=y)
    fixedfitfun = Partial(make_fixed_fitfun(), y=y)
    fitfuns = [fixedfitfun] + [fitfun for _ in range(L)]
    model = AdditiveModel(None, None, fitfuns, None)
    model = update_additive_model(model, maxiter=maxiter, tol=tol)

    fixed_effect = model.components[0]
    sers = tree_stack(model.components[1:])
    fit = Susie(fixed_effect, sers, model.state)
    return fit
