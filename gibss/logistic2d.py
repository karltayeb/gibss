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


# only put a prior on the effect, the intercept
def nloglik(coef, x, y, offset, prior_variance):
    """Logistic log-likelihood"""
    psi = offset + coef[0] + x * coef[1]
    ll = (
        jnp.sum(y * psi - jnp.logaddexp(0, psi))
        + norm.logpdf(coef[1], jnp.zeros(1), jnp.sqrt(prior_variance)).sum()
    )
    return -ll


def nloglik0(coef, y, offset):
    """Logistic log-likelihood"""
    psi = offset + coef[0]
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi))
    return -ll


@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def fit2d(coef_init, x, y, offset, prior_variance):
    f = lambda coef: nloglik(coef, x, y, offset, prior_variance)
    solver = newton_factory(f)
    return solver(coef_init)


@partial(jax.vmap, in_axes=(0, None, None))
def compute_lbf(fit, fit0, prior_variance):
    lbf = (
        0.5 * jnp.log(2 * jnp.pi)
        + 0.5
        * jnp.log(
            fit0.h[0, 0]
            / ((fit.h[1, 1] + 1 / prior_variance) * fit.h[0, 0] - fit.h[1, 0] ** 2)
        )
        + fit0.f
        - fit.f
    )
    return lbf


SER = namedtuple("SER", ["fits", "fit0", "psi", "lbf", "alpha", "lbf_ser"])


@jax.jit
def logistic_ser(X, y, offset, prior_variance):
    p, n = X.shape

    # fit null model
    f0 = lambda coef: nloglik0(coef, y, offset)
    solver0 = newton_factory(f0)
    fit0 = solver0(jnp.zeros(1))

    # fit for each variable,
    fits = fit2d(jnp.zeros((p, 2)), X, y, offset, prior_variance)
    lbf = compute_lbf(fits, fit0, prior_variance)
    lbf_ser = logsumexp(lbf) - jnp.log(p)
    alpha = jnp.exp(lbf - logsumexp(lbf))
    psi = (fits.x[:, 1] * alpha) @ X
    return SER(fits=fits, fit0=fit0, lbf=lbf, lbf_ser=lbf_ser, alpha=alpha, psi=psi)


def make_logitic2d_laplace_ser(X, prior_variance):
    def f(psi, fit, y):
        return logistic_ser(X, y, psi, prior_variance)

    return f


Susie = namedtuple("SER", ["fixed_effect", "sers", "state"])


def fit_logistic2d_susie(X, y, L, prior_variance=1.0, maxiter=100, tol=1e-5):
    fitfun = Partial(make_logitic2d_laplace_ser(X, prior_variance), y=y)
    fixedfitfun = Partial(make_fixed_fitfun(), y=y)
    fitfuns = [fixedfitfun] + [fitfun for _ in range(L)]
    model = AdditiveModel(None, None, fitfuns, None)
    model = update_additive_model(model, maxiter=maxiter, tol=tol)

    fixed_effect = model.components[0]
    sers = tree_stack(model.components[1:])
    fit = Susie(fixed_effect, sers, model.state)
    return fit


"""
from gibss.logistic2d import fit_logistic2d_susie
import numpy as np

n = 10000
p = 50

X = np.random.binomial(1, 0.1, size=(p, n))
x = X[0]
logit = x - 1
y = np.random.binomial(1, 1/(1 + np.exp(-logit)), size=n)
prior_variance = 1.0
offset = 0.

susiefit = fit_logistic2d_susie(X, y, L=3)
"""

