import jax
import jax.numpy as jnp
import numpy as np
from gibss.newton import newton_factory
from jax.scipy.stats import norm
from gibss.logistic import UnivariateRegression
from functools import partial
from jax.tree_util import Partial
from gibss.ser import ser
from gibss.additive import AdditiveModel, update_additive_model
from gibss.utils import tree_stack
from gibss.logistic import SusieFit


def cumlogsumexp(x):
    m = x.max()
    return jnp.log(jnp.cumsum(jnp.exp(x - m))) + m


def partial_loglik(b, x, y, offset):
    ranks = y["ranks"]
    censored = y["censored"]
    psi = b * x + offset
    psi_ord = psi[ranks]
    return jnp.sum((psi - cumlogsumexp(jnp.flip(psi_ord))) * jnp.logical_not(censored))


def nloglik(b, x, y, offset, prior_variance):
    return -partial_loglik(b[0], x, y, offset) - norm.logpdf(
        b[0], 0, jnp.sqrt(prior_variance)
    )


def fit_null(y, offset):
    """Logistic SER"""
    ll0 = partial_loglik(jnp.zeros(1), jnp.zeros_like(y["ranks"]), y, offset)
    return UnivariateRegression(ll0, 0.0, 0.0, 0.0, None)


def hermite_factory(m):
    base_nodes, base_weights = np.polynomial.hermite.hermgauss(m)

    def hermite(
        coef_init,
        x,
        y,
        offset,
        nullfit,
        prior_variance,
        newtonkwargs=dict(maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1),
    ):
        solver = newton_factory(
            Partial(nloglik, x=x, y=y, offset=offset, prior_variance=prior_variance),
            **newtonkwargs,
        )
        state = solver(coef_init)
        params = state.x
        hessian = -state.h

        # set up quadrature
        mu = params[-1]
        sigma = jnp.sqrt(-1 / hessian[-1, -1])
        nodes = base_nodes * jnp.sqrt(2) * sigma + mu
        weights = base_weights / jnp.sqrt(jnp.pi)

        # compute logp
        nloglik_vmap = jax.vmap(nloglik, (0, None, None, None, None))
        y = -nloglik_vmap(nodes[:, None], x, y, offset, prior_variance)
        logp = jax.scipy.special.logsumexp(
            y - norm.logpdf(nodes, loc=mu, scale=sigma) + jnp.log(weights)
        )
        lbf = logp - nullfit.logp

        # compute posterior mean of effect
        y2 = nodes * jnp.exp(y - logp)
        beta = jnp.sum(y2 / norm.pdf(nodes, loc=mu, scale=sigma) * weights)
        return UnivariateRegression(logp, lbf, beta, prior_variance, state)

    hermite_jit = jax.jit(hermite)
    return hermite_jit


@partial(jax.jit, static_argnames=["m"])
def cox_ser_hermite(
    coef_init,
    X,
    y,
    offset,
    prior_variance,
    m,
    newtonkwargs=dict(maxiter=50, tol=0.1, alpha=0.1, gamma=0.0),
):
    vhermite = jax.vmap(
        Partial(
            hermite_factory(m), prior_variance=prior_variance, newtonkwargs=newtonkwargs
        ),
        in_axes=(0, 0, None, None, None),
    )
    return ser(coef_init, X, y, offset, vhermite, fit_null)


def make_cox_hermite_ser(
    X,
    y,
    prior_variance,
    m=1,
    newtonkwargs=dict(maxiter=50, tol=0.1, alpha=0.1, gamma=0.0),
):
    def f(psi, fit):
        if fit is None:
            coef = jnp.zeros((X.shape[0], 1))
        else:
            coef = fit.fits.state.x
        return cox_ser_hermite(coef, X, y, psi, prior_variance, m, newtonkwargs)

    return f


def fit_cox_hermite_susie(X, y, L, prior_variance=1.0, m=1, maxiter=100, tol=1e-5):
    fitfun = make_cox_hermite_ser(X, y, prior_variance, m)
    fitfuns = [fitfun for _ in range(L)]
    model = AdditiveModel(None, None, fitfuns, None)
    model = update_additive_model(model, maxiter=maxiter, tol=tol)
    sers = tree_stack(model.components)
    fit = SusieFit(None, sers, model.state)
    return fit
