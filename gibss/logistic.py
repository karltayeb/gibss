# Implement Logistic SER using JAX
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
import jax
from jax import Array
import numpy as np
from functools import partial
from gibss.ser import ser, _ser
from jax.tree_util import Partial
from typing import Any
from dataclasses import dataclass
from gibss.newton import newton_factory
from gibss.gd_backtracking import gd_factory
from gibss.gibss import gibss
from gibss.logisticprofile import UnivariateRegression
from gibss.ser import SER
from gibss.additive import fit_additive_model, AdditiveComponent
from gibss.additive import make_fitfun
from gibss.utils import tree_stack
from jax.tree_util import tree_map
import warnings
from gibss.additive import AdditiveState
from gibss.credible_sets import compute_cs
from gibss.utils import ensure_dense_and_float

# @jax.jit
# def nloglik_mle(coef, x, y, offset):
#     """Logistic log-likelihood"""
#     psi = offset + (x @ coef)
#     ll = jnp.sum(y * psi - jnp.logaddexp(0, psi))
#     return -ll

# nloglik_mle_hess = jax.hessian(nloglik_mle)
# nloglik_mle_vmap = jax.vmap(nloglik_mle, in_axes=(0, None, None, None))

@jax.jit
def nloglik(coef, x, y, offset, prior_variance=1.):
    """Logistic log-likelihood"""
    psi = offset + (x @ coef)
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi)) + norm.logpdf(coef, 0, jnp.sqrt(prior_variance)).sum()
    return -ll

@jax.jit
def nloglik1d(coef, x, y, offset, prior_variance=1.):
    """Logistic log-likelihood"""
    psi = offset + (x * coef[0])
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi)) + norm.logpdf(coef[0], 0, jnp.sqrt(prior_variance))
    return -ll

def fit_logistic_nd(coef_init, X, y, offset, prior_variance=1e6, newtonkwargs=dict(tol=1e-3, maxiter=500, alpha=0.8, gamma=-0.1)):
    # estimate the MLE
    fun = partial(nloglik, x=X, y=y, offset=offset, prior_variance=prior_variance)
    newton_solver = newton_factory(fun, **newtonkwargs)
    res = newton_solver(coef_init)
    psi = X @ res.x
    return AdditiveComponent(psi, res)

nloglik_vmap = jax.vmap(nloglik, in_axes=(0, None, None, None, None))
nloglik1d_vmap = jax.vmap(nloglik1d, in_axes=(0, None, None, None, None))

def fit_logistic_1d(coef_init, x, y, offset, prior_variance, newtonkwargs = dict(maxiter=50, tol=1e-3, alpha=0.5, gamma=0.0)):
    fun = Partial(nloglik1d, x=x, y=y, offset=offset, prior_variance=prior_variance)
    newton_solver = newton_factory(fun, **newtonkwargs)
    return newton_solver(coef_init)

# gauss-hermite quadrature nodes and weights
def hermite_factory(m):
    base_nodes, base_weights = np.polynomial.hermite.hermgauss(m)
    def hermite(coef_init, x, y, offset, prior_variance, newtonkwargs = dict(maxiter=50, tol=1e-3, alpha=0.5, gamma=0.0)):
        state = fit_logistic_1d(coef_init, x, y, offset, prior_variance, newtonkwargs)
        params = state.x
        hessian = -state.h

        # set up quadrature
        mu = params[0]
        sigma = jnp.sqrt(-1/hessian[0,0])
        nodes = base_nodes * jnp.sqrt(2) * sigma + mu
        weights = base_weights / jnp.sqrt(jnp.pi)

        # compute logp
        ll = -nloglik1d_vmap(jnp.atleast_2d(nodes).T, x, y, offset, prior_variance)
        logp = jax.scipy.special.logsumexp(ll - norm.logpdf(nodes, loc=mu, scale=sigma) + jnp.log(weights)) 
        lbf = logp + nloglik1d(jnp.zeros(1), x, y, offset, 1/jnp.sqrt(2 * jnp.pi)) # MAGIC NUMBER makes the logN(0, 0, 1/(2 pi)) = 0

        # compute posterior mean of effect
        y2 = nodes * jnp.exp(ll - logp)
        beta = jnp.sum(y2/norm.pdf(nodes, loc=mu, scale=sigma) * weights)
        return UnivariateRegression(logp, lbf, beta, prior_variance, state)
    
    hermite_jit = jax.jit(hermite)
    return hermite_jit

@partial(jax.jit, static_argnames=['m'])
def logistic_ser_hermite(coef_init, X, y, offset, m=1, prior_variance=1.0, newtonkwargs=dict()):
    vhermite = jax.vmap(Partial(hermite_factory(m), y=y, offset=offset, prior_variance=prior_variance, newtonkwargs=newtonkwargs), in_axes=(0, 0))
    fits = vhermite(coef_init, X)
    logp = -fits.state.f
    ll0 = jnp.sum(y * offset - jnp.logaddexp(0, offset))
    lbf = -fits.state.f - ll0
    alpha = jnp.exp(lbf - jax.scipy.special.logsumexp(lbf))
    psi = (alpha * fits.beta) @ X
    lbf_ser = jax.scipy.special.logsumexp(lbf) - jnp.log(lbf.size)
    return SER(psi, alpha, lbf_ser, prior_variance, fits)

# gauss-hermite quadrature nodes and weights
def hermite_grid_factory(m):
    base_nodes, base_weights = np.polynomial.hermite.hermgauss(m)
    def hermite_grid(coef_init, x, y, offset, prior_variance_grid, newtonkwargs=dict(maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1)):
        def hermite_inner(coef_init, prior_variance):
            state = fit_logistic_1d(coef_init, x, y, offset, prior_variance, newtonkwargs)
            params = state.x
            hessian = -state.h
        
            # set up quadrature
            mu = params[0]
            sigma = jnp.sqrt(-1/hessian[0,0])
            nodes = base_nodes * jnp.sqrt(2) * sigma + mu
            weights = base_weights / jnp.sqrt(jnp.pi)

            # compute logp
            ll = -nloglik1d_vmap(jnp.atleast_2d(nodes).T, x, y, offset, prior_variance)
            logp = jax.scipy.special.logsumexp(ll - norm.logpdf(nodes, loc=mu, scale=sigma) + jnp.log(weights)) 
            lbf = logp + nloglik1d(jnp.zeros(1), x, y, offset, 1/jnp.sqrt(2 * jnp.pi)) # MAGIC NUMBER makes the logN(0, 0, 1/(2 pi)) = 0

            # compute posterior mean of effect
            y2 = nodes * jnp.exp(ll - logp)
            beta = jnp.sum(y2/norm.pdf(nodes, loc=mu, scale=sigma) * weights)
            return state.x, UnivariateRegression(logp, lbf, beta, prior_variance, state)
        res = jax.lax.scan(hermite_inner, coef_init, prior_variance_grid)
        return res[1]
    return jax.jit(hermite_grid)

@partial(jax.jit, static_argnames=['m'])
def logistic_ser_hermite_grid(coef_init, X, y, offset, m=1, prior_variance_grid=jnp.array([1e-6, 1e-4, 1e-2, 1e-1, 1., 10., 100.]), newtonkwargs=dict()):
    # fit at all grid points
    vhermite = jax.vmap(Partial(hermite_grid_factory(m), y=y, offset=offset, prior_variance_grid=prior_variance_grid, newtonkwargs=newtonkwargs), in_axes=(0, 0))
    gridfits = vhermite(coef_init, X)

    # select the value of the prior variance that maximizes the marginal evidence for the SER
    best_prior_variance = jax.scipy.special.logsumexp(gridfits.lbf, axis=0).argmax()
    fits = jax.tree.map(lambda x: x[:, best_prior_variance], gridfits)

    # do SER
    logp = -fits.state.f
    ll0 = jnp.sum(y * offset - jnp.logaddexp(0, offset))
    lbf = -fits.state.f - ll0
    alpha = jnp.exp(lbf - jax.scipy.special.logsumexp(lbf))
    psi = (alpha * fits.beta) @ X
    lbf_ser = jax.scipy.special.logsumexp(lbf) - jnp.log(lbf.size)
    return SER(psi, alpha, lbf_ser, prior_variance_grid[best_prior_variance], gridfits)


def make_fixed_fitfun(X, y, kwargs=dict()):
    kwargs2 = dict(
        prior_variance = float(1e6),
        newtonkwargs=dict(tol=1e-3, maxiter=500, alpha=0.8, gamma=-0.1)
    )
    kwargs2.update(kwargs)

    @jax.jit
    def fitfun(psi, old_fit):
        # TODO: take Z as an argument
        coef_init = jnp.zeros(1) 
        Z = np.ones((y.size, 1))

        # Fit fixed effects
        return fit_logistic_nd(
            coef_init = coef_init,
            X = Z, y = y, offset = psi,
            **kwargs2
        )
    return fitfun

def make_logistic_hermite_grid_fitfun(X, y, kwargs=dict()):
    kwargs2 = dict(
        prior_variance_grid = jnp.array([1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1., 2., 4.]),
        newtonkwargs=dict(tol=1e-3, maxiter=5, alpha=0.8, gamma=-0.1)
    )
    kwargs2.update(kwargs)

    @jax.jit
    def fitfun(psi, old_fit):
        coef_init = np.zeros((X.shape[0], 1))
        return logistic_ser_hermite_grid(
            coef_init = coef_init,
            X = X, y=y, offset = psi,
            **kwargs2
        )
    return fitfun



@dataclass
class SusieRes:
    fixed_effects: AdditiveComponent
    sers: SER
    state: AdditiveState

def logistic_ser_hermite_init(X, y, psi, fit=None, warm=False):
    """
    If fit is None or warm=False initialize with zeros
    Otherwise, initialize with with current effect estimates from fit.
    """
    if ((fit is None) or (not warm)): 
        coef_init = jnp.zeros((X.shape[0], 1))
    else:   
        coef_init = fit.fits.state.x
    return coef_init


from collections import namedtuple
SusieFit = namedtuple('SusieFit', ['fixed_effect', 'sers', 'state'])
SusieSummary = namedtuple('SusieSummary', [
    'fixed_effects', 'alpha', 'lbf', 'beta', 'prior_variance', 'lbf_ser', 'credible_sets'])


def fit_logistic_susie(X, y, L=5, method='hermite', warm=True, serkwargs=dict(), **kwargs):
    fixedfitfun = make_fixed_fitfun(X, y)
    match method:   
        case 'hermite':
            defaultserkwargs  = dict(
                prior_variance = float(10.),
                newtonkwargs=dict(tol=1e-2, maxiter=5, alpha=0.2, gamma=-0.1)
            )
            defaultserkwargs.update(serkwargs)
            fitfun = make_fitfun(X, y, logistic_ser_hermite, Partial(logistic_ser_hermite_init, warm=warm), defaultserkwargs)
        case 'hermite_eb':
            defaultserkwargs = dict(
                prior_variance_grid = jnp.array([1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1., 2., 4., 8.]),
                newtonkwargs=dict(tol=1e-2, maxiter=20, alpha=0.8, gamma=-0.1)
            )
            defaultserkwargs.update(serkwargs)
            fitfun = make_fitfun(X, y, logistic_ser_hermite_grid, Partial(logistic_ser_hermite_init, warm=False), defaultserkwargs)
            if warm:
                warnings.warn('warm=True not supported for hermite_eb, continuing with warm=False')
        case _:
            raise Exception(f'method = {method} is not a valid option')
    fitfuns = [fixedfitfun] + [fitfun for _ in range(L)]
    model= fit_additive_model(fitfuns, **kwargs)

    fixed_effect = model.components[0]
    sers = tree_stack(model.components[1:])
    fit = SusieFit(fixed_effect, sers, model.state)
    return fit

def summarize_susie(fit):
    fixed_effects = fit.fixed_effect.fit.x
    alpha = np.array(fit.sers.alpha)
    lbf = np.array(fit.sers.fits.lbf)
    beta = np.array(fit.sers.fits.beta.shape)
    prior_variance = np.array(fit.sers.fits.prior_variance[:, 0])
    lbf_ser = np.array(fit.sers.lbf_ser)
    credible_sets = [compute_cs(a) for a in alpha]
    res = SusieSummary(
        fixed_effects,
        alpha,
        lbf,
        beta,
        prior_variance,
        lbf_ser,
        credible_sets
    )
    return res

def fit_logistic_susie2(X, y, L=5, prior_variance=10., estimate_prior_variance=False, maxiter=50, tol=1e-3):
    X = ensure_dense_and_float(X)
    y = ensure_dense_and_float(y)
    fit = fit_logistic_susie(X, y, L=L, warm=True, serkwargs=dict(prior_variance=prior_variance), maxiter=50, tol=1e-3)
    summary = summarize_susie(fit)
    return summary
