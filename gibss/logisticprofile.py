# Implement Logistic SER using JAX
# for each variable maximize the intercept, "profile likelihood" method
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
import jax
import numpy as np
from functools import partial
from gibss.ser import ser, _ser
from jax.tree_util import Partial
from typing import Any
from dataclasses import dataclass
from gibss.newton import newton_factory
from gibss.additive import fit_additive_model, make_fitfun
from gibss.utils import tree_stack

@partial(jax.tree_util.register_dataclass,
         data_fields=['logp', 'lbf', 'beta', 'prior_variance', 'state'], meta_fields=[])
@dataclass
class UnivariateRegression:
    logp: float
    lbf: float
    beta: float
    prior_variance: float
    state: Any

@jax.jit
def nloglik_mle(coef, x, y, offset):
    """Logistic log-likelihood"""
    psi = offset + coef[0] + x * coef[1]
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi))
    return -ll

nloglik_mle_hess = jax.hessian(nloglik_mle)
nloglik_mle_vmap = jax.vmap(nloglik_mle, in_axes=(0, None, None, None))

@jax.jit
def nloglik(coef, x, y, offset, prior_variance):
    """Logistic log-likelihood"""
    psi = offset + coef[0] + x * coef[1]
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi)) + norm.logpdf(coef[1], 0, jnp.sqrt(prior_variance))
    return -ll

nloglik_hess = jax.hessian(nloglik)
nloglik_vmap = jax.vmap(nloglik, in_axes=(0, None, None, None, None))

def compute_wakefield_lbf(betahat, s2, prior_variance):
    lbf = norm.logpdf(betahat, loc=0, scale=jnp.sqrt(s2 + prior_variance)) \
        - norm.logpdf(betahat, loc=0, scale=jnp.sqrt(s2))
    return lbf

def wakefield(coef_init, x, y, offset, nullfit, prior_variance, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
    solver = newton_factory(Partial(nloglik_mle, x=x, y=y, offset=offset), maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma)
    state = solver(coef_init)
    params = state.x
    hessian = -state.h

    # approximate BF with wakefield
    # see appendix of Wakefield 2009 for justicatioin of why there is no dependence on the intercept
    # s2 = -1/hessian[1,1]
    s2 = - hessian[0 ,0] / (hessian[0,0] * hessian[1, 1] - hessian[1,0] * hessian[0,1])
    lbf = compute_wakefield_lbf(params[1], s2, prior_variance)

    # shrink  the effect size
    # this gives an asymptotic approximation to the MAP estimate from the MLE
    # by combining the quadratic approximation of the likelihood at the MLE 
    # with a normal prior
    beta = params[1] * prior_variance / (s2 + prior_variance)
    # NOTE: the value for logp might not make sense
    return UnivariateRegression(lbf + nullfit.logp, lbf, beta, prior_variance, prior_variance, state)

def estimate_prior_variance_wakefield(fits, prior_variance_init):
    """Estimate prior variance using Wakefield approximation"""
    def f(ln_prior_variance):
        return -logsumexp(compute_wakefield_lbf(
            fits.state.x[:, 1],
            1/fits.state.h[:, 1,1], 
            jnp.exp(ln_prior_variance)))
    fopt = gd_factory(f, maxiter=100, init_ss=1.)
    return jnp.exp(fopt(jnp.atleast_1d(jnp.log(prior_variance_init))).x)[0]

@partial(jax.vmap, in_axes=(0, None))
def update_prior_variance_wakefield(fit: UnivariateRegression, prior_variance: float) -> UnivariateRegression:
    # extract state
    betahat = fit.state.x[1]
    # s2 = 1/fit.state.h[1, 1]
    hessian = - fit.state.h
    s2 = - hessian[0 ,0] / (hessian[0,0] * hessian[1, 1] - hessian[1,0] * hessian[0,1])
    ll0 = fit.logp - fit.lbf

    # compute wakefield lbf
    lbf = compute_wakefield_lbf(betahat, s2, prior_variance)
    beta = betahat * prior_variance / (s2 + prior_variance)
    return UnivariateRegression(lbf + ll0, lbf, beta, prior_variance, fit.state)

# def compute_lapmle_logp(ll, betahat, tau1, tau):
#     logp = ll + \
#         0.5 * jnp.log(tau1/(tau1 + tau)) \
#         - 0.5 * tau*tau1/(tau + tau1) * betahat**2
#     return logp

def compute_lapmle_logp(ll, betahat, s2, prior_variance):
    logp = ll + \
        0.5 * jnp.log(2 * jnp.pi * s2) + \
        norm.logpdf(betahat, loc=0, scale=jnp.sqrt(s2 + prior_variance))
    return logp

def laplace_mle(coef_init, x, y, offset, nullfit, prior_variance, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
    solver = newton_factory(Partial(nloglik_mle, x=x, y=y, offset=offset), maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma)
    state = solver(coef_init)
    params = state.x
    hessian = -state.h

    # compute wakefield lbf
    # s2 = -1/hessian[1,1]
    s2 = - hessian[0 ,0] / (hessian[0,0] * hessian[1, 1] - hessian[1,0] * hessian[0,1])
    ll = - state.f
    betahat = params[1]
    logp = compute_lapmle_logp(ll, betahat, s2, prior_variance)
    lbf = logp - nullfit.logp
    beta = betahat * prior_variance / (s2 + prior_variance)
    return UnivariateRegression(logp, lbf, beta, prior_variance, state)

def compute_lbf_score(x, y, offset, nullfit, prior_variance):
    # evaluate the gradient and the hessian at the null
    b = jnp.array([nullfit.beta, 0.])
    grad = jax.grad(nloglik)(b, x, y, 0., prior_variance)    
    hess = jax.hessian(nloglik)(b, x, y, 0., prior_variance)

    # approximate the log BF with a quadratic approximation of log p(y, b) about b=0
    tau = jnp.linalg.det(hess) / hess[0, 0]  # 1 / H^{-1}_{11}
    lbf = 0.5 * jnp.log(2 * jnp.pi) -0.5 * jnp.log(tau) + 0.5 * grad[1]**2 / tau
    return lbf    
    
@partial(jax.vmap, in_axes=(0, None))
def update_prior_variance_lapmle(fit: UnivariateRegression, prior_variance: float) -> UnivariateRegression:
    # extract state
    betahat = fit.state.x[1]
    # s2 = 1/fit.state.h[1, 1]
    hessian = - fit.state.h
    s2 = - hessian[0 ,0] / (hessian[0,0] * hessian[1, 1] - hessian[1,0] * hessian[0,1])
    ll = -fit.state.f

    # compute wakefield lbf
    logp = compute_lapmle_logp(ll, betahat, s2, prior_variance)
    lbf = fit.lbf - fit.logp + logp
    beta = betahat * prior_variance / (s2 + prior_variance)
    return UnivariateRegression(logp, lbf, beta, prior_variance, fit.state)
    

def estimate_prior_variance_lapmle(fits: UnivariateRegression, prior_variance_init: float) -> float:
    """Estimate prior variance using Wakefield approximation"""
    def f(ln_prior_variance):
        return -logsumexp(compute_lapmle_logp(
            -fits.state.f, fits.state.x[:, 1], 1/fits.state.h[:, 1,1], jnp.exp(ln_prior_variance))
        )
    fopt = gd_factory(f, maxiter=100, init_ss=1.)
    return jnp.exp(fopt(jnp.atleast_1d(jnp.log(prior_variance_init))).x)[0]

# gauss-hermite quadrature nodes and weights
def hermite_factory(m):
    base_nodes, base_weights = np.polynomial.hermite.hermgauss(m)
    def hermite(coef_init, x, y, offset, nullfit, prior_variance, newtonkwargs=dict(maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1)):
        solver = newton_factory(Partial(nloglik, x=x, y=y, offset=offset, prior_variance=prior_variance), **newtonkwargs)
        state = solver(coef_init)
        params = state.x
        hessian = -state.h

        # set up quadrature
        mu = params[1]
        sigma = jnp.sqrt(-1/hessian[1,1])
        nodes = base_nodes * jnp.sqrt(2) * sigma + mu
        weights = base_weights / jnp.sqrt(jnp.pi)

        # compute logp
        coef_nodes = jnp.stack([jnp.ones_like(nodes) * params[0], nodes], axis=1)
        y = -nloglik_vmap(coef_nodes, x, y, offset, prior_variance)
        logp = jax.scipy.special.logsumexp(y - norm.logpdf(nodes, loc=mu, scale=sigma) + jnp.log(weights)) 
        lbf = logp - nullfit.logp

        # compute posterior mean of effect
        y2 = nodes * jnp.exp(y - logp)
        beta = jnp.sum(y2/norm.pdf(nodes, loc=mu, scale=sigma) * weights)
        return UnivariateRegression(logp, lbf, beta, prior_variance, state)
    
    hermite_jit = jax.jit(hermite)
    return hermite_jit


def hermite_grid_factory(m):
    base_nodes, base_weights = np.polynomial.hermite.hermgauss(m)
    def hermite_grid(coef_init, x, y, offset, nullfit, prior_variance_grid, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
        def hermite_inner(coef_init, prior_variance):
            solver = newton_factory(
                Partial(nloglik, x=x, y=y, offset=offset, prior_variance=prior_variance),
                maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma
            )
            state = solver(coef_init)
            params = state.x
            hessian = -state.h
        
            # set up quadrature
            mu = params[1]
            sigma = jnp.sqrt(-1/hessian[1,1])
            nodes = base_nodes * jnp.sqrt(2) * sigma + mu
            weights = base_weights / jnp.sqrt(jnp.pi)
        
            # compute logp
            coef_nodes = jnp.stack([jnp.ones_like(nodes) * params[0], nodes], axis=1)
            nll = -nloglik_vmap(coef_nodes, x, y, offset, prior_variance)
            logp = jax.scipy.special.logsumexp(nll - norm.logpdf(nodes, loc=mu, scale=sigma) + jnp.log(weights)) 
            lbf = logp - nullfit.logp
        
            # compute posterior mean of effect
            tmp = nodes * jnp.exp(nll - logp)
            beta = jnp.sum(tmp/norm.pdf(nodes, loc=mu, scale=sigma) * weights)
            return state.x, UnivariateRegression(logp, lbf, beta, prior_variance, state)
            
        res = jax.lax.scan(hermite_inner, coef_init, prior_variance_grid)
        return(res[1])
    return(jax.jit(hermite_grid))


@jax.jit
def nloglik0(b0, y, offset):
    """Logistic log-likelihood"""
    psi = offset + b0
    ll = jnp.sum(y * psi - jnp.logaddexp(0, psi))
    return -ll


@jax.jit
def fit_null(y, offset, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
    """Logistic SER"""
    # we fit null model by giving a covariate with no variance and setting prior variance to ~=0
    # so that we can just reuse the code for fitting the full model
    solver = newton_factory(Partial(nloglik0, y=y, offset=offset), maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma)
    state = solver(jnp.zeros(1))
    params = state.x    
    return UnivariateRegression(-state.f, 0, state.x[0], 0., state)


# use Partial so that you can pass to jitted ser function
# see https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.Partial.html
@partial(jax.jit, static_argnames = ['maxiter'])
def logistic_ser_wakefield(coef_init, X, y, offset, prior_variance, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
    vwakefield = jax.vmap(Partial(wakefield, prior_variance=prior_variance, maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma), in_axes=(0, 0, None, None, None))
    return ser(coef_init, X, y, offset, vwakefield, fit_null)


@jax.jit
def logistic_ser_wakefield_eb(coef_init, X, y, offset, prior_variance, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
    # 1. fit ser, choice of prior variance doesn't matter
    nullfit = fit_null(y, offset)
    vwakefield = jax.vmap(Partial(wakefield, prior_variance=prior_variance, maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma), in_axes=(0, 0, None, None, None))

    fits = vwakefield(coef_init, X, y, offset, nullfit)
    prior_variance = estimate_prior_variance_wakefield(fits, prior_variance)
    fits2 = update_prior_variance_wakefield(fits, prior_variance)
    return _ser(fits2, X)


@jax.jit
def logistic_ser_lapmle(coef_init, X, y, offset, prior_variance, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
    vlapmle = jax.vmap(Partial(laplace_mle, prior_variance=prior_variance, maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma), in_axes=(0, 0, None, None, None))
    return ser(coef_init, X, y, offset, vlapmle, fit_null)


@jax.jit
def logistic_ser_lapmle_eb(coef_init, X, y, offset, prior_variance, maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1):
    # 1. fit ser, choice of prior variance doesn't matter
    nullfit = fit_null(y, offset)
    vlapmle = jax.vmap(Partial(laplace_mle, prior_variance=prior_variance, maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma), in_axes=(0, 0, None, None, None))
    fits = vlapmle(coef_init, X, y, offset, nullfit)
    prior_variance = estimate_prior_variance_lapmle(fits, prior_variance)
    fits2 = update_prior_variance_lapmle(fits, prior_variance)
    return _ser(fits2, X)


@partial(jax.jit, static_argnames = ['m'])
def logistic_ser_hermite(coef_init, X, y, offset, prior_variance, m, newtonkwargs=dict(maxiter=50, tol=0.1, alpha=0.1, gamma=-0.1)):
    vhermite = jax.vmap(Partial(hermite_factory(m), prior_variance = prior_variance, newtonkwargs=newtonkwargs), in_axes=(0, 0, None, None, None))
    return ser(coef_init, X, y, offset, vhermite, fit_null)


@partial(jax.jit, static_argnames = ['m', 'maxiter'])
def logistic_ser_hermite_grid(coef_init, X, y, offset, prior_variance_grid, m, maxiter=5, tol=0.1, alpha=0.1, gamma=-0.1):
    vhermite = jax.vmap(Partial(hermite_grid_factory(m), 
                                prior_variance_grid = prior_variance_grid, 
                                maxiter=maxiter, tol=tol, alpha=alpha, gamma=gamma),
                        in_axes=(0, 0, None, None, None))
    nullfit = fit_null(y, offset)  # fit null model
    gridfits = vhermite(coef_init, X, y, offset, nullfit)  # fit univariate models 

    # select best prior variance
    best_prior_variance = jax.scipy.special.logsumexp(gridfits.lbf, axis=0).argmax()
    fits = jax.tree.map(lambda x: x[:, best_prior_variance], gridfits)
    return _ser(fits, X)


def initialize_coef_zeros(X, y, offset):
    """Initialize univarate regression coefficients using null model"""
    return jnp.zeros((X.shape[0], 2)) 

def initialize_coef_null(X, y, offset):
    # initialize with null coef
    nullfit = fit_null(y, offset)  # fit null model
    p = X.shape[0]
    coef_init = jnp.stack([jnp.ones(p) * nullfit.state.x, jnp.zeros(p)], axis=1)
    return coef_init


def logistic_ser_hermite_init(X, y, psi, fit=None, warm=False):
    """
    If fit is None or warm=False initialize with zeros
    Otherwise, initialize with with current effect estimates from fit.
    """
    if ((fit is None) or (not warm)): 
        coef_init = jnp.zeros((X.shape[0], 2))
    else:   
        coef_init = fit.fits.state.x
    return coef_init


# def fit_logistic_susie(X, y, L=5, maxiter=10, tol=0.1, method='hermite', serkwargs: dict = dict(), cleanup=True):
def fit_logistic_susie(X, y, L=5, method='hermite', warm=True, serkwargs: dict = dict(), **kwargs):
    match method:
        case 'hermite':
            defaultserkwargs  = dict(
                prior_variance = float(10.),
                newtonkwargs=dict(tol=1e-2, maxiter=5, alpha=0.2, gamma=-0.1)
            )
            defaultserkwargs.update(serkwargs)
            fitfun = make_fitfun(X, y, logistic_ser_hermite, Partial(logistic_ser_hermite_init, warm=warm), defaultserkwargs)
        case 'hermite_eb':
            initfun = initialize_coef_null
            serkwargs2 = dict(m=1, prior_variance_grid=jnp.array([0.0001, 0.001, 0.01, 0.1, 1., 10., 100.]))
            serkwargs2.update(serkwargs)
            serfun = partial(logistic_ser_hermite_grid, **serkwargs2)
        case 'hermite_eb':
            defaultserkwargs = dict(
                prior_variance_grid = jnp.array([1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1., 2., 4.]),
                newtonkwargs=dict(tol=1e-2, maxiter=20, alpha=0.8, gamma=-0.1)
            )
            defaultserkwargs.update(serkwargs)
            fitfun = make_fitfun(X, y, logistic_ser_hermite_grid, Partial(logistic_ser_hermite_init, warm=False), defaultserkwargs)
            if warm:
                warnings.warn('warm=True not supported for hermite_eb, continuing with warm=False')
        case _:
            raise Exception(f'method = {method} is not a valid option')

        # case 'wakefield':
        #     initfun = initialize_coef_null
        #     serfun = partial(logistic_ser_wakefield, **serkwargs)
        # case 'wakefield_eb':
        #     initfun = initialize_coef_null
        #     serfun = partial(logistic_ser_wakefield_eb, **serkwargs)
        # case 'laplace_mle':
        #     initfun = initialize_coef_null
        #     serfun = partial(logistic_ser_lapmle, **serkwargs)
        # case 'laplace_mle_eb':
        #     initfun = initialize_coef_null
        #     serfun = partial(logistic_ser_lapmle_eb, **serkwargs)
    
    fitfuns = [fitfun for _ in range(L)]
    model = fit_additive_model(fitfuns, **kwargs)
    # sers = tree_stack(components)
    return sers, state
