import jax.numpy as jnp
import jax
from functools import partial
from dataclasses import dataclass
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp
from gibss.logistic import make_fixed_fitfun
from gibss.additive import fit_additive_model, AdditiveComponent
from scipy import sparse
from gibss.utils import tree_stack, ensure_dense_and_float, npize
from gibss.logistic import SusieFit
from gibss.credible_sets import compute_cs
import numpy as np
from collections import namedtuple
from jax.tree_util import tree_map
from jax.scipy.stats import norm


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["x", "f", "g", "h", "stepsize", "alpha", "gamma"],
    meta_fields=[],
)
@dataclass
class OptState:
    x: ArrayLike
    f: ArrayLike
    g: ArrayLike
    h: ArrayLike
    stepsize: ArrayLike
    alpha: ArrayLike
    gamma: ArrayLike


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "b",
        "llr",
        "lbf",
        "alpha",
        "lbf_ser",
        "pi",
        "prior_variance",
        "optstate",
    ],
    meta_fields=[],
)
@dataclass
class SparseSER:
    b: ArrayLike
    llr: ArrayLike
    lbf: ArrayLike
    alpha: ArrayLike
    lbf_ser: float
    pi: ArrayLike
    prior_variance: float
    optstate: OptState


def _get_sizes(partition):
    return (jnp.roll(partition, -1) - partition)[:-1]


def make_sparse_logistic_ser1d(X_sp, y, prior_variance=1.0, alpha=0.8, gamma=0.0):
    partition = X_sp.indptr
    sizes = _get_sizes(partition)
    indices = X_sp.indices
    xlong = X_sp.data
    ylong = y[X_sp.indices]
    p = X_sp.shape[0]

    ALPHA = alpha * jnp.ones(p)
    GAMMA = gamma * jnp.ones(p)

    def _splitsum(long):
        cumsum = jnp.cumsum(long)[partition[1:] - 1]
        _, splitsum = jax.lax.scan(lambda carry, x: (x, x - carry), 0, cumsum)
        return splitsum

    def _compute_llr(ylong, psilong, psi0long):
        lrlong = (
            ylong * (psilong - psi0long)
            - jnp.log(1 + jnp.exp(psilong))
            + jnp.log(1 + jnp.exp(psi0long))
        )
        return _splitsum(lrlong)

    def decay_stepsize_opt_state(old_opt_state):
        return OptState(
            old_opt_state.x,
            old_opt_state.f,
            old_opt_state.g,
            old_opt_state.h,
            old_opt_state.stepsize * old_opt_state.alpha,
            old_opt_state.alpha,
            old_opt_state.gamma,
        )

    def merge_optstate(old_state, new_state):
        return jax.lax.cond(
            (
                old_state.f
                < new_state.f + new_state.gamma * new_state.g**2 / new_state.h
            ),
            lambda old, new: new,
            lambda old, new: decay_stepsize_opt_state(old),
            old_state,
            new_state,
        )

    merge_opstate_vmap = jax.vmap(merge_optstate, 0, 0)

    def make_optstate(b, psi0long, prior_variance) -> OptState:
        blong = jnp.repeat(b, sizes)
        psilong = psi0long + xlong * blong
        plong = 1 / (1 + jnp.exp(-psilong))
        gradlong = (ylong - plong) * xlong
        hesslong = -plong * (1 - plong) * xlong * xlong
        g = _splitsum(gradlong) - b / prior_variance  # gradient of log p(y, b)
        h = _splitsum(hesslong) - 1 / prior_variance  # hessian of logp(y, b) < 0
        llr = _compute_llr(ylong, psilong, psi0long) + norm.logpdf(
            b, 0, jnp.sqrt(prior_variance)
        )
        # construct new proposed state
        optstate = OptState(b, llr, g, h, jnp.ones_like(llr), ALPHA, GAMMA)
        return optstate

    @jax.jit
    def update_b(state, psi0):
        psi0long = psi0[indices]
        # need to recompute state because `psi0` may have changed
        optstate = make_optstate(state.optstate.x, psi0long, state.prior_variance)
        for _ in range(5):
            # propose newton step
            update_direction = -optstate.g / optstate.h
            b = optstate.x + update_direction * optstate.stepsize
            # package into new optimization state
            optstate_proposed = make_optstate(b, psi0long, state.prior_variance)
            # accept state if sufficient increase
            optstate = merge_opstate_vmap(optstate, optstate_proposed)

        # ser computations
        lbf = optstate.f + 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(-optstate.h))
        lbf_ser = logsumexp(lbf + jnp.log(state.pi))
        alpha = jnp.exp(lbf + jnp.log(state.pi) - lbf_ser)
        state = SparseSER(
            optstate.x,
            optstate.f,
            lbf,
            alpha,
            lbf_ser,
            state.pi,
            state.prior_variance,
            optstate,
        )
        return state

    def logistic_ser_1d(psi, fit):
        if fit is None:
            p, n = X_sp.shape
            opt0 = OptState(
                jnp.ones(p),
                -jnp.inf * jnp.ones(p),
                jnp.ones(p),
                jnp.ones(p),
                jnp.ones(p),
                ALPHA,
                GAMMA,
            )
            state0 = SparseSER(
                jnp.zeros(p), 0, 0, 0, -jnp.inf, jnp.ones(p) / p, prior_variance, opt0
            )
            state = update_b(state0, psi)
        else:
            state = update_b(fit.fit, psi)
        psi = (state.alpha * state.b) @ X_sp
        return AdditiveComponent(psi, state)

    return logistic_ser_1d


def fit_logistic_susie(X_sp, y, L, prior_variance=1.0, alpha=0.8, gamma=0.0, **kwargs):
    fitfun = make_sparse_logistic_ser1d(X_sp, y, prior_variance, alpha, gamma)
    fixedfitfun = make_fixed_fitfun(X_sp, y)
    fitfuns = [fixedfitfun] + [fitfun for _ in range(L)]
    model = fit_additive_model(fitfuns, **kwargs)

    fixed_effect = model.components[0]
    sers = tree_stack(model.components[1:])
    fit = SusieFit(fixed_effect, sers, model.state)
    return fit


SusieSummary = namedtuple(
    "SusieSummary",
    [
        "fixed_effects",
        "alpha",
        "lbf",
        "beta",
        "prior_variance",
        "lbf_ser",
        "credible_sets",
        "optstate",
        "state",
    ],
)


def summarize_susie(fit):
    fixed_effects = np.array(fit.fixed_effect.fit.x)
    alpha = np.array(fit.sers.fit.alpha)
    lbf = np.array(fit.sers.fit.lbf)
    beta = np.array(fit.sers.fit.b)
    prior_variance = np.array(fit.sers.fit.prior_variance)
    lbf_ser = np.array(fit.sers.fit.lbf_ser)
    optstate = tree_map(np.array, fit.sers.fit.optstate).__dict__
    credible_sets = [compute_cs(a).__dict__ for a in alpha]
    res = SusieSummary(
        fixed_effects,
        alpha,
        lbf,
        beta,
        prior_variance,
        lbf_ser,
        credible_sets,
        optstate,
        npize(fit.state.__dict__),
    )
    return res


def fit_logistic_susie2(
    X, y, L=10, prior_variance=1.0, alpha=0.8, gamma=0.0, maxiter=50, tol=1e-3, **kwargs
):
    X_sp = sparse.csr_matrix(X)
    y = ensure_dense_and_float(y)
    fit = fit_logistic_susie(
        X_sp,
        y,
        L=L,
        prior_variance=prior_variance,
        alpha=alpha,
        gamma=gamma,
        maxiter=maxiter,
        tol=tol,
        **kwargs,
    )
    summary = summarize_susie(fit)
    return summary
