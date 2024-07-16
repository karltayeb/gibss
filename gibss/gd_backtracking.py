import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from jax.tree_util import Partial

@partial(jax.tree_util.register_dataclass, data_fields=['x', 'f', 'g', 'stepsize', 'converged', 'tol', 'init_ss', 'alpha', 'gamma', 'iter', 'maxiter'], meta_fields=[])
@dataclass
class GDState:
    x: jax.Array
    f: float
    g: jax.Array
    stepsize: float
    converged: bool
    tol: float
    init_ss: float
    alpha: float
    gamma: float
    iter: int
    maxiter: int

def gd_step(state, fun, grad):
    # take step proposed by state
    x = state.x - state.stepsize * state.g 

    # compute likelihood at new value
    f = fun(x)

    def f1(x, f, state):
        g = grad(x)
        return GDState(x, f, g, state.init_ss, jnp.sum(g**2) < state.tol, state.tol, state.init_ss, state.alpha, state.gamma, state.iter + 1, state.maxiter)

    def f2(x, f, state):
        return GDState(state.x, state.f, state.g, state.stepsize * state.alpha, jnp.sum(state.g**2) < state.tol, state.tol, state.init_ss, state.alpha, state.gamma, state.iter + 1, state.maxiter)
    # update state if we accept move (Armijo condition)
    new_state = jax.lax.cond(f < state.f - state.gamma * state.stepsize * jnp.sum(state.g**2),
                             f1, f2, x, f, state) 
    return new_state

@partial(jax.jit)
def gd(x0, fun, grad, maxiter = 10, init_ss = 10.0, tol=1e-4, alpha=0.5, gamma=0.5):
    def converged_or_maxiter_reached(state):
        return jax.lax.cond(
            state.converged | (state.iter >= state.maxiter),
            lambda: False,
            lambda: True
        )
    body_fun = Partial(gd_step, fun=fun, grad=grad)
    state = GDState(x0, fun(x0), grad(x0), init_ss, False, tol, init_ss, alpha, gamma, 0, maxiter)
    state = jax.lax.while_loop(converged_or_maxiter_reached, body_fun, state)
    return state

def gd_factory(f, maxiter=1000, init_ss=1., tol=1e-4, alpha=0.5, gamma=0.5):
    fp = Partial(f)
    grad = Partial(jax.grad(f))
    return partial(gd, fun=fp, grad=grad, maxiter = maxiter, init_ss = init_ss, tol=tol, alpha=alpha, gamma = gamma)
