# A simple newton method
# 1. run for a fixed number of *function evaluations*
# 2. try taking a newton step with stepsize = 1
# 3. if the likelihood increasses, keep the proposed move and set the stepsize to 1
# 4. if the likelihood decreases, reject the proposed move and halve the step size
# Here we assume `f` is a function to MINIMIZE
import jax
from functools import partial
from dataclasses import dataclass
import numpy as np
from jax.tree_util import Partial

@partial(jax.tree_util.register_dataclass,
         data_fields=['x', 'f', 'g', 'h', 'stepsize'], meta_fields=[])
@dataclass
class NewtonState:
    x: jax.Array
    f: float
    g: jax.Array
    h: jax.Array
    stepsize: float

def newton_step(state, fun, grad, hess):
    # take newton step proposed by state
    x = state.x - state.stepsize * jax.scipy.linalg.solve(state.h, state.g)

    # compute likelihood at new value
    f = fun(x)

    # update state if we accept move
    new_state = jax.lax.cond(f < state.f,
                 lambda x, f, state, fun, grad, hess: NewtonState(x, f, grad(x), hess(x), 1.),
                 lambda x, f, state, fun, grad, hess: NewtonState(state.x, state.f, state.g, state.h, state.stepsize / 2.),
                 x, f, state, fun, grad, hess)
    return new_state

@partial(jax.jit, static_argnames=['niter'])
def newton(x0, f, grad, hess, niter = 10):
    state = NewtonState(x0, f(x0), grad(x0), hess(x0), 1.0)
    for i in range(niter):
        state = newton_step(state, f, grad, hess)
    return state

def newton_factory(f, niter=5):
    fp = Partial(f)
    grad = Partial(jax.grad(f))
    hess = Partial(jax.hessian(f))
    return partial(newton, f=fp, grad=grad, hess=hess, niter=niter)

