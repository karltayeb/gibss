
import dataclasses
import jax
import numpy as np

def todict(x):
    # if its a dataclass, convert to dict
    if dataclasses.is_dataclass(x):
        x = dataclasses.asdict(x)
    # numpy-ize the pytree
    x = jax.tree.map(lambda x: np.array(x) if isinstance(x, jax.Array) else x, x)

    # NOTE: eventually we should improve monitor and report information about model fitting
    monitor = x.pop('monitor', None)
    if monitor is not None:
        x['converged'] = monitor.converged
        x['tol'] = monitor.tol 
    return x