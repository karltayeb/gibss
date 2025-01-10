
import dataclasses
import jax
import numpy as np
import jax.numpy as jnp
from scipy import sparse
from jax.tree_util import tree_map
from jax import Array

def npize(pytree):
    return tree_map(lambda x: np.array(x) if isinstance(x, Array) else x, pytree)

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

def tree_stack(trees):
    # https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def ensure_dense_and_float(matrix):
    """
    X = np.random.binomial(1, np.ones((5, 5))*0.5)
    Xsp = sparse.csr_matrix(X)
    ensure_dense_and_float(Xsp)

    y = np.random.binomial(1, np.ones(5)*0.5)
    ensure_dense_and_float(y)
    """
    # Check if the input is a sparse matrix
    if sparse.issparse(matrix):
        # Convert sparse matrix to a dense array
        matrix = matrix.toarray()
        # Provide a message to the user
        print("Input is a sparse matrix. Converting to a dense array.")
    
    if isinstance(matrix, list):
        try:
            matrix = np.array(matrix)
        except:
            raise Exception("Failed to convert list to np.ndarray")


    # Ensure the matrix is a numpy array (in case it's a list or other type)
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a sparse matrix or a numpy array.")
    
    # Ensure the matrix is of float type
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(float)
        print("Converting matrix to float type.")
    return matrix

