from dataclasses import dataclass
import numpy as np
from scipy import sparse
from gibss.logisticprofile import logistic_susie, logistic_ser_hermite
from typing import Any 

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
    
    # Ensure the matrix is a numpy array (in case it's a list or other type)
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a sparse matrix or a numpy array.")
    
    # Ensure the matrix is of float type
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(float)
        print("Converting matrix to float type.")
    return matrix

@dataclass
class SuSiEOutput:
  alpha: np.ndarray
  effects: np.ndarray
  log_bfs: np.ndarray
  prior_variances: np.ndarray
  fit: Any
  
get_alpha = lambda fit: np.array([c.alpha for c in fit.components])
get_beta = lambda fit: np.array([c.fits.beta for c in fit.components])
get_lbf = lambda fit: np.array([c.lbf_ser for c in fit.components])
get_prior_variance = lambda fit: np.array([c.fits.prior_variance[0] for c in fit.components])

def fit_logistic_susie(X, y, L=5, prior_variance = 1, maxiter = 10, tol=1e-3, return_fit=False):
    X = ensure_dense_and_float(X)
    y = ensure_dense_and_float(y)
    fit = logistic_susie(
        X, y,
        L=int(L), method='hermite',
        serkwargs=dict(m=1, prior_variance=1.0)
    )
    return SuSiEOutput(get_alpha(fit), get_beta(fit), get_lbf(fit), get_prior_variance(fit), fit if return_fit else None)

