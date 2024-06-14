#%%
import sys
import os
import numpy as np
from gibss.logisticprofile import wakefield, laplace_mle, hermite_factory, fit_null
import pytest
from argparse import Namespace

@pytest.fixture
def data():
    np.random.seed(0)
    x = np.random.normal(size=1000)
    y = np.random.binomial(1, 1/(1 + np.exp(-(-1 + 0.5 * x))))
    offset = 0.
    prior_variance = 1.
    coef_init = np.array([0., 0.])

    # store in a dotdict
    data = Namespace(x=x, y=y, offset=offset, prior_variance=prior_variance, coef_init=coef_init)
    return data

def test_fit_null(data):
    # null fit with no offset agrees with log odds
    nullfit = fit_null(data.y, np.zeros_like(data.y))
    b0 = nullfit.beta
    ybar = data.y.mean()
    beta0 = np.log(ybar/(1-ybar))
    assert abs(b0 - beta0) < 1e-2

    # if we include the mle for the intercept in the offset, 
    # then we should estimate the intercept to be 0
    nullfit = fit_null(data.y, np.ones_like(data.y)*np.log(ybar/(1-ybar)))
    b0 = nullfit.beta
    assert abs(b0) < 1e-2

def test_wakefield(data):
    nullfit = fit_null(data.y, data.offset)
    res = wakefield(data.coef_init, data.x, data.y, data.offset, data.prior_variance, nullfit)
    print('Wakefield approximation...')
    print(f'\tlbf: {res.lbf:.2f}, beta: {res.beta:.2f}, params: {res.params}')
    assert True

# test_fit_null(data())
# test_wakefield(data())