#%%
import sys
import os
import numpy as np
from gibss.logisticprofile import wakefield, laplace_mle, hermite_factory, fit_null
import pytest
from argparse import Namespace

# @pytest.fixture
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
    b0, ll0 = fit_null(data.y, data.offset)
    print('Fitting null model...')
    print(f'\tb0: {b0}, ll0: {ll0}')
    assert True

def test_wakefield(data):
    lbf, beta, params = wakefield(data.coef_init, data.x, data.y, data.offset, data.prior_variance)
    print('Wakefield approximation...')
    print(f'\tlbf: {lbf:.2f}, beta: {beta:.2f}, params: {params}')
    assert True

def test_laplace_mle(data):
    lbf, beta, params = laplace_mle(data.coef_init, data.x, data.y, data.offset, data.prior_variance)
    from gibss.logisticprofile import nloglik_mle
    ll = -nloglik_mle(params, data.x, data.y, data.offset)
    print('Laplace MLE...')
    print(f'\tlogp: {lbf:.2f}, beta: {beta:.2f}, params: {params}')
    print(f'\tll: {ll:.2f}')
    assert True

def test_hermite(data):
    hermite = hermite_factory(5)
    lbf, beta, params = hermite(data.coef_init, data.x, data.y, data.offset, data.prior_variance)
    print('Hermite quadrature...')
    print(f'\tlbf: {lbf:.2f}, beta: {beta:.2f}, params: {params}')
    assert True

test_fit_null(data())
test_wakefield(data())
test_laplace_mle(data())
test_hermite(data())