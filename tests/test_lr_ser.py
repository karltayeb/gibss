import pytest
from argparse import Namespace
import numpy as np

# @pytest.fixture
def data():
    np.random.seed(0)
    x = np.random.normal(size=1000)
    y = np.random.binomial(1, 1/(1 + np.exp(-(-1 + 0.5 * x))))
    offset = 0.
    prior_variance = 1.
    coef_init = np.array([0., 0.])

    X = np.random.normal(size = (100, 1000))
    X[0,:] = x

    Coef_init = np.zeros((100, 2))
    
    # store in a dotdict
    data = Namespace(X=X, x=x, y=y, offset=offset, prior_variance=prior_variance, Coef_init = Coef_init, coef_init=coef_init)
    return data

def test_logistic_ser(data):
    from gibss.logisticprofile import ser, vhermite, fit_null
    fit = ser(data.Coef_init,
        data.X, data.y,
        data.offset,
        data.prior_variance,
        vhermite,
        fit_null
    )
    alpha = np.array(fit.alpha)
    assert (fit.alpha.argmax() == 0)

def test_logistic_ser_hermite(data):
    from gibss.logisticprofile import logistic_ser_hermite
    fit = logistic_ser_hermite(
        data.Coef_init,
        data.X, data.y,
        data.offset,
        data.prior_variance
    )
    alpha = np.array(fit.alpha)
    assert (alpha.argmax() == 0)

def test_logistic_ser_wakefield(data):
    from gibss.logisticprofile import logistic_ser_wakefield
    fit = logistic_ser_wakefield(
        data.Coef_init,
        data.X, data.y,
        data.offset,
        data.prior_variance
    )
    alpha = np.array(fit.alpha)
    assert (alpha.argmax() == 0)

def test_logistic_ser_lapmle(data):
    from gibss.logisticprofile import logistic_ser_lapmle
    fit = logistic_ser_lapmle(
        data.Coef_init,
        data.X, data.y,
        data.offset,
        data.prior_variance
    )
    alpha = np.array(fit.alpha)
    assert (alpha.argmax() == 0)

test_logistic_ser_hermite(data())
test_logistic_ser_lapmle(data())
test_logistic_ser_wakefield(data())