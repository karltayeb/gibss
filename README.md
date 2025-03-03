# Generalized IBSS

This Python software implements Generalized iterative Bayesian
stepwise selection (GIBSS), an algorithm for estimating posterior
distributions under "Generalized Sum of Single Effects" (GSuSiE)
models. The current software implements several variants of the GSuSiE
model, including for logistic regression and Cox regression.

<!-- The idea is simple: accurately estimate the single effect
regression, which can be done through various approximation techniques
(Laplace, quadrature, etc.), conditional on point estimates from other
effects.  A natural criticism of the approach is that it does not
propagate uncertainty in the point estimates. However, we find
empirically it works well in many situations. -->

## Installation

`gibss` is hosted on PyPI, so once Python is installed, gibss can be
installed with pip:

```bash
pip install gibss
```

It you prefer to install within a new virtual environment (e.g.,
with the name "gibss_venv"), run these lines:

```bash
python3 -m venv gibbs_venv
source gibbs_venv/bin/activate # On Windows use `gibbs_venv\Scripts\activate`
pip install gibss
```

Alternatively, you can install the latest version available on GitHub by cloning or downloading this repository, then running pip:

```bash
git clone https://github.com/karltayeb/gibss.git
cd gibss
pip install .
```

## Minimal example

Here's a very minimal example to get you started:

```python
from gibss.logistic1d import fit_logistic1d_susie
import numpy as np

n = 1000 # number of observations
p = 50 # number of variables
X = np.random.normal(size=(p, n))
logit = X[0] - 1
y = 1 / (1 + np.exp(-logit))

# Fit a logisitic GSuSiE model with L = 5 effects.
fit = fit_logistic1d_susie(X, y, L=5)
```

