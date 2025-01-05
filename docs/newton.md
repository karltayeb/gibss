# Newton's method in JAX

## Motivation 

In order to apply GIBSS you need to provide a univariate regression function that returns, at a minimum, a Bayes factor and a point estimate of the predictions for each observation.
We use GIBSS to do Bayesian variable selection in logistic regression or Cox regression.
In both these examples (and GLMs more generally) the univariate regression problems are convex.

We make the process of implementing new models in GIBSS simple. We implement a Newton's method with backgracking line search in JAX. 
By providing a an implementation of the objective function in JAX, and utilizing JAXs automatic differentiation and vectorization features,
GIBSS automates the implementation of of the single effect regression by first computing the MAP estimate for each variable, and then approximating the Bayes factor via Gauss-Hermite quadrature. 


## The Newton optimizer.

We use JAX to implement a simple, light weight newton optimizer. 
At a high level, the optimizer takes a function to be minimzed, uses JAX to compute the gradient and hessian,
and performs newton steps. 
For convex objectives, the newton *direction* is guarunteed to be a descent direction, meaning that there is a stepsize such that $x_{t+1} = x_t - \beta H_t^{-1}g_t$ will decrease the objective function. We start with a step size of $1$ and decay geometrically by a factor $\alpha \in (0, 1)$ until the function achieved a *sufficient decrease*. That means we require $f(x_{t+1}) < f(x_t) - \gamma ||g||_2^2$. 

The two parameters controlling the behavior of the optimizer are `alpha`, which control the decay of the stepsize in the backtracking line search, and `gamma` which controls the decrease criteria of accepting a proposed move. Selecting $\gamma = -\inf$ corresponds with always accepting the move. When a convex optimization problem is initialized sufficiently close to its optimum it can be shown that a step size of $\beta = 1$ is always decreasing. However, when initialzed far from the optimum there is no guaruntee. 

### Small, negative values of $\gamma$. 
In practice we find that due to numerical stablity issues can cause updates to slightly increase the objective, while substantially decreasing Newton's decrement, which provides an upper bound on the suboptimality of the solution. We monitor convergence by Newton's decrement, so we find it useful to set `gamma` to a small negative value, e.g. $0.1$. In practice, we find this prevents us from divergent behavior at poor initialization, without causing the algorithm to stall.

### A comment on numerical issues
JAX JIT compiler is known to change the numeric output of comptutations. 
That is, the JIT compiled version of a function might produce slightly different output compared to the original function. 
This is due to the fact that floating point airthmatic is only an approximation of regular arithmetic, 
and clever optimizations made by the compiler might change the sequence of operations the machine carries out.
In practice, we find that even extremely small steps in the descent direction can apprently *increase* the objective by a small amount, while dramatically decreasing an upper bound/approximation to the sub-optimality. 

