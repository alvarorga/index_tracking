"""Functions to minimize tracking error."""

import numpy as np
from scipy.optimize import minimize
from numba import njit, jit

def minimize_w(portfolio):
    """Find set of weights that minimizes the tracking error.

    Parameters
    ----------
    portfolio: Portfolio object.
        Object with information about purchased stocks, daily returns and correlation matrices.

    """
    # Number and indices of purchased stocks.
    d = np.count_nonzero(portfolio.n)
    ix_d = np.where(d == True)[0]

    # Make correlation matrices only with purchased stocks.
    Σ = np.zeros((d, d))
    g = np.zeros(d)
    for i, ix in ix_d:
        g[i] = portfolio.g[ix]
        for j, jx in ix_d:
            Σ[i, j] = portfolio.Σ[ix, jx]

    # Loss function with tracking error.    
    def f(w):
        Terr = np.dot(w, Σ@w) - 2*np.dot(g, w) + portfolio.ε0
        return Terr
    
    # Jacobian of tracking error.
    def j(w):
        dTerr = 2*np.dot(Σ, w) - 2*g
        return dTerr
    
    # Constraints: all weights are positive and add up to 1.
    # w_i >= 0.
    bnds = [(0, None) for i in range(d)]
    # sum(w_i) == 1.
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})

    # Random initalization of weights.
    w0 = np.random.rand(d)
    w0 /= np.sum(w0)
    
    # Solve minimization problem.
    res = minimize(f, w0, jac=j, bounds=bnds, constraints=cons)
    portfolio.w = res.x

    return


def choose_stocks_brute_force(d, w, Σ, g, ε0):
    """Choose a number d of stocks that minimizes the tracking error.

    The function searches over all possible combinations of d stocks.

    Parameters
    ----------
    d: int
        Number of stocks that can be bought. 0 < d < N.

    w: array_like
        Vector of weights.

    Σ: array_like
        Correlation matrix.

    g: array_like
        Correlation with index vector.

    ε0: float
        Average squared index return.

    Return
    ------
    x: array_like
        Array of booleans indicating which d stocks to buy.

    fun: float
        Value of the minimized tracking error.

    """
    # Total number of available stocks.
    N = g.size

    # Modify Σ and g to make computing the error simpler.
    # A is an upper triangular matrix that is multiplied by n_i*n_j
    # with i != j in the tracking error, corresponding to
    # A[i, j] = 2*Σ[i, j]*w[i]*w[j].
    # b is a vector corresponding to the n_i terms. It is defined by
    # b[i] = Σ[i, i]*w[i]**2 - 2*g[i]*w[i]
    A = np.zeros((N, N))
    b = np.zeros(N)
    for i in range(N):
        b[i] = Σ[i, i]*w[i]**2 - 2*g[i]*w[i]
        for j in range(i+1, N):
            A[i, j] = 2*Σ[j, i]*w[i]*w[j]

    # Loop trough all possible combinations of d stocks.
    max_s = 1<<N
    min_terr = 1e5
    min_terr_state = 0
    for s in range(max_s):
        # Discard all states s without d bits.
        nbits = 0
        for i in range(N):
            nbits += (s>>i)&1
        if nbits != d:
            continue

        # Check the tracking error of this combination and store if minimal.
        terr = 0.0
        for i in range(N):
            if (s>>i)&1 == 1:
                terr += b[i]
                for j in range(i+1, N):
                    if (s>>j)&1 == 1:
                        terr += A[i, j]

        if terr < min_terr:
            min_terr = terr
            min_terr_state = s

    # Final tracking error and stock combination.
    terr = min_terr + ε0
    stocks = np.zeros(N, dtype=np.bool)
    for i in range(N):
        if (min_terr_state>>i)&1 == 1:
            stocks[i] = 1

    return stocks, terr