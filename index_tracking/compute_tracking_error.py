"""Compute tracking error from quantum circuit results."""

import numpy as np


def compute_mean_terr(res, N, d, Σ, g, ε0):
    """Compute mean tracking error from a set of quantum measurements.
    
    Parameters
    ----------
    res: dict
        Measurements from a quantum circuit in Qiskit format.

    N: int
        Number of available stocks.

    d: int
        Number of chosen stocks.

    Σ: 2d array of floats.
        Stocks correlation matrix.

    g: 1d array of floats.
        Stocks correlation with index.

    ε0: float.
        Correlation of index with itself.

    Return
    ------

    terr: float.
        Mean tracking error.

    """

    terr = 0.0
    shots = 0
    for k, v in res.items():
        state = int(k, 2)
        shots += v
        # Number of stocks in this state.
        ns = 0
        # Compute tracking error of this state.
        terrs = 0.0
        for i in range(N):
            ns += (state>>i)&1
            terrs += (Σ[i, i] - 2*g[i])*((state>>i)&1)
            for j in range(i+1, N):
                terrs += 2*Σ[i, j]*((state>>i)&1)*((state>>j)&1)

        if ns == d:
            terr += terrs*v

    terr /= shots
    terr += ε0
    return terr


def find_min_terr(res, N, d, Σ, g, ε0):
    """Find minimum tracking error from a set of quantum states.
    
    Parameters
    ----------
    res: dict
        Measurements from a quantum circuit in Qiskit format.

    N: int
        Number of available stocks.

    d: int
        Number of chosen stocks.

    Σ: 2d array of floats.
        Stocks correlation matrix.

    g: 1d array of floats.
        Stocks correlation with index.

    ε0: float.
        Correlation of index with itself.

    Return
    ------
    stocks: 1d array of bools.
        Vector with True if stock number i is chosen.

    terr: float.
        Minimum tracking error.

    """
    terr = 1e5
    stocks = ''
    for k, v in res.items():
        state = int(k, 2)
        # Number of stocks in this state.
        ns = 0
        # Compute energy of this state.
        terrs = 0.0
        for i in range(N):
            ns += (state>>i)&1
            terrs += (Σ[i, i] - 2*g[i])*((state>>i)&1)
            for j in range(i+1, N):
                terrs += 2*Σ[i, j]*((state>>i)&1)*((state>>j)&1)
    
        if ns == d and terrs < terr:
            terr = terrs
            stocks = k
            
    return terr, stocks