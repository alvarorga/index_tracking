import matplotlib.pyplot as plt
import numpy as np


def visualize_picked_stocks(pf):
    """Plot daily returns of stocks, index and portfolio."""

    # Label- and tickfontsize.
    lfs = 20
    tfs = 16 

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    # Returns of purchased stocks.
    for ix, ticker in enumerate(pf.tickers):
        if pf.n[ix]:
            ax.plot(pf.returns[ticker], '0.9', lw=1)

    # Index returns.
    ax.plot(pf.returns[pf.ticker_index], 'C0', lw=2, marker='o', label='Index')

    # Porfolio returns.
    pf_returns = np.zeros(np.size(pf.returns[pf.ticker_index]))
    for ix, ticker in enumerate(pf.tickers):
        if pf.n[ix]:
            pf_returns += pf.returns[ticker]*pf.w[ix]

    ax.plot(pf_returns, 'C1', lw=2, marker='o', label='Portfolio')

    # Legend and ticks.
    ax.legend(fontsize=lfs)
    ax.tick_params(axis='x', labelrotation=60, labelsize=tfs)
    ax.tick_params(axis='y', labelsize=tfs)

    plt.show()

    return