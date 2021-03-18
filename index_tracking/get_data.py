# Get the data stored in .csv format and compute correlation matrices.

import numpy as np
import pandas as pd


def read_stock_data(ticker, data_location, day_first, day_last):
    """Read daily data of one stock between two dates and output daily returns.

    Parameters
    ----------
    ticker: string
        Ticker name of the stock. E.g. 'CSX'.

    data_location: string
        Location of dataset. E.g. '../data/' will look for the datafile
        '../data/CSX_data.csv'.

    day_first: string
        Date of first day taken into account. Format: year(4 digits)-month-day.
        The return of this day is not included in the output.

    day_last: string
        Date of last day taken into account. The return of this day is included
        in the output.

    Examples
    --------
    >>> read_stock_data('CHX', '../data/', '2021-01-01', '2021-02-27')
    
    """
    # Read data into Pandas datafile. Set index of rows by date.
    datafile = data_location + f'{ticker}_data.csv'
    df = pd.read_csv(datafile, parse_dates=['Date'], index_col='Date')
            
    # Compute daily returns between selected dates.
    day_first = pd.to_datetime(day_first, format="%Y-%m-%d")
    day_last = pd.to_datetime(day_last, format="%Y-%m-%d")
    close = df.loc[(df.index >= day_first) & (df.index <= day_last),
                   'Close'].to_numpy()
                
    r = (close[1:] - close[:-1])/close[:-1]
    # Convert any nan data to 0.
    r = np.nan_to_num(r, copy=False)

    return r


def make_correlation_matrices(ix_ticker, tickers, data_location, 
                              day_first, day_last,
                              normalize=True):
    """Return the correlation matrices Σ, g, ε0 from the stock returns.

    The data is selected between two dates and for specific stocks in the index. 
    If normalize is True then the matrices are normalized as:
        Σ[i, j] -> Σ[i, j]/np.sqrt(Σ[i, i]*Σ[j, j])
        g[i] -> g[i]/np.sqrt(Σ[i, i])

    Example
    -------
    >>> make_correlation_matrices('OSX', ['CHX', 'CLB'], '../data/', 
                                  '2021-01-01', '2021-02-27')

    Return the normalized correlation matrices of the index 'OSX' with stocks 
    'CSH' and 'CLB' between dates '2021-01-01' and '2021-02-27' (first day 
    returns are not included). The data is taken from '../data/'.
    
    """
    # Number of stocks.
    N = len(tickers)

    # Read index data.
    rI = read_stock_data(ix_ticker, data_location, day_first, day_last)
    # Get number of datapoints.
    datapts = np.size(rI, axis=0)
    # Read stock data.
    r = np.zeros((N, datapts))
    for ix, ticker in enumerate(tickers):
        r[ix] = read_stock_data(ticker, data_location, day_first, day_last)
        
    # Make correlation matrices
    Σ = np.zeros((N, N))
    g = np.zeros((N))
    ε0 = np.dot(rI, rI)
    for i in range(N):
        g[i] = np.dot(r[i], rI)
        for j in range(N):
            Σ[i, j] = np.dot(r[i], r[j])

    # Normalize.
    if normalize is True:
        for i in range(N):
            g[i] /= np.sqrt(Σ[i, i])
            for j in range(N):
                Σ[i, j] /= np.sqrt(Σ[i, i]*Σ[j, j])

    return Σ, g, ε0