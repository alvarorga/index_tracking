# Get the data stored in .csv format and make Portfolio class.

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
                   'Close']
    arr_close = close.to_numpy()
                
    r = (arr_close[1:] - arr_close[:-1])/arr_close[:-1]
    # Convert any nan data to 0.
    r = np.nan_to_num(r, copy=False)

    return r, close.index


class Portfolio:
    """Class with returns of several stocks and the index between a date range.
    
    Attributes
    ----------

    N: int
        Number of available stocks to purchase.

    tickers: list of str
        List with all ticker names of the stocks.

    ticker_index: str
        Ticker name of the index.

    w: 1d array of floats
        Weights of each stock in our portfolio.

    n: 1d array of bools
        Array with all purchased stocks: either true or false.
    
    date_range: list of str
        Day of first returns and day of last returns.

    returns: pandas DataFrame
        Dictionary with daily returns of all available stocks indexed by date.

    Σ: 2d array of floats
        Correlation matrix of returns between date range.
    
    g: 1d array of floats
        Correlation matrix of index and all stocks returns between date range.
    
    ε0: float
        Correlation matrix of index returns between date range.
    """

    def __init__(self, tickers, ticker_index, data_location, day_first, day_last):

        self.tickers = tickers
        self.ticker_index = ticker_index 
        self.N = len(tickers)

        # Weight of current stocks in the portfolio.
        self.w = np.full(self.N, 1/self.N)
        # Purchased stocks.
        self.n = np.full(self.N, True)

        # Date range of data.
        self.date_range = [day_first, day_last]

        # Read daily returns and their dates.
        r_dict = {}
        r_dict[ticker_index], date_index = read_stock_data(
            ticker_index, data_location, day_first, day_last
            )
        for ticker in tickers:
            r_dict[ticker], _ = read_stock_data(ticker, data_location, day_first, day_last)

        # Store into DataFrame, remove first day from return dates.
        self.returns = pd.DataFrame(data=r_dict, index=date_index[1:])

        # Make correlation matrices
        self.Σ = np.zeros((self.N, self.N))
        self.g = np.zeros((self.N))
        self.ε0 = np.dot(self.returns[ticker_index], self.returns[ticker_index])

        for i, ticker_i in enumerate(tickers):
            self.g[i] = np.dot(self.returns[ticker_i], self.returns[ticker_index])
            for j, ticker_j in enumerate(tickers):
                self.Σ[i, j] = np.dot(self.returns[ticker_i], self.returns[ticker_j])
