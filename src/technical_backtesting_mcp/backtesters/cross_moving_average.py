from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
from typing import Sequence, cast

from price_loaders.tradingview import load_asset_price


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


class CrossMovingAverage(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20

    sma1: Sequence
    sma2: Sequence

    def init(self):
        # Precompute the two moving averages
        self.sma1 = cast(Sequence, self.I(SMA, self.data.Close, self.n1))
        self.sma2 = cast(Sequence, self.I(SMA, self.data.Close, self.n2))

    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()


def perform_backtest(
    symbol: str,
    n1: int = 10,
    n2: int = 20,
    cash: float = 10_000,
    commission: float = 0.002,
) -> str:
    """Perform a backtest for a given symbol and return the stats.

    Args:
        symbol (str): TradingView symbol identifier (e.g. "NASDAQ:META", "SET:BH", "BITSTAMP:BTCUSD").
        n1 (int, optional): The first moving average lag. Defaults to 10.
        n2 (int, optional): The second moving average lag. Defaults to 20.
        cash (float, optional): The initial cash. Defaults to 10_000.
        commission (float, optional): The commission. Defaults to 0.002.

    Returns:
        str: The stats of the backtest.
    """
    df = load_asset_price(symbol, 5000, "1D")
    df = df.set_index("time")
    try:
        df.columns = ["Open", "High", "Low", "Close", "Volume", "pe_ratio"]
    except ValueError:
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

    df.index = pd.to_datetime(cast(pd.DatetimeIndex, df.index.date))
    df.index = pd.DatetimeIndex(df.index)

    bt = Backtest(df, CrossMovingAverage, cash=10_000, commission=0.002)
    stats = bt.run()
    try:
        bt.plot(filename=f"{symbol}.backtest.html")
    except Exception as e:
        # Skip plotting if the file already exists
        print(e)
    return stats.to_string()


if __name__ == "__main__":
    print(perform_backtest("SET:KBANK"))
