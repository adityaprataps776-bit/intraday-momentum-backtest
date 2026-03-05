import yfinance as yf
import pandas as pd
import numpy as np

def download_data(ticker="SPY", period="60d", interval="5m"):
    data = yf.download(ticker, period=period, interval=interval)
    return data[['Close']].dropna()

def compute_strategy(data, window=6, transaction_cost=0.0002):
    data['returns'] = data['Close'].pct_change()
    data['rolling_return'] = data['Close'].pct_change(window)

    data['signal'] = np.where(data['rolling_return'] > 0, 1, -1)
    data['signal'] = data['signal'].shift(1)

    data['strategy_returns'] = data['signal'] * data['returns']

    data['trade'] = data['signal'].diff().abs()
    data['strategy_returns'] -= transaction_cost * data['trade']

    data['cum_market'] = (1 + data['returns']).cumprod()
    data['cum_strategy'] = (1 + data['strategy_returns']).cumprod()

    return data

def sharpe_ratio(returns, periods_per_year=252*78):
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def max_drawdown(cum_returns):
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return drawdown.min()

if __name__ == "__main__":
    data = download_data()
    data = compute_strategy(data)

    sharpe = sharpe_ratio(data['strategy_returns'].dropna())
    mdd = max_drawdown(data['cum_strategy'])

    print("Sharpe Ratio:", sharpe)
    print("Max Drawdown:", mdd)