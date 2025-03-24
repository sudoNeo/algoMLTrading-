# algoMLTrading

A modular algorithmic trading system for equities using a combination of machine learning, technical strategies, and risk management. Supports historical backtesting and live/paper trading via Alpaca API.

## Features

- Fetches and processes live or historical data (via `yfinance` and `Alpaca`)
- Calculates technical indicators and fundamental data
- Trains and uses an XGBoost model for directional prediction
- Supports strategies: Mean Reversion, Momentum, Statistical Arbitrage, and Combined
- Risk-managed position sizing and loss limits
- Modular backtester and live trading loop

## Limitations

- **Performance**: Current backtests show negative Sharpe ratio and drawdowns.
- **Signal Quality**: Combined strategy often produces unprofitable trades.
- **Machine Learning**: Functional but needs tuning and better features.
- **Execution**: No real trades are placed unless performance thresholds are passed.
- **Pair Arbitrage**: Only tested on `AAPL` and `BA`.

## Still Needed

- Improve feature engineering and model selection
- Tune XGBoost hyperparameters
- Broaden to more tickers
- Log executed trades in live mode
- Refactor signal-weight logic in CombinedStrategy
- Add unit tests

## Install

Install all required dependencies:
```bash
pip install -r requirements.txt
```


## Output 
![image](https://github.com/user-attachments/assets/22a2bedc-18b5-4752-a74c-3d7a1119670f)

![image](https://github.com/user-attachments/assets/ea0fe36d-b735-449a-be5d-046fd354edaf)

