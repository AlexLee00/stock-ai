import pandas as pd
import yfinance as yf

from core.features import add_quant_features
from core.model import compute_signal_score
from core.backtest import backtest_simple

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

START = "2018-01-01"
END = "2024-01-01"


def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    # yfinance가 가끔 MultiIndex를 줄 때 방어
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}, got={list(df.columns)}")

    return df[needed].copy()


featured_map: dict[str, pd.DataFrame] = {}

for t in TICKERS:
    print(f"Downloading {t}...")
    df = load_data(t)

    df = add_quant_features(df)

    # ✅ 여기서 바로 feature 컬럼 확인 (한 번만)
    if t == TICKERS[0]:
        print("FEATURE COLS:", [c for c in df.columns if "macd" in c or "bb_" in c])

    df = compute_signal_score(df)
    featured_map[t] = df


print("\nRunning backtest...\n")

result = backtest_simple(
    featured_map,
    top_n=3,
    rebalance_every_days=21,
    vol_window=60,
    market_ticker="SPY",   # ✅ backtest_simple이 이 인자를 받도록 아래 2)에서 수정
    cash_ticker="CASH",
)

print(result.tail())

if not result.empty:
    total_return = result["port_value"].iloc[-1] - 1
    max_dd = (result["port_value"] / result["port_value"].cummax() - 1).min()

    print("\n======================")
    print("총 수익률:", round(total_return * 100, 2), "%")
    print("최대 낙폭:", round(max_dd * 100, 2), "%")
    print("======================")