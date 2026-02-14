import numpy as np
import pandas as pd


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df: columns = tickers, index = DatetimeIndex, values = Close
    """
    return price_df.pct_change().dropna()


def inverse_vol_weights(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    간단 리스크 패리티(= inverse volatility weighting)
    - 각 자산의 최근 변동성(표준편차)의 역수로 비중 부여
    """
    vol = returns.rolling(window).std(ddof=0).iloc[-1]
    vol = vol.replace(0, np.nan).dropna()

    if vol.empty:
        # fallback: 동일비중
        w = pd.Series(1.0, index=returns.columns)
        return w / w.sum()

    inv = 1 / vol
    w = inv / inv.sum()
    return w


def apply_cash_gate(weights: pd.Series, risk_off: bool, cash_ticker: str = "CASH") -> pd.Series:
    """
    risk_off이면 전액 현금(또는 단기채)로 이동
    """
    if not risk_off:
        return weights

    return pd.Series({cash_ticker: 1.0})


def normalize_weights(weights: pd.Series) -> pd.Series:
    w = weights.copy()
    w = w[w > 0]
    s = float(w.sum())
    if s == 0:
        return w
    return w / s