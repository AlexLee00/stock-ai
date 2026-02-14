import pandas as pd
import numpy as np


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_macd(
    df: pd.DataFrame,
    price_col: str = "Close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    out = df.copy()
    price = out[price_col].astype(float)

    macd_line = ema(price, fast) - ema(price, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line

    # ✅ 컬럼명 통일 (소문자)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    return out


def add_bollinger_bands(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()
    price = out[price_col].astype(float)

    mid = price.rolling(window).mean()
    std = price.rolling(window).std(ddof=0)

    upper = mid + num_std * std
    lower = mid - num_std * std

    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower

    denom = (upper - lower).replace(0, np.nan)
    out["bb_percent_b"] = (price - lower) / denom
    out["bb_bandwidth"] = (upper - lower) / mid.replace(0, np.nan)

    return out


def add_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD + Bollinger Bands + 안전 처리
    - yfinance가 MultiIndex 컬럼을 주는 경우도 있어서 flatten 처리 포함
    - dropna는 '존재하는 컬럼'만 대상으로 하되, 절대 KeyError 안 나게 처리
    """
    out = df.copy()

    # ✅ yfinance MultiIndex 방어
    if isinstance(out.columns, pd.MultiIndex):
        # 보통 첫 레벨이 OHLCV라서 0레벨만 쓰면 됨
        out.columns = out.columns.get_level_values(0)

    # ✅ Close 컬럼 확인
    if "Close" not in out.columns:
        raise ValueError(f"'Close' column not found. columns={list(out.columns)}")

    out = add_macd(out)
    out = add_bollinger_bands(out)

    out = out.replace([np.inf, -np.inf], np.nan)

    needed = ["macd_hist", "bb_percent_b", "bb_bandwidth"]

    # ✅ 진짜로 존재하는 컬럼만 subset으로 dropna
    cols = [c for c in needed if c in out.columns]
    if cols:
        out = out.dropna(subset=cols)
    else:
        # cols가 비면(=지표 컬럼 생성 실패) 전체 dropna로라도 진행
        out = out.dropna()

    return out