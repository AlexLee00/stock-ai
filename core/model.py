import numpy as np
import pandas as pd


def zscore(s: pd.Series, window: int = 60) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - mean) / std


def minmax_rolling(s: pd.Series, window: int = 252) -> pd.Series:
    """
    롤링 min-max로 0~1 스케일.
    (추후 백테스트에서 lookahead 피하려면 window 고정/확정 필요)
    """
    rmin = s.rolling(window).min()
    rmax = s.rolling(window).max()
    denom = (rmax - rmin).replace(0, np.nan)
    return (s - rmin) / denom


def sigmoid(x: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-x))


def compute_signal_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    add_quant_features 이후 df 기준
    필요 컬럼: macd_hist, bb_percent_b, bb_bandwidth
    결과 컬럼:
      macd_z, bb_pb_score, vol_penalty, raw_score, rank_score(0~100),
      risk_regime(on/off), gated_score(0~100)
    """
    out = df.copy()

    required = ["macd_hist", "bb_percent_b", "bb_bandwidth"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ===== 1) 트렌드: MACD hist z-score =====
    out["macd_z"] = zscore(out["macd_hist"].astype(float), window=60).fillna(0.0)

    # ===== 2) Mean-reversion 힌트: Bollinger %B =====
    pb = out["bb_percent_b"].astype(float).clip(-5, 5)
    out["bb_pb_score"] = np.where(pb < 0.1, 1.0, np.where(pb > 0.9, -1.0, 0.0))

    # ===== 3) 변동성 패널티: Bollinger bandwidth z-score =====
    bw_z = zscore(out["bb_bandwidth"].astype(float), window=60).fillna(0.0)
    out["vol_penalty"] = np.clip(bw_z, 0, None)  # 0 이상만 패널티

    # ===== 4) raw_score (가중합) =====
    w_macd = 0.7
    w_bb = 0.3
    w_vol = 0.4

    out["raw_score"] = (
        w_macd * out["macd_z"]
        + w_bb * out["bb_pb_score"]
        - w_vol * out["vol_penalty"]
    )

    # ===== 5) 랭킹 점수(0~100) =====
    # 방법1(안정): 시그모이드로 0~1 압축 후 0~100
    # raw_score가 극단값이어도 점수 폭주 방지
    out["rank_score"] = (sigmoid(out["raw_score"]) * 100).clip(0, 100)

    # ===== 6) 리스크 게이트(레짐) =====
    # 아이디어: 변동성 급등 + 추세 약화 → Risk-Off
    # - vol_penalty가 크면(예: bw_z 양수로 급등) off
    # - macd_z가 음수(추세 약함)일 때 off 확률 증가
    #
    # 기준은 “학습용”이니 단순 룰로 시작하고, 다음 단계에서 튜닝/학습으로 진화시키자.
    vol_gate = out["vol_penalty"] > 1.0          # 변동성 급등
    trend_break = out["macd_z"] < -0.5           # 추세 붕괴 느낌
    out["risk_regime"] = np.where(vol_gate & trend_break, "off", "on")

    # off면 점수를 크게 깎는다(=현금/단기채로 보내는 효과)
    out["gated_score"] = np.where(out["risk_regime"] == "off", out["rank_score"] * 0.2, out["rank_score"])
    out["gated_score"] = out["gated_score"].clip(0, 100)

    return out