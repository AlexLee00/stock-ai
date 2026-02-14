import pandas as pd

from core.portfolio import (
    compute_returns,
    inverse_vol_weights,
    apply_cash_gate,
    normalize_weights,
)


def rank_assets(featured_map: dict[str, pd.DataFrame], top_n: int = 5) -> pd.DataFrame:
    rows: list[dict] = []

    for ticker, df in featured_map.items():
        if df is None or df.empty:
            continue

        needed = ["gated_score", "rank_score", "risk_regime"]
        if any(c not in df.columns for c in needed):
            continue

        valid = (
            df.replace([float("inf"), float("-inf")], pd.NA)
              .dropna(subset=["gated_score"])
        )
        if valid.empty:
            continue

        last = valid.iloc[-1]

        date_val = None
        if isinstance(valid.index, pd.DatetimeIndex):
            date_val = str(valid.index[-1].date())

        rows.append(
            {
                "ticker": ticker,
                "gated_score": float(last["gated_score"]),
                "rank_score": float(last["rank_score"]),
                "regime": str(last["risk_regime"]),
                "date": date_val,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ticker", "gated_score", "rank_score", "regime", "date"])

    ranked = (
        pd.DataFrame(rows)
        .sort_values(["gated_score", "rank_score"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return ranked


def build_price_matrix(featured_map: dict[str, pd.DataFrame], price_col: str = "Close") -> pd.DataFrame:
    """
    여러 종목 Close를 합쳐 price_df 생성.
    - 종목 수가 늘어날수록 공통날짜만 남기는 dropna()는 데이터 손실이 큼
    - concat 후 ffill, 맨 앞 구간 NaN 제거 방식 추천
    """
    series_list = []
    for ticker, df in featured_map.items():
        if df is None or df.empty or price_col not in df.columns:
            continue

        s = df[price_col].rename(ticker)
        series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    price_df = pd.concat(series_list, axis=1).sort_index()

    # forward fill로 휴일/결측 일부 완화 (상장 전 구간은 여전히 NaN)
    price_df = price_df.ffill()

    # 맨 앞 NaN 남은 구간 제거
    price_df = price_df.dropna(how="any")

    return price_df


def latest_risk_off(featured_map: dict[str, pd.DataFrame], market_ticker: str = "SPY") -> bool:
    df = featured_map.get(market_ticker)
    if df is not None and not df.empty and "risk_regime" in df.columns:
        valid = df.dropna(subset=["risk_regime"])
        if not valid.empty:
            last = valid.iloc[-1]
            return str(last["risk_regime"]) == "off"

    # fallback: 첫 종목 기준
    for _, df in featured_map.items():
        if df is None or df.empty or "risk_regime" not in df.columns:
            continue
        valid = df.dropna(subset=["risk_regime"])
        if valid.empty:
            continue
        last = valid.iloc[-1]
        return str(last["risk_regime"]) == "off"

    return False


def backtest_simple(
    featured_map: dict[str, pd.DataFrame],
    top_n: int = 3,
    rebalance_every_days: int = 21,
    vol_window: int = 60,
    cash_ticker: str = "CASH",
    market_ticker: str = "SPY",
    fee_bps: float = 5.0,         # ✅ 왕복/편도는 프로젝트 룰로 정하자 (일단 5bp)
    slippage_bps: float = 5.0,    # ✅ 단순 슬리피지
) -> pd.DataFrame:
    """
    단순 백테스트:
    - 리밸런싱: gated_score top_n -> inverse-vol weights
    - 리스크오프: market_ticker 기준 off면 CASH 100%
    - 거래비용: 리밸런싱 시 turnover 기반으로 port_value에서 차감

    가정:
    - 리밸런싱 시점 i의 일간 수익률도 새 비중으로 적용(장 시작 리밸런싱 가정)
      * 종가 리밸런싱으로 바꾸려면 period를 i+1부터 적용하면 됨
    """
    price_df = build_price_matrix(featured_map)
    if price_df.empty:
        return pd.DataFrame()

    returns = compute_returns(price_df)
    returns[cash_ticker] = 0.0

    dates = returns.index
    port_value = 1.0
    rows = []

    prev_w: dict[str, float] | None = None

    warmup = max(vol_window, 60)  # ✅ 지표 안정화까지 포함
    start = warmup if warmup < len(dates) else 0

    for i in range(start, len(dates), rebalance_every_days):
        dt = dates[i]

        # 스냅샷
        snap_map: dict[str, pd.DataFrame] = {}
        for ticker, df in featured_map.items():
            if df is None or df.empty:
                continue
            try:
                snap_df = df.loc[:dt]
            except Exception:
                snap_df = df
            snap_map[ticker] = snap_df

        # 랭킹
        ranked = rank_assets(snap_map, top_n=top_n)
        selected = ranked["ticker"].tolist() if not ranked.empty else []

        # 배분 (inverse vol)
        hist_returns = returns.loc[:dt, selected].dropna()
        if hist_returns.empty or len(selected) == 0:
            w = pd.Series({cash_ticker: 1.0})
        else:
            w = inverse_vol_weights(hist_returns, window=vol_window)
            w = normalize_weights(w)

        # 리스크 게이트
        risk_off = latest_risk_off(snap_map, market_ticker=market_ticker)
        w = apply_cash_gate(w, risk_off=risk_off, cash_ticker=cash_ticker)

        w_dict = {str(k): float(v) for k, v in w.to_dict().items()}

        # turnover (리밸런싱 이벤트)
        turnover = 0.0
        if prev_w is not None:
            all_keys = set(prev_w.keys()) | set(w_dict.keys())
            turnover = sum(abs(prev_w.get(k, 0.0) - w_dict.get(k, 0.0)) for k in all_keys)

        prev_w = w_dict

        # ✅ 거래비용: 리밸런싱 당일에만 차감
        cost_rate = turnover * ((fee_bps + slippage_bps) / 10000.0)
        if cost_rate > 0:
            port_value *= (1.0 - cost_rate)

        next_i = min(i + rebalance_every_days, len(dates))
        period = returns.iloc[i:next_i]

        for day, r in period.iterrows():
            day_ret = 0.0
            for t, weight in w.items():
                day_ret += float(weight) * float(r.get(t, 0.0))

            port_value *= (1.0 + day_ret)

            rows.append(
                {
                    "date": day,
                    "port_ret": day_ret,
                    "port_value": port_value,
                    "risk_off": risk_off,
                    "turnover": turnover,
                    "cost_rate": cost_rate,
                    "weights": w_dict,
                    "selected": selected,
                    "rebalance_date": dt,
                }
            )

    return pd.DataFrame(rows).set_index("date")