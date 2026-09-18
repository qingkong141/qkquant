"""Fixed, event-level ETF entry diagnostics. Not a portfolio backtest."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from qkquant.etf_chan import _fractals, classify_chan_structure
from qkquant.etf_signal import EtfSignalConfig
from qkquant.factors.library import trend_quality_60d


def entry_events(close, high, low, cfg):
    """Only past bars; nested plain pullback isolates extra Chan classification."""
    signal = classify_chan_structure(close, high, low, cfg.chan_fractal_order, cfg.chan_tolerance)
    frame = pd.concat({'close': close, 'high': high, 'low': low}, axis=1)
    if frame.empty or frame.iloc[-1].isna().any():
        return {'chan': False, 'pullback': False, 'trend': False}
    frame = frame.dropna().tail(200)
    highs, _ = _fractals(frame.high, cfg.chan_fractal_order)
    _, lows = _fractals(frame.low, cfg.chan_fractal_order)
    ma = frame.close.rolling(20).mean()
    plain = False
    if len(highs) >= 2 and len(lows) >= 2 and len(frame) >= 30:
        plain = bool(
            highs.index[-2] < highs.index[-1] < lows.index[-1]
            and frame.index.get_loc(lows.index[-1]) + cfg.chan_fractal_order == len(frame) - 1
            and highs.iloc[-1] > highs.iloc[-2] * (1 + cfg.chan_tolerance)
            and lows.iloc[-1] >= highs.iloc[-2] * (1 - cfg.chan_tolerance)
            and frame.close.iloc[-1] > lows.iloc[-1]
            and frame.close.iloc[-1] > ma.iloc[-1]
        )
    trend = bool(len(frame) >= 21 and frame.close.iloc[-1] > ma.iloc[-1]
                 and frame.close.iloc[-2] <= ma.iloc[-2])
    return {'chan': signal.state == 'third_buy', 'pullback': plain, 'trend': trend}


def event_return(panel, code, position, horizon, cfg, budget=100_000 * .4 / 3):
    """t+1 open -> t+horizon+1 open; fixed budget, lots and costs."""
    result = event_outcome(panel, code, position, horizon, cfg, budget)
    return result if result['status'] == 'valid' else None


def event_outcome(panel, code, position, horizon, cfg, budget=100_000 * .4 / 3):
    """Keep unfinished and unexecutable observations separate from valid returns."""
    entry_pos, exit_pos = position + 1, position + horizon + 1
    if entry_pos >= len(panel['open']):
        return dict(status='pending', reason='entry_not_observed')
    entry = panel['open'][code].iloc[entry_pos]
    amount = panel['amount'][code]
    if not np.isfinite(entry) or not np.isfinite(amount.iloc[entry_pos]):
        return dict(status='missing', reason='entry_price_or_amount_missing')
    if entry <= 0 or not amount.iloc[entry_pos] > 0:
        return dict(status='unexecutable', reason='nonpositive_entry_price_or_amount')
    buy_price = entry * (1 + cfg.slippage_pct)
    available = min(max(0, budget - cfg.commission_min), budget / (1 + cfg.commission_rate))
    qty = int(available / buy_price / 100) * 100
    if qty <= 0:
        return dict(status='unexecutable', reason='budget_below_one_lot')
    if exit_pos >= len(panel['open']):
        return dict(status='pending', reason='holding_window_incomplete')
    exit_ = panel['open'][code].iloc[exit_pos]
    lows = panel['low'][code].iloc[entry_pos:exit_pos]
    if (not np.isfinite(exit_) or not np.isfinite(amount.iloc[exit_pos]) or lows.isna().any()):
        return dict(status='missing', reason='exit_price_amount_or_path_missing')
    if exit_ <= 0 or not amount.iloc[exit_pos] > 0:
        return dict(status='unexecutable', reason='nonpositive_exit_price_or_amount')
    sell_price = exit_ * (1 - cfg.slippage_pct)
    buy_value, sell_value = qty * buy_price, qty * sell_price
    fees = max(buy_value * cfg.commission_rate, cfg.commission_min) + max(sell_value * cfg.commission_rate, cfg.commission_min)
    return dict(status='valid', reason='', gross=float(exit_ / entry - 1), net=float((sell_value - buy_value - fees) / budget),
                mae=float(min(0, min(lows.min(), exit_) / entry - 1)), qty=qty)


def collect_diagnostics(panel, cfg=None, progress=None, eligible_dates=None):
    cfg = cfg or EtfSignalConfig()
    stress_cfg = replace(cfg, commission_min=5, slippage_pct=.004)
    close, high, low, amount = (panel[k] for k in ('close', 'high', 'low', 'amount'))
    ret = close.pct_change(fill_method=None)
    mom60, mom120 = close.pct_change(60, fill_method=None), close.pct_change(120, fill_method=None)
    risk_mom = mom60 / ret.rolling(60, min_periods=60).std().replace(0, np.nan)
    trend_quality = trend_quality_60d(panel)
    downside = ret.clip(upper=0).rolling(20, min_periods=20).std()
    ma120 = close.rolling(120, min_periods=120).mean()
    amount20 = amount.rolling(20, min_periods=20).mean()
    if eligible_dates is not None:
        eligible_dates = eligible_dates.reindex(close.index, fill_value=False)
    rows = []
    for pos in range(120, len(close)):
        if eligible_dates is not None and not eligible_dates.iloc[pos]:
            continue
        valid = close.iloc[pos].notna() & ma120.iloc[pos].notna()
        breadth = (close.iloc[pos][valid] > ma120.iloc[pos][valid]).mean()
        if not breadth >= cfg.risk_off_breadth:
            continue
        frame = pd.DataFrame(dict(mom60=mom60.iloc[pos], mom120=mom120.iloc[pos],
            risk_mom=risk_mom.iloc[pos], quality=trend_quality.iloc[pos], downside=downside.iloc[pos],
            liquidity=amount20.iloc[pos], above=close.iloc[pos] > ma120.iloc[pos])).dropna()
        frame = frame[(frame.mom60 > 0) & (frame.mom120 > 0) & frame.above & (frame.liquidity >= cfg.min_amount_20d)]
        score = .35 * frame.risk_mom.rank(pct=True) + .25 * frame.mom120.rank(pct=True) + .20 * frame.quality.rank(pct=True) + .20 * (1 - frame.downside.rank(pct=True))
        for code in frame.index[score >= cfg.score_threshold]:
            flags = entry_events(close[code].iloc[:pos+1], high[code].iloc[:pos+1], low[code].iloc[:pos+1], cfg)
            row = dict(date=str(close.index[pos].date()), position=pos, code=code, **flags)
            for horizon in (10, 20):
                for cost_name, cost_cfg in [('base', cfg), ('stress', stress_cfg)]:
                    result = event_outcome(panel, code, pos, horizon, cost_cfg)
                    row[f'{cost_name}_status_{horizon}'] = result['status']
                    row[f'{cost_name}_reason_{horizon}'] = result['reason']
                    for metric in ('gross', 'net', 'mae'):
                        row[f'{cost_name}_{metric}_{horizon}'] = result.get(metric, np.nan)
            rows.append(row)
        if progress and pos % 100 == 0:
            progress(pos, len(close), len(rows))
    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError('no eligible events')
    # Same-date pool comparisons adjust only date composition, not other factors.
    for horizon in (10, 20):
        for cost in ('base', 'stress'):
            col = f'{cost}_net_{horizon}'
            events[f'{cost}_edge_{horizon}'] = events[col] - events.groupby('date')[col].transform('mean')
    return events


def summarize(events, evaluation_start=None):
    """De-duplicate using the full signal history, then select the evaluation dates."""
    rows, selected_rows = [], []
    cutoff = str(pd.Timestamp(evaluation_start).date()) if evaluation_start is not None else None
    evaluated = events if cutoff is None else events[events.date >= cutoff]
    years = ['all', *sorted(evaluated.date.str[:4].unique())]
    masks = {name: events[name] for name in ('chan', 'pullback', 'trend')}
    masks['pullback_without_chan'] = events.pullback & ~events.chan
    for name, mask in masks.items():
        raw = events[mask].sort_values(['position', 'code'])
        last, kept = {}, []
        for idx, row in raw.iterrows():
            if row.position - last.get(row.code, -1000) > 20:
                kept.append(idx)
                last[row.code] = row.position
        chosen = events.loc[kept].copy()
        if cutoff is not None:
            raw = raw[raw.date >= cutoff]
            chosen = chosen[chosen.date >= cutoff]
        chosen['group'] = name
        selected_rows.append(chosen)
        for sample, subset in [('all', raw), ('spaced', chosen)]:
            for year in years:
                piece = subset if year == 'all' else subset[subset.date.str.startswith(year)]
                for horizon in (10, 20):
                    for cost in ('base', 'stress'):
                        valid = piece.dropna(subset=[f'{cost}_net_{horizon}'])
                        statuses = piece.get(f'{cost}_status_{horizon}',
                            pd.Series(np.where(piece[f'{cost}_net_{horizon}'].notna(), 'valid', 'missing'), index=piece.index))
                        values = valid[f'{cost}_net_{horizon}']
                        edge = valid[f'{cost}_edge_{horizon}']
                        rows.append(dict(group=name, sample=sample, year=year, horizon=horizon, cost=cost,
                            events=len(piece), n=len(valid), dates=valid.date.nunique(), etfs=valid.code.nunique(), mean_net=values.mean(), median_net=values.median(),
                            pending=int((statuses == 'pending').sum()), missing=int((statuses == 'missing').sum()),
                            unexecutable=int((statuses == 'unexecutable').sum()),
                            positive_share=(values > 0).mean() if len(valid) else np.nan,
                            mean_mae=valid[f'{cost}_mae_{horizon}'].mean(),
                            mean_edge=edge.mean(), date_equal_edge=valid.assign(edge=edge).groupby('date').edge.mean().mean()))
    return pd.DataFrame(rows), pd.concat(selected_rows, ignore_index=True)
