"""Fixed daily fresh-entry experiment; no signal or risk parameter search."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from qkquant import etf_portfolio_backtest as engine
from qkquant.data.storage import DuckStore
from qkquant.etf_drawdown import DrawdownConfig
from qkquant.etf_signal import EtfSignalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base, risk = EtfSignalConfig(), DrawdownConfig()
    candidates = [('scheduled', False, base), ('daily', True, base),
                  ('daily_stress', True, replace(base, slippage_pct=.004, commission_min=5))]
    original = engine.classify_chan_structure
    cache = {}
    def cached(close, high, low, order, tolerance):
        # All runs below use the identical immutable historical panel. Cache
        # only classification, never scores, holdings or trade decisions.
        key = (close.name, close.index[-1], len(close), order, tolerance)
        if key not in cache:
            cache[key] = original(close, high, low, order, tolerance)
        return cache[key]
    engine.classify_chan_structure = cached
    rows = []
    try:
        with DuckStore(read_only=True) as store:
            coverage = store.conn.execute("""
                SELECT d.trade_date, count(*) AS codes FROM daily_bars d
                JOIN instruments i USING(code)
                WHERE i.etf_category='equity' AND d.adjust='qfq'
                GROUP BY 1 ORDER BY 1
            """).df()
            end = str(coverage.loc[coverage.codes >= coverage.codes.max() * .9, 'trade_date'].iloc[-1].date())
            for name, daily, cfg in candidates:
                for offset in range(10):
                    result = engine.run_etf_portfolio_backtest(store, start='2024-01-01', end=end,
                        rebalance_days=10, rebalance_offset=offset, config=cfg,
                        risk_config=risk, daily_entries=daily)
                    rows.append(dict(candidate=name, offset=offset, **result['strategy'],
                        trade_count=result['trade_count'], daily_buys=sum(t['reason']=='daily_entry' for t in result['trades']),
                        mean_exposure=float(result['exposure'].mean()), min_cash=float(result['cash'].min()),
                        commission=result['commission_total'], turnover=result['turnover_ratio']))
                    stem = f'{name}_{offset}'
                    pd.DataFrame(dict(equity=result['equity'], cash=result['cash'], exposure=result['exposure'])).to_csv(args.output/f'{stem}_equity.csv')
                    pd.DataFrame(result['trades']).to_csv(args.output/f'{stem}_trades.csv', index=False)
                    pd.DataFrame(rows).to_csv(args.output/'comparison.csv', index=False)
                    print(name, offset, rows[-1], flush=True)
            (args.output/'metadata.json').write_text(json.dumps(dict(start=result['start'],end=end,
                config=asdict(base),risk_config=asdict(risk),
                research='same previously examined data; not independent OOS; no threshold search'),indent=2),encoding='utf-8')
    finally:
        engine.classify_chan_structure = original


if __name__ == '__main__':
    main()
