"""Fixed 10-day ETF strategy: compare daily risk control over all 10 phases."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from qkquant.data.storage import DuckStore
from qkquant.etf_drawdown import DrawdownConfig
from qkquant.etf_portfolio_backtest import run_etf_portfolio_backtest
from qkquant.etf_signal import EtfSignalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/etf_drawdown_research'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg, risk = EtfSignalConfig(), DrawdownConfig()
    candidates = [('baseline', cfg, None), ('daily_risk', cfg, risk),
                  ('daily_risk_stress', replace(cfg, slippage_pct=.004, commission_min=5), risk)]
    rows = []
    with DuckStore(read_only=True) as store:
        coverage = store.conn.execute("""
            SELECT d.trade_date, count(*) AS codes FROM daily_bars d
            JOIN instruments i USING(code)
            WHERE i.etf_category='equity' AND d.adjust='qfq'
            GROUP BY 1 ORDER BY 1
        """).df()
        if coverage.empty:
            raise ValueError('no equity ETF history')
        end = str(coverage.loc[coverage.codes >= coverage.codes.max() * .9, 'trade_date'].iloc[-1].date())
        coverage.to_csv(args.output / 'coverage.csv', index=False)
        for name, signal_cfg, risk_cfg in candidates:
            for offset in range(10):
                result = run_etf_portfolio_backtest(store, start='2024-01-01', end=end,
                    rebalance_days=10, rebalance_offset=offset, config=signal_cfg, risk_config=risk_cfg)
                rows.append(dict(candidate=name, offset=offset, **result['strategy'],
                    turnover=result['turnover_ratio'], trades=result['trade_count'],
                    min_cash=float(result['cash'].min()), average_exposure=float(result['exposure'].mean()),
                    halted=any(row['halted'] for row in result['risk_history'])))
                stem = f'{name}_{offset}'
                pd.DataFrame(dict(equity=result['equity'], cash=result['cash'], exposure=result['exposure'])).to_csv(args.output / f'{stem}_equity.csv')
                pd.DataFrame(result['trades']).to_csv(args.output / f'{stem}_trades.csv', index=False)
                if result['risk_history']:
                    pd.DataFrame(result['risk_history']).to_csv(args.output / f'{stem}_risk.csv', index=False)
                print(name, offset, result['strategy'], flush=True)
            pd.DataFrame(rows).to_csv(args.output / 'comparison.csv', index=False)
        meta = dict(start=result['start'], end=result['end'], signal=asdict(cfg), risk=asdict(risk),
                    methodology='Retrospective fixed hypotheses; all phases share history, not independent OOS.',
                    stress=dict(slippage_pct=.004, commission_min=5))
        (args.output / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
