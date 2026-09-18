"""Compare frozen old/new Chan classifiers without tuning thresholds."""

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from qkquant import etf_portfolio_backtest as engine
from qkquant.data.storage import DuckStore
from qkquant.etf_drawdown import DrawdownConfig
from qkquant.etf_signal import EtfSignalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True,
                        help='frozen pre-fix etf_chan.py (research only)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order-only', action='store_true', help='diagnostic ablation: chronology only')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('chan_before_fix', args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    fixed = engine.classify_chan_structure
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    cfg = EtfSignalConfig()
    def ordered_only(*values, **kwargs):
        signal = fixed(*values, **kwargs)
        return replace(signal, state='third_buy', allowed=True) if signal.state == 'third_buy_active' else signal
    versions = [('order_only', ordered_only)] if args.order_only else [('before', baseline.classify_chan_structure), ('after', fixed)]
    try:
        with DuckStore(read_only=True) as store:
            coverage = store.conn.execute("""
                SELECT d.trade_date, count(*) AS codes FROM daily_bars d
                JOIN instruments i USING(code)
                WHERE i.etf_category='equity' AND d.adjust='qfq'
                GROUP BY 1 ORDER BY 1
            """).df()
            end = str(coverage.loc[coverage.codes >= coverage.codes.max() * .9, 'trade_date'].iloc[-1].date())
            for label, classifier in versions:
                engine.classify_chan_structure = classifier
                for risk_name, risk in [('no_risk', None), ('daily_risk', DrawdownConfig())]:
                    for offset in range(10):
                        result = engine.run_etf_portfolio_backtest(store, start='2024-01-01', end=end,
                            rebalance_days=10, rebalance_offset=offset, config=cfg, risk_config=risk)
                        rows.append(dict(version=label, risk=risk_name, offset=offset,
                            **result['strategy'], trade_count=result['trade_count'],
                            mean_exposure=float(result['exposure'].mean())))
                        stem = f'{label}_{risk_name}_{offset}'
                        pd.DataFrame(result['trades']).to_csv(args.output / f'{stem}_trades.csv', index=False)
                        pd.DataFrame(dict(equity=result['equity'], exposure=result['exposure'])).to_csv(args.output / f'{stem}_equity.csv')
                    pd.DataFrame(rows).to_csv(args.output / 'comparison.csv', index=False)
                    print(label, risk_name, rows[-10], flush=True)
            (args.output / 'metadata.json').write_text(json.dumps(dict(
                start=result['start'], end=result['end'], rebalance_days=10,
                note='retrospective diagnosis, same data and costs, no parameter search or independent OOS'), indent=2), encoding='utf-8')
    finally:
        engine.classify_chan_structure = fixed


if __name__ == '__main__':
    main()
