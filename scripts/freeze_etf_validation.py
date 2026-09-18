"""Freeze the current ETF research rules and audited existing universe, read-only."""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from qkquant.config import PROJECT_ROOT
from qkquant.data.storage import DuckStore
from qkquant.etf_drawdown import DrawdownConfig
from qkquant.etf_signal import EtfSignalConfig


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'freeze_manifest.json'
    if manifest_path.exists():
        raise SystemExit('Use a new output directory; never overwrite an existing freeze.')
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    with DuckStore(read_only=True) as store:
        coverage = store.conn.execute("""
            SELECT d.code, i.name, i.exchange, i.list_date, i.delist_date,
                   min(d.trade_date) AS first_bar, max(d.trade_date) AS last_bar,
                   count(*) AS bars
            FROM daily_bars d JOIN instruments i USING(code)
            WHERE i.etf_category='equity' AND d.adjust='qfq'
            GROUP BY 1,2,3,4,5 ORDER BY 1
        """).df()
        dates = store.conn.execute("""
            SELECT d.trade_date, count(*) AS codes
            FROM daily_bars d JOIN instruments i USING(code)
            WHERE i.etf_category='equity' AND d.adjust='qfq'
            GROUP BY 1 ORDER BY 1
        """).df()
        if coverage.empty:
            raise SystemExit('No existing equity ETF history to freeze.')
        db_path = store.path
        metadata_count = store.conn.execute("SELECT count(*) FROM instruments WHERE etf_category='equity'").fetchone()[0]
    coverage.to_csv(args.output / 'frozen_universe.csv', index=False)
    dates.to_csv(args.output / 'existing_date_coverage.csv', index=False)
    latest_seen = coverage.last_bar.max().date()
    reliable_end = dates.loc[dates.codes >= len(coverage) * .9, 'trade_date'].max().date()
    source_paths = [
        'src/qkquant/etf_signal.py', 'src/qkquant/etf_chan.py',
        'src/qkquant/etf_drawdown.py', 'src/qkquant/etf_portfolio_backtest.py',
        'src/qkquant/factors/library.py', 'src/qkquant/factors/pipeline.py',
        'src/qkquant/etf_signal_diagnostics.py', 'scripts/research_signal_value.py',
        'src/qkquant/data/fetcher.py',
        'src/qkquant/data/storage.py', 'src/qkquant/config.py',
        'config/settings.yaml', 'docs/SIGNAL_VALUE_PLAN.md',
    ]
    hashes = {}
    for relative in source_paths:
        source = PROJECT_ROOT / relative
        destination = args.output / 'frozen_sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[relative] = digest(destination)
    manifest = dict(
        created_at=now.isoformat(), project_root=str(PROJECT_ROOT),
        strategy_status='research_reference_only_not_validated',
        config=asdict(EtfSignalConfig()), drawdown_config=asdict(DrawdownConfig()),
        portfolio=dict(rebalance_days=10, rebalance_offset=0, daily_entries=False),
        event_protocol=dict(main_horizon=10, secondary_horizon=20, spacing_bars=20,
                            budget=100000*.4/3, stress_min_commission=5, stress_slippage=.004),
        existing_database=dict(path=str(db_path), sha256=digest(db_path),
                               source_provenance='unknown', adjustment_label='qfq_unverified'),
        universe=dict(codes=coverage.code.tolist(), count=len(coverage),
                      metadata_equity_count=metadata_count,
                      missing_list_date=int(coverage.list_date.isna().sum()),
                      point_in_time_membership_verified=False,
                      universe_sha256=digest(args.output / 'frozen_universe.csv')),
        development_reliable_coverage_through=str(reliable_end),
        previously_seen_any_bar_through=str(latest_seen),
        historical_extension_start=str(latest_seen + timedelta(days=1)),
        prospective_after=now.isoformat(),
        latest_completed_date_cap=str(now.date() - timedelta(days=1)),
        independent_oos=False,
        data_gates=['one documented adjustment convention', 'full-history refresh in quarantine',
                    'no raw/adjusted concatenation', 'OHLC and duplicate checks',
                    'at least 90 percent of frozen universe per evaluated date',
                    'missing listings/delistings remain an explicit limitation'],
        source_sha256=hashes,
    )
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: manifest[key] for key in (
        'created_at', 'development_reliable_coverage_through',
        'previously_seen_any_bar_through', 'historical_extension_start')}, ensure_ascii=False))
    print(f'Frozen ETF count: {len(coverage)}; missing listing dates: {coverage.list_date.isna().sum()}')


if __name__ == '__main__':
    main()
