"""Fixed signal-value diagnostics, not a new production strategy."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from qkquant.config import PROJECT_ROOT
from qkquant.data.storage import DuckStore
from qkquant.etf_signal import EtfSignalConfig
from qkquant.etf_signal_diagnostics import collect_diagnostics, summarize
from qkquant.factors.pipeline import load_panel


REQUIRED_FROZEN_SOURCES = (
    'src/qkquant/etf_signal.py', 'src/qkquant/etf_chan.py',
    'src/qkquant/etf_drawdown.py', 'src/qkquant/etf_portfolio_backtest.py',
    'src/qkquant/factors/library.py', 'src/qkquant/factors/pipeline.py',
    'src/qkquant/etf_signal_diagnostics.py', 'scripts/research_signal_value.py',
    'config/settings.yaml', 'docs/SIGNAL_VALUE_PLAN.md',
)
EVENT_PROTOCOL = dict(main_horizon=10, secondary_horizon=20, spacing_bars=20,
                      budget=100000*.4/3, stress_min_commission=5, stress_slippage=.004)


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_manifest(manifest, cfg):
    hashes = manifest.get('source_sha256', {})
    if not set(REQUIRED_FROZEN_SOURCES).issubset(hashes):
        raise ValueError('Freeze manifest is missing required rule source hashes.')
    for relative, expected in hashes.items():
        source = (PROJECT_ROOT / relative).resolve()
        if not source.is_relative_to(PROJECT_ROOT.resolve()) or not source.is_file():
            raise ValueError(f'Invalid frozen source: {relative}')
        if file_sha256(source) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    if manifest.get('config') != json.loads(json.dumps(asdict(cfg))):
        raise ValueError('Signal config differs from the freeze manifest.')
    if manifest.get('event_protocol') != EVENT_PROTOCOL:
        raise ValueError('Event protocol differs from the freeze manifest.')
    codes = manifest.get('universe', {}).get('codes', [])
    if (not codes or len(set(codes)) != len(codes)
            or manifest['universe'].get('count') != len(codes)):
        raise ValueError('Frozen universe codes/count are invalid.')
    date.fromisoformat(manifest['latest_completed_date_cap'])


def frozen_panel(store, manifest):
    codes = manifest['universe']['codes']
    cap = manifest['latest_completed_date_cap']
    panel = load_panel(store, codes, start='2024-01-01', end=cap, adjust='qfq')
    last_bar = panel['close'].index[-1]
    calendar = pd.DatetimeIndex(pd.to_datetime(store.load_calendar(start='2024-01-01', end=cap)))
    if (calendar.empty or calendar[-1] < last_bar
            or not panel['close'].index.isin(calendar).all()):
        raise ValueError('Frozen validation requires a trading calendar covering the supplied bars.')
    # Keep missing sessions on the time axis so holding periods do not become shorter.
    calendar = calendar[(calendar >= panel['close'].index[0]) & (calendar <= last_bar)]
    panel = {field: frame.reindex(index=calendar, columns=codes) for field, frame in panel.items()}
    counts = panel['close'].notna().sum(axis=1)
    coverage = pd.DataFrame({'codes': counts, 'frozen_codes': len(codes),
                             'coverage_fraction': counts / len(codes), 'eligible': counts >= len(codes)*.9})
    coverage.index.name = 'trade_date'
    if not coverage.eligible.any():
        raise ValueError('No date reaches 90% coverage of the frozen universe.')
    end = coverage.index[coverage.eligible][-1]
    panel = {field: frame.loc[:end] for field, frame in panel.items()}
    return panel, coverage, str(end.date())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation-start', type=date.fromisoformat,
                        help='Evaluate events from this date, retaining earlier signals for de-duplication.')
    parser.add_argument('--freeze-manifest', type=Path)
    parser.add_argument('--database', type=Path, help='Read-only database, including an isolated validation snapshot.')
    args = parser.parse_args()
    frozen = args.freeze_manifest is not None
    report_files = ('date_coverage.csv', 'all_candidates.csv', 'summary.csv', 'spaced_events.csv', 'metadata.json')
    if frozen and any((args.output/name).exists() for name in report_files):
        raise ValueError('Frozen validation reports cannot be overwritten; use a new output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = EtfSignalConfig()
    manifest = json.loads(args.freeze_manifest.read_text(encoding='utf-8')) if frozen else None
    if frozen:
        validate_manifest(manifest, cfg)
    database = args.database or (Path(manifest['existing_database']['path']) if frozen else None)
    if database is not None and not database.is_file():
        raise ValueError(f'Database does not exist: {database}')
    eligible_dates = None
    with DuckStore(database, read_only=True) as store:
        database_path = str(store.path)
        if frozen:
            panel, coverage, end = frozen_panel(store, manifest)
            eligible_dates = coverage.eligible
        else:
            coverage = store.conn.execute("""
            SELECT d.trade_date, count(*) AS codes FROM daily_bars d
            JOIN instruments i USING(code)
            WHERE i.etf_category='equity' AND d.adjust='qfq'
            GROUP BY 1 ORDER BY 1
            """).df()
            if coverage.empty:
                raise ValueError('no equity ETF history')
            end = str(coverage.loc[coverage.codes >= coverage.codes.max() * .9, 'trade_date'].iloc[-1].date())
            panel = load_panel(store, store.load_etf_codes('equity'), start='2024-01-01', end=end)
    coverage.to_csv(args.output/'date_coverage.csv', index=frozen)
    events = collect_diagnostics(panel, cfg, eligible_dates=eligible_dates,
        progress=lambda p,n,count: print(f'{p}/{n} bars, {count} eligible observations',flush=True))
    summary, selected = summarize(events, evaluation_start=args.evaluation_start)
    evaluation_start = args.evaluation_start.isoformat() if args.evaluation_start else None
    evaluated = events if evaluation_start is None else events[events.date >= evaluation_start]
    events.to_csv(args.output/'all_candidates.csv', index=False)
    summary.to_csv(args.output/'summary.csv', index=False)
    selected.to_csv(args.output/'spaced_events.csv', index=False)
    meta = dict(data_start=str(panel['close'].index[0].date()), data_end=end,
        first_signal=events.date.min(), last_signal=events.date.max(), candidate_count=len(events),
        config=asdict(cfg), independent_oos=False, main_horizon=10,
        database=database_path, database_sha256=file_sha256(Path(database_path)),
        freeze_manifest=str(args.freeze_manifest) if frozen else None,
        manifest_sha256=file_sha256(args.freeze_manifest) if frozen else None,
        price_adjustment_verified=False, data_source_provenance='not_verified_by_this_tool',
        frozen_universe=manifest['universe']['codes'] if frozen else None,
        latest_completed_date_cap=manifest['latest_completed_date_cap'] if frozen else None,
        low_coverage_dates=[str(day.date()) for day in coverage.index[~coverage.eligible]] if frozen else [],
        coverage_note='Frozen mode excludes candidate dates below 90% coverage without removing trading sessions; '
                      'end is the last qualifying date no later than the frozen cap. See date_coverage.csv.',
        evaluation_start=evaluation_start, evaluation_candidate_count=len(evaluated),
        pending_counts={f'{cost}_{horizon}': int((evaluated[f'{cost}_status_{horizon}'] == 'pending').sum())
                        for cost in ('base', 'stress') for horizon in (10, 20)},
        evaluation_note='Date filtering alone does not establish independent out-of-sample evidence; '
                        'pending outcomes are retained and excluded from return averages; '
                        '20-day spacing is applied before the evaluation-date filter.',
        execution='t+1 open buy; t+11 or t+21 open sell; 100-share lots; fixed per-event budget 100000*0.4/3')
    (args.output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[(summary['sample']=='spaced')&(summary.year=='all')&(summary.horizon==10)].to_string(index=False))


if __name__ == '__main__':
    main()
