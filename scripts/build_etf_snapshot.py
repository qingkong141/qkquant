"""Build an isolated, cross-checked ETF price snapshot for a frozen universe."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from qkquant.data.storage import DuckStore


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def retrieve(folder, name, url, params, offline):
    """Cache response text with its source, request and retrieval timestamp."""
    target, provenance = folder/f'{name}.txt', folder/f'{name}.source.json'
    if target.exists() and provenance.exists():
        record = json.loads(provenance.read_text(encoding='utf-8'))
        if record['sha256'] != sha256(target) or record['url'] != url or record['params'] != params:
            raise ValueError(f'Cached source changed: {name}')
        return target.read_text(encoding='utf-8')
    if offline:
        raise ValueError(f'Missing cached source: {name}')
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    target.write_text(response.text, encoding='utf-8')
    save_json(provenance, dict(url=url, params=params, retrieved_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                               sha256=sha256(target), bytes=target.stat().st_size))
    time.sleep(.12)
    return response.text


def download_one(code, args, start, end):
    folder = args.output/'sources'/code
    folder.mkdir(parents=True, exist_ok=True)
    symbol = ('sz' if code.startswith('1') else 'sh') + code
    retrieve(folder, 'sina_native', f'https://finance.sina.com.cn/realstock/company/{symbol}/hisdata_klc2/klc_kl.js', {}, args.offline)
    retrieve(folder, 'sina_factors', f'https://finance.sina.com.cn/realstock/company/{symbol}/qfq.js', {}, args.offline)
    for year in range(int(start[:4]), int(end[:4])+1):
        left, right = max(start, f'{year}-01-01'), min(end, f'{year}-12-31')
        for adjust in ('raw', 'qfq'):
            params = {'param': f'{symbol},day,{left},{right},640,{"" if adjust == "raw" else adjust}'}
            retrieve(folder, f'tencent_{adjust}_{year}', 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get', params, args.offline)
    return dict(code=code, status='downloaded')


def parse_sources(folder, code, start, end):
    from akshare.stock.cons import hk_js_decode
    import py_mini_racer

    text = (folder/'sina_native.txt').read_text(encoding='utf-8')
    encoded = re.search(r'=\s*"([^\"]+)"', text)
    if encoded is None:
        raise ValueError('Unrecognized Sina native response')
    with py_mini_racer.MiniRacer() as decoder:
        decoder.eval(hk_js_decode)
        raw = pd.DataFrame(decoder.call('d', encoded.group(1)))
    raw = raw.rename(columns={'date':'trade_date'})
    raw['trade_date'] = pd.to_datetime(raw.trade_date).dt.tz_localize(None).dt.normalize()
    factors = json.JSONDecoder().raw_decode((folder/'sina_factors.txt').read_text(encoding='utf-8').split('=',1)[1].lstrip())[0]
    if factors.get('total') != len(factors.get('data', [])):
        raise ValueError('Sina factor count does not match response')
    symbol = ('sz' if code.startswith('1') else 'sh') + code
    frames, returned_keys = {}, {}
    for adjust in ('raw', 'qfq'):
        chunks, keys = [], []
        for year in range(int(start[:4]), int(end[:4])+1):
            result = json.loads((folder/f'tencent_{adjust}_{year}.txt').read_text(encoding='utf-8'))
            if result.get('code') != 0:
                raise ValueError(f'Tencent failed: {adjust} {year}')
            data = result['data'][symbol]
            requested_key = 'day' if adjust == 'raw' else 'qfqday'
            # Tencent uses day for an unchanged series. Accept only after factor/price audit.
            key = requested_key if requested_key in data else 'day'
            rows = data.get(key)
            if not rows:
                raise ValueError(f'Missing Tencent {requested_key}: {year}')
            frame = pd.DataFrame([row[:6] for row in rows], columns=['trade_date','open','close','high','low','volume'])
            frame.trade_date = pd.to_datetime(frame.trade_date)
            frame = frame[(frame.trade_date >= max(start, f'{year}-01-01')) & (frame.trade_date <= min(end, f'{year}-12-31'))]
            chunks.append(frame)
            keys.append(dict(year=year, returned_key=key, rows=len(frame)))
        frames[adjust] = pd.concat(chunks, ignore_index=True)
        returned_keys[adjust] = keys
    return raw, frames['raw'], frames['qfq'], pd.DataFrame(factors['data']), returned_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calendar', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'snapshot_manifest.json').exists():
        raise SystemExit('Completed snapshot is immutable; choose a new output directory.')
    manifest = json.loads(args.freeze_manifest.read_text(encoding='utf-8'))
    start, end = '2024-01-01', manifest['latest_completed_date_cap']
    codes = manifest['universe']['codes']
    initial_db_hash = sha256(Path(manifest['existing_database']['path']))
    if initial_db_hash != manifest['existing_database']['sha256']:
        raise ValueError('Existing database differs from the frozen reference.')

    def wrapped(code):
        try:
            return download_one(code, args, start, end)
        except Exception as exc:
            return dict(code=code, status='download_failed', error_type=type(exc).__name__)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(wrapped, codes):
            results.append(result)
            if len(results) % 10 == 0 or result['status'] != 'downloaded':
                print(f'{len(results)}/{len(codes)} downloaded; failures={sum(x["status"] != "downloaded" for x in results)}', flush=True)
    save_json(args.output/'download_status.json', results)
    if args.download_only:
        return

    from qkquant.data.etf_snapshot import SnapshotValidationError, audit_etf

    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(args.calendar).trade_date))
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    if calendar.empty or str(calendar[-1].date()) != end or calendar.has_duplicates:
        raise ValueError('Calendar is empty, duplicated or does not reach the frozen completed date.')
    audits, accepted_frames = [], []
    for result in results:
        code = result['code']
        if result['status'] != 'downloaded':
            audits.append(result)
            continue
        try:
            raw, tx_raw, tx_qfq, factors, keys = parse_sources(args.output/'sources'/code, code, start, end)
            qfq_daily, raw_daily, audit = audit_etf(code, raw, tx_raw, tx_qfq, factors, calendar, start, end)
            audit['tencent_returned_keys'] = keys
            accepted_frames.extend([raw_daily, qfq_daily])
            audits.append(audit)
        except SnapshotValidationError as exc:
            audits.append(exc.audit)
        except Exception as exc:
            audits.append(dict(code=code, status='rejected', reason=str(exc), error_type=type(exc).__name__))
    save_json(args.output/'etf_audit.json', audits)
    print(json.dumps(dict(audited=len(audits), accepted=sum(audit['status']=='accepted' for audit in audits)), ensure_ascii=False), flush=True)
    if args.audit_only:
        return
    if not accepted_frames:
        raise ValueError('No ETF passed the price and corporate-action checks.')
    daily = pd.concat(accepted_frames, ignore_index=True)
    database = args.output/'verified_etf_snapshot.duckdb'
    if database.exists():
        raise ValueError('Database already exists; do not overwrite a snapshot.')
    with DuckStore(manifest['existing_database']['path'], read_only=True) as original:
        instruments = original.load_instruments(codes)
    with DuckStore(database) as target:
        target.upsert_instruments(instruments)
        target.upsert_daily(daily)
        target.upsert_calendar([day.date() for day in calendar])
    instruments.to_csv(args.output/'instrument_metadata.csv', index=False)
    pd.DataFrame({'trade_date': calendar}).to_csv(args.output/'trade_calendar.csv', index=False)
    coverage = daily[daily.adjust == 'qfq'].groupby('trade_date').code.nunique().reindex(calendar, fill_value=0)
    coverage.rename('codes').to_csv(args.output/'date_coverage.csv', index_label='trade_date')
    accepted = [audit['code'] for audit in audits if audit['status'] == 'accepted']
    snapshot = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        start=start, end=end, status='ready_for_frozen_research' if coverage.min() >= len(codes)*.9 else 'insufficient_coverage',
        database=str(database.resolve()), database_sha256=sha256(database),
        frozen_manifest=str(args.freeze_manifest.resolve()), frozen_manifest_sha256=sha256(args.freeze_manifest),
        calendar_source_file=str(args.calendar.resolve()), calendar_source_sha256=sha256(args.calendar),
        codes_requested=codes, codes_accepted=accepted, accepted_count=len(accepted), rejected_count=len(codes)-len(accepted),
        minimum_daily_coverage=int(coverage.min()), daily_rows=int((daily.adjust=='qfq').sum()),
        price_adjustment_verified=True, verification_scope='Every accepted bar: Tencent native price vs Sina native; Tencent qfq vs Sina raw/s-u, half a 0.001 tick tolerance.',
        native_volume_amount_source='Sina; unadjusted shares and CNY; Tencent sixth field is NOT cash amount.',
        availability_protocol='v2_common_source_gaps_explicitly_unavailable',
        shared_missing_code_dates=sum(len(audit.get('calendar_missing_dates', [])) for audit in audits if audit['status']=='accepted'),
        duplicate_or_single_source_missing_bars_allowed=False, synthetic_gap_prices_created=False,
        sources_independent_ultimate_feeds_verified=False,
        point_in_time_universe_verified=False, dividend_cashflow_accounting_verified=False,
        limitations=['Frozen existing 101-code Shenzhen sample only.', 'Cross-vendor agreement is not an exchange guarantee.',
                     'Official notices are representative spot checks, not a complete issuer-by-issuer audit.',
                     'No portfolio returns or drawdown claims; no point-in-time membership reconstruction.'],
        original_database_unchanged=sha256(Path(manifest['existing_database']['path'])) == initial_db_hash,
        builder_sha256=sha256(Path(__file__)),
        audit_module_sha256=sha256(Path(__import__('qkquant.data.etf_snapshot', fromlist=['audit_etf']).__file__)))
    save_json(args.output/'snapshot_manifest.json', snapshot)
    print(json.dumps({k:snapshot[k] for k in ('status','accepted_count','rejected_count','minimum_daily_coverage','daily_rows','database')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
