"""Read the audited ETF snapshot; never silently fall back to the legacy database."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from qkquant.config import PROJECT_ROOT
from qkquant.data.storage import DuckStore
from qkquant.factors.pipeline import load_panel

DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "verified_etf_20260918"
# SHA256 of the sorted, newline-separated 101 codes fixed before this validation.
FROZEN_CODES_SHA256 = "74e2f9f41840d1eb447ab852a076f9069682b60a01cac45712a130e76b470fd3"
MIN_COVERAGE = 0.95
EXCLUDED_CODES = {"159925"}


class VerifiedInputError(ValueError):
    """The supplied snapshot cannot be used for the fixed ETF research workflow."""


@dataclass
class VerifiedEtfInput:
    store: DuckStore
    panel: dict[str, pd.DataFrame]
    raw_panel: dict[str, pd.DataFrame]
    codes: list[str]
    coverage: pd.DataFrame
    manifest: dict
    freeze_manifest: dict
    metadata: dict


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified_hash(path: Path, expected: str | None) -> str:
    if not path.is_file():
        raise VerifiedInputError(f"Required snapshot file is missing: {path}")
    actual = _sha256(path)
    if not expected or actual != expected:
        raise VerifiedInputError(f"Snapshot SHA256 mismatch: {path.name}")
    return actual


@contextmanager
def open_verified_etf_input(snapshot_dir: str | Path | None = None) -> Iterator[VerifiedEtfInput]:
    """Validate local companion files and yield readonly data on the full calendar.

    Missing code/date bars remain NaN. Every session must cover at least 95% of
    the original 101-code pool, including the quarantined code in the denominator.
    Manifest provenance paths are descriptive only: data is read beside the local
    manifest, making copied snapshots portable and preventing legacy DB fallback.
    Source-code freeze validation belongs to the research runner, not this gate.
    """
    root = Path(snapshot_dir) if snapshot_dir is not None else DEFAULT_SNAPSHOT_DIR
    root = root.resolve()
    store = None
    try:
        manifest_path = root / "snapshot_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "ready_for_frozen_research"
                or manifest.get("price_adjustment_verified") is not True
                or manifest.get("synthetic_gap_prices_created") is not False
                or manifest.get("duplicate_or_single_source_missing_bars_allowed") is not False
                or manifest.get("availability_protocol") != "v2_common_source_gaps_explicitly_unavailable"):
            raise VerifiedInputError("Snapshot has not passed the required price/availability audit.")
        database = root / "verified_etf_snapshot.duckdb"
        database_hash = _verified_hash(database, manifest.get("database_sha256"))
        freeze_path = root / "freeze_manifest.json"
        freeze_hash = _verified_hash(freeze_path, manifest.get("frozen_manifest_sha256"))
        calendar_path = root / "trade_calendar.csv"
        calendar_hash = _verified_hash(calendar_path, manifest.get("calendar_source_sha256"))
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        codes = freeze.get("universe", {}).get("codes", [])
        if (len(codes) != 101 or len(set(codes)) != 101
                or freeze["universe"].get("count") != 101
                or hashlib.sha256("\n".join(sorted(codes)).encode()).hexdigest() != FROZEN_CODES_SHA256
                or manifest.get("codes_requested") != codes):
            raise VerifiedInputError("Snapshot does not match the fixed 101-code ETF universe.")
        accepted = manifest.get("codes_accepted", [])
        if (len(accepted) != 100 or set(accepted) != set(codes) - EXCLUDED_CODES
                or manifest.get("accepted_count") != 100 or manifest.get("rejected_count") != 1):
            raise VerifiedInputError("Accepted ETF universe must exclude quarantined 159925.")
        start, end = pd.Timestamp(manifest["start"]), pd.Timestamp(manifest["end"])
        if start > end or end > pd.Timestamp(freeze["latest_completed_date_cap"]):
            raise VerifiedInputError("Snapshot date range exceeds the frozen completed-date cap.")
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar_path)["trade_date"]))
        if (calendar.empty or calendar.hasnans or calendar.has_duplicates
                or not calendar.is_monotonic_increasing or (calendar != calendar.normalize()).any()
                or calendar[0] < start or calendar[-1] != end):
            raise VerifiedInputError("Snapshot trading calendar is invalid or does not reach its end date.")
        store = DuckStore(database, read_only=True)
        if not calendar.equals(pd.DatetimeIndex(pd.to_datetime(store.load_calendar()))):
            raise VerifiedInputError("Database calendar differs from the hashed full trading calendar.")
        counts = dict(store.conn.execute("SELECT adjust, count(*) FROM daily_bars GROUP BY adjust").fetchall())
        rows = manifest.get("daily_rows")
        if not rows or counts != {"": rows, "qfq": rows}:
            raise VerifiedInputError("Database must contain matching native ('') and qfq row counts only.")
        database_codes = {row[0] for row in store.conn.execute("SELECT DISTINCT code FROM daily_bars").fetchall()}
        if database_codes != set(accepted):
            raise VerifiedInputError("Database bars contain unexpected or quarantined ETF codes.")
        panels = [load_panel(store, codes, adjust=adjust) for adjust in ("qfq", "")]
        for panel in panels:
            if not panel["close"].index.isin(calendar).all():
                raise VerifiedInputError("Database bars fall outside the full trading calendar.")
            for field, frame in panel.items():
                panel[field] = frame.reindex(index=calendar, columns=codes)
            present = panel["close"].notna()
            for field in ("open", "high", "low", "close", "volume", "amount"):
                frame = panel[field]
                if (not frame.notna().equals(present)
                        or not np.isfinite(frame.to_numpy()[present.to_numpy()]).all()
                        or ((frame <= 0) & present).any().any() and field not in ("volume", "amount")
                        or (frame < 0).any().any()):
                    raise VerifiedInputError(f"Invalid or inconsistent {field} bars in snapshot.")
            if ((panel["high"] < panel["open"]) | (panel["high"] < panel["close"])
                    | (panel["low"] > panel["open"]) | (panel["low"] > panel["close"])).any().any():
                raise VerifiedInputError("Invalid OHLC ordering in snapshot.")
        panel, raw_panel = panels
        if (not panel["close"].notna().equals(raw_panel["close"].notna())
                or any(not panel[field].equals(raw_panel[field]) for field in ("volume", "amount"))):
            raise VerifiedInputError("Native and qfq bars must share dates and native volume/amount.")
        counts = panel["close"].notna().sum(axis=1)
        coverage = pd.DataFrame({"codes": counts, "frozen_codes": len(codes),
                                 "coverage_fraction": counts / len(codes),
                                 "eligible": counts >= len(codes) * MIN_COVERAGE})
        coverage.index.name = "trade_date"
        if not coverage["eligible"].all():
            failed = coverage.index[~coverage["eligible"]][0].date()
            raise VerifiedInputError(f"ETF coverage is below 95% of the fixed universe on {failed}.")
        if int(counts.min()) != manifest.get("minimum_daily_coverage"):
            raise VerifiedInputError("Actual daily coverage differs from the snapshot manifest.")
        metadata = dict(snapshot_dir=str(root), database=str(database), database_sha256=database_hash,
                        snapshot_manifest_sha256=_sha256(manifest_path), freeze_manifest_sha256=freeze_hash,
                        calendar_sha256=calendar_hash, data_start=str(calendar[0].date()),
                        data_end=str(calendar[-1].date()), calendar_days=len(calendar),
                        universe_count=len(codes), accepted_codes=accepted,
                        excluded_codes=sorted(EXCLUDED_CODES), minimum_coverage=float(coverage.coverage_fraction.min()),
                        minimum_required_coverage=MIN_COVERAGE, price_adjustment_verified=True,
                        point_in_time_universe_verified=manifest.get("point_in_time_universe_verified", False),
                        dividend_cashflow_accounting_verified=manifest.get("dividend_cashflow_accounting_verified", False),
                        limitations=manifest.get("limitations", []))
    except VerifiedInputError:
        if store is not None:
            store.close()
        raise
    except Exception as exc:
        if store is not None:
            store.close()
        raise VerifiedInputError(f"Cannot verify ETF snapshot at {root}: {exc}") from exc
    try:
        yield VerifiedEtfInput(store, panel, raw_panel, codes, coverage, manifest, freeze, metadata)
    finally:
        store.close()
