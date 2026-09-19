"""Select document-backed financial versions without changing strategy rules.

Publication chronology and explicitly reviewed EPS bases are handled here.
Coverage, accounting scope, units and action completeness require evidence checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qkquant.oneil import OneilConfig, financial_state


def select_financial_versions(rows: pd.DataFrame, day) -> pd.DataFrame:
    """Keep the latest documented version available before the decision date.

    Publications are conservatively usable from the next calendar date, as in
    the existing daily research model. Provider update times are not used.
    Missing values in a newer version must not fall back to an older version.
    """
    key = ["code", "kind", "report_date"]
    required = [*key, "published_at", "source_url", "version_id"]
    if missing := set(required) - set(rows.columns):
        raise ValueError(f"missing provenance columns: {sorted(missing)}")
    result = rows.copy()
    for column in ("code", "kind", "source_url", "version_id"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing provenance: {column}")
    for column in ("report_date", "published_at"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
        if result[column].isna().any():
            raise ValueError(f"missing date: {column}")
    if (result.published_at < result.report_date).any():
        raise ValueError("financial publication predates report period end")
    cutoff = pd.Timestamp(day).normalize()
    known = result[(result.published_at < cutoff) & (result.report_date < cutoff)]
    if known.duplicated([*key, "published_at"]).any():
        raise ValueError("ambiguous same-day financial versions")
    if known.duplicated([*key, "version_id"]).any():
        raise ValueError("duplicate financial version")
    return (known.sort_values([*key, "published_at"])
            .drop_duplicates(key, keep="last").reset_index(drop=True))


def prepare_financials_asof(rows: pd.DataFrame, actions: pd.DataFrame, day) -> pd.DataFrame:
    """Align explicitly reviewed EPS bases using known, effective bonus actions.

    Each source row declares which action IDs its EPS already incorporates.
    Only proportional bonus shares/splits without a change in owners' equity
    belong here; ordinary share issuance and option dilution must not be undone.
    Input coverage is supplied by the evidence audit, not inferred here.
    """
    selected = select_financial_versions(rows, day)
    if selected.empty:
        return selected
    cutoff = pd.Timestamp(day).normalize()
    required = {"code", "action_id", "action_type", "factor", "published_at", "effective_date", "source_url"}
    if missing := required - set(actions.columns):
        raise ValueError(f"missing EPS action columns: {sorted(missing)}")
    active = actions.copy()
    for col in ("code", "action_id"):
        if active[col].isna().any() or active[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing EPS action provenance: {col}")
    for col in ("published_at", "effective_date"):
        active[col] = pd.to_datetime(active[col]).dt.normalize()
        if active[col].isna().any():
            raise ValueError(f"missing EPS action date: {col}")
    active = active[(active.published_at < cutoff) & (active.effective_date <= cutoff)]
    active["factor"] = pd.to_numeric(active.factor, errors="raise").astype(float)
    if active.action_id.duplicated().any():
        raise ValueError("duplicate EPS action")
    if not active.action_type.isin(["bonus_shares", "split"]).all():
        raise ValueError("only equity-neutral bonus shares/splits can adjust historical EPS")
    if not np.isfinite(active.factor).all() or (active.factor <= 0).any():
        raise ValueError("invalid EPS action factor")
    if active.source_url.isna().any() or active.source_url.astype(str).str.strip().eq("").any():
        raise ValueError("EPS action requires a source")
    selected["eps_reported"] = selected.eps
    selected["eps_adjustment_factor"] = 1.0
    selected["available_at"] = selected.published_at
    selected["eps_basis_actions"] = pd.Series([[] for _ in range(len(selected))], dtype=object)
    for i, row in selected.iterrows():
        incorporated = row.get("eps_applied_actions")
        if not isinstance(incorporated, list) or len(incorporated) != len(set(incorporated)):
            raise ValueError("EPS source must declare its incorporated actions")
        company_actions = active[active.code == row.code]
        if set(incorporated) - set(company_actions.action_id):
            raise ValueError("EPS source incorporates an unavailable action")
        included = company_actions[company_actions.action_id.isin(incorporated)]
        if (included.published_at > row.published_at).any() or (included.effective_date > row.published_at).any():
            raise ValueError("EPS source predates its incorporated action")
        pending = company_actions[~company_actions.action_id.isin(incorporated)]
        factor = pending.factor.prod()
        for field in ("eps", "eps_lower", "eps_upper"):
            if field in selected:
                selected.loc[i, field] = row[field] / factor
        selected.loc[i, "eps_adjustment_factor"] = factor
        selected.at[i, "eps_basis_actions"] = company_actions.action_id.tolist()
        if not pending.empty:
            selected.loc[i, "available_at"] = max(row.published_at, pending.published_at.max(), pending.effective_date.max())
    return selected


def financial_state_asof(rows: pd.DataFrame, actions: pd.DataFrame, day, cfg: OneilConfig) -> dict:
    """Apply the unchanged growth rules to reviewed, comparable historical EPS.

    When EPS intervals are supplied, their pessimistic endpoints must also
    satisfy the EPS growth rules. A crossing interval is unavailable evidence,
    not a passing point estimate. This does not certify full report coverage.
    """
    selected = prepare_financials_asof(rows, actions, day)
    if selected.empty:
        return {"fundamental_ok": False, "reason": "missing_financials"}
    if selected.code.nunique() != 1:
        raise ValueError("financial state requires exactly one company")
    inputs = selected.assign(notice_date=selected.published_at, update_date=selected.published_at)
    state = financial_state(inputs, day, cfg)
    if "quarter_eps_growth" not in state:
        return state
    if {"eps_lower", "eps_upper"} - set(selected.columns):
        raise ValueError("reviewed EPS requires lower and upper bounds")
    quarter = selected[selected.kind == "quarter"].sort_values("report_date")
    current = quarter.iloc[-1]
    previous = quarter[quarter.report_date == current.report_date - pd.DateOffset(years=1)].iloc[-1]
    annual = selected[selected.kind == "annual"].sort_values("report_date").tail(4)
    required = pd.concat([quarter.tail(1), previous.to_frame().T, annual])
    bounds = required[["eps_lower", "eps_upper"]].astype(float)
    if (not np.isfinite(bounds.to_numpy()).all() or (bounds.eps_lower <= 0).any()
            or (bounds.eps_upper < bounds.eps_lower).any()
            or (required.eps.astype(float) < bounds.eps_lower).any()
            or (required.eps.astype(float) > bounds.eps_upper).any()):
        return {**state, "fundamental_ok": False, "reason": "invalid_eps_bounds"}
    q_low = current.eps_lower / previous.eps_upper - 1
    q_high = current.eps_upper / previous.eps_lower - 1
    a_low = (annual.eps_lower.iloc[-1] / annual.eps_upper.iloc[0]) ** (1 / 3) - 1
    increasing = np.all(annual.eps_lower.to_numpy()[1:] > annual.eps_upper.to_numpy()[:-1])
    state.update(quarter_eps_growth_lower=float(q_low), quarter_eps_growth_upper=float(q_high),
                 annual_eps_cagr_lower=float(a_low), latest_evidence_available=selected.available_at.max())
    eps_ok = q_low >= cfg.quarterly_growth and a_low >= cfg.annual_cagr and increasing
    if state["fundamental_ok"] and not eps_ok:
        state.update(fundamental_ok=False, reason="eps_threshold_not_proven_by_bounds")
    return state


def documented_financial_failure(evidence: dict, code: str, day, cfg: OneilConfig) -> bool:
    """Verify a reviewed, date-scoped necessary-condition failure.

    This can only reject a candidate, never certify a passing financial state.
    The caller verifies file hashes; human source review establishes that the
    report is the latest applicable version for the listed signal dates.
    """
    day = pd.Timestamp(day).normalize()
    if (evidence.get("decision") != "verified_fail" or evidence.get("code") != code
            or str(day.date()) not in evidence.get("signal_dates", [])):
        return False
    for field in ("source_url", "source_file", "source_sha256", "pdf_page"):
        if not evidence.get(field):
            return False
    report = pd.Timestamp(evidence.get("report_date"))
    published = pd.Timestamp(evidence.get("published_at"))
    if pd.isna(report) or pd.isna(published) or not report <= published < day:
        return False
    check = evidence.get("check")
    if check == "quarter_eps_growth":
        inputs = evidence.get("inputs") or {}
        latest_report = pd.Timestamp(evidence.get("latest_quarter_report_date", report))
        latest_published = pd.Timestamp(evidence.get("latest_quarter_published_at", published))
        previous_date = pd.Timestamp(inputs.get("previous_report_date"))
        previous, current, lower, upper = pd.to_numeric(
            [inputs.get(k) for k in ("previous_eps", "current_eps", "previous_eps_lower", "current_eps_upper")],
            errors="coerce")
        return bool(evidence.get("latest_quarter_verified") is True and (day - report).days <= 200
                    and latest_report == report and latest_published == published
                    and report.is_quarter_end and previous_date == report - pd.DateOffset(years=1)
                    and inputs.get("same_share_basis") is True and inputs.get("same_accounting_basis") is True
                    and np.isfinite([previous, current, lower, upper]).all()
                    and 0 < lower <= previous and current <= upper and upper / lower - 1 < cfg.quarterly_growth)
    if evidence.get("latest_annual_verified") is not True:
        return False
    latest_report = pd.Timestamp(evidence.get("latest_annual_report_date", report))
    latest_published = pd.Timestamp(evidence.get("latest_annual_published_at", published))
    if (pd.isna(report) or pd.isna(published) or not report <= published < day
            or pd.isna(latest_report) or pd.isna(latest_published) or not latest_report <= latest_published < day
            or report > latest_report or latest_report.month != 12 or latest_report.day != 31
            or report.month != 12 or report.day != 31 or (day - latest_report).days > 550):
        return False
    if latest_report != report and not all(evidence.get(k) for k in (
            "latest_annual_source_file", "latest_annual_source_url", "latest_annual_source_sha256")):
        return False
    if report == latest_report and published != latest_published:
        return False
    if check == "annual_eps_cagr":
        inputs = evidence.get("inputs") or {}
        previous_date, current_date = (pd.Timestamp(inputs.get(k)) for k in ("previous_report_date", "current_report_date"))
        previous_published = pd.Timestamp(inputs.get("previous_eps_published_at"))
        current_published = pd.Timestamp(inputs.get("current_eps_published_at"))
        previous, current, lower, upper = pd.to_numeric(
            [inputs.get(k) for k in ("previous_eps", "current_eps", "previous_eps_lower", "current_eps_upper")],
            errors="coerce")
        return bool(report == latest_report == current_date
                    and previous_date == current_date - pd.DateOffset(years=3)
                    and previous_date <= previous_published < day and current_published == published
                    and inputs.get("same_share_basis") is True and inputs.get("same_accounting_basis") is True
                    and np.isfinite([previous, current, lower, upper]).all()
                    and 0 < lower <= previous and 0 < current <= upper
                    and (upper / lower) ** (1 / 3) - 1 < cfg.annual_cagr)
    if check == "annual_roe":
        value = pd.to_numeric([evidence.get("value")], errors="coerce")[0]
        return bool(evidence.get("units") == "percentage_points" and report == latest_report
                    and np.isfinite(value) and value < cfg.min_roe)
    if check == "annual_eps_positive":
        eps_date = pd.Timestamp((evidence.get("inputs") or {}).get("eps_report_date"))
        value = pd.to_numeric([evidence.get("value")], errors="coerce")[0]
        return bool(np.isfinite(value) and value <= 0 and eps_date.month == 12 and eps_date.day == 31
                    and eps_date <= report
                    and latest_report.year - 3 <= eps_date.year <= latest_report.year)
    if check == "annual_eps_increasing":
        inputs = evidence.get("inputs") or {}
        previous_date, current_date = (pd.Timestamp(inputs.get(k)) for k in ("previous_report_date", "current_report_date"))
        previous, current = pd.to_numeric([inputs.get(k) for k in ("previous_eps", "current_eps")], errors="coerce")
        return bool(inputs.get("same_share_basis") is True and inputs.get("same_accounting_basis") is True
                    and np.isfinite([previous, current]).all()
                    and current <= previous and previous_date.month == current_date.month == 12
                    and current_date <= report
                    and previous_date.day == current_date.day == 31 and current_date.year == previous_date.year + 1
                    and latest_report.year - 3 <= previous_date.year < current_date.year <= latest_report.year)
    return False
