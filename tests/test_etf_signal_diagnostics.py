import numpy as np
import pandas as pd
import pytest
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from qkquant.etf_signal import EtfSignalConfig
from qkquant.etf_signal_diagnostics import collect_diagnostics, entry_events, event_outcome, event_return, summarize


def test_nested_pullback_and_freshness():
    anchors={0:10,10:10.4,16:10.9,20:10.2,24:10.8,28:10.4,32:11.3,36:10.9,40:12.7,44:11.5,48:12.8,52:13.4}
    close=pd.Series(np.interp(np.arange(53),list(anchors),list(anchors.values())))
    cfg=EtfSignalConfig()
    before=entry_events(close.iloc[:46],close.iloc[:46]+.02,close.iloc[:46]-.02,cfg)
    fresh=entry_events(close.iloc[:47],close.iloc[:47]+.02,close.iloc[:47]-.02,cfg)
    old=entry_events(close.iloc[:48],close.iloc[:48]+.02,close.iloc[:48]-.02,cfg)
    assert not before['chan'] and not before['pullback']
    assert fresh['chan'] and fresh['pullback']
    assert not old['chan'] and not old['pullback']


def test_next_open_horizon_and_both_costs():
    opens=pd.DataFrame({'ETF':[99.,10.,*([10.]*9),11.,22.]})
    panel={'open':opens,'amount':opens*1e8,'low':opens.copy()}
    cfg=EtfSignalConfig(commission_min=5,commission_rate=0,slippage_pct=.002)
    result=event_return(panel,'ETF',0,10,cfg,budget=10000)
    assert result['gross']==pytest.approx(.1)
    assert result['qty']==900
    assert result['net']==pytest.approx(((11*.998-10*1.002)*900-10)/10000)
    assert event_return(panel,'ETF',0,20,cfg) is None


def test_missing_exit_or_zero_amount_is_not_filled():
    opens=pd.DataFrame({'ETF':[10.]*25})
    panel={'open':opens,'amount':opens*1e8,'low':opens.copy()}
    panel['open'].iloc[11]=np.nan
    assert event_return(panel,'ETF',0,10,EtfSignalConfig()) is None
    assert event_outcome(panel,'ETF',0,10,EtfSignalConfig())['status']=='missing'
    panel['open'].iloc[11]=10
    panel['amount'].iloc[1]=0
    assert event_return(panel,'ETF',0,10,EtfSignalConfig()) is None
    assert event_outcome(panel,'ETF',0,10,EtfSignalConfig())['status']=='unexecutable'


def diagnostic_panel(n):
    positions=np.arange(n)
    close=pd.DataFrame({'ETF':10+.01*positions+.2*np.sin(positions/3)},
                       index=pd.bdate_range('2027-01-01',periods=n))
    return {'close':close,'open':close.copy(),'high':close+.02,'low':close-.02,
            'amount':pd.DataFrame(1e8,index=close.index,columns=close.columns)}


def test_recent_ten_day_outcomes_are_kept_while_twenty_day_pending():
    panel=diagnostic_panel(145)
    events=collect_diagnostics(panel)
    recent=events[events.position==133].iloc[0]
    latest=events[events.position==144].iloc[0]
    for cost in ('base','stress'):
        assert recent[f'{cost}_status_10']=='valid'
        assert pd.notna(recent[f'{cost}_net_10'])
        assert recent[f'{cost}_status_20']=='pending'
        assert pd.isna(recent[f'{cost}_net_20'])
        assert latest[f'{cost}_status_10']=='pending'
        assert latest[f'{cost}_status_20']=='pending'


def test_future_data_does_not_rewrite_candidate_signal_identity():
    panel=diagnostic_panel(165)
    earlier={field:frame.iloc[:145] for field,frame in panel.items()}
    before=collect_diagnostics(earlier)
    after=collect_diagnostics(panel)
    identity=['date','position','code','chan','pullback','trend']
    assert before[['chan','pullback','trend']].any(axis=1).any()
    pd.testing.assert_frame_equal(before[identity],
        after[after.position<145][identity].reset_index(drop=True))
    assert before.iloc[-1].base_status_10=='pending'
    assert after[after.position==144].iloc[0].base_status_10=='valid'


def test_evaluation_split_keeps_prior_spacing_and_reports_pending():
    rows=[]
    for pos in [0,10,21,22]:
        row=dict(date=f'2027-01-{pos+1:02}',position=pos,code='A',chan=True,pullback=True,trend=False)
        for cost in ('base','stress'):
            for horizon in (10,20):
                status='pending' if pos==22 else 'valid'
                value=np.nan if status=='pending' else .01
                row.update({f'{cost}_status_{horizon}':status,f'{cost}_net_{horizon}':value,
                            f'{cost}_mae_{horizon}':value,f'{cost}_edge_{horizon}':value})
        rows.append(row)
    summary,selected=summarize(pd.DataFrame(rows),evaluation_start='2027-01-10')
    assert selected[selected.group=='chan'].position.tolist()==[21]
    assert set(summary.year)=={'all','2027'}
    raw=summary[(summary.group=='chan')&(summary['sample']=='all')&(summary.year=='all')
                &(summary.horizon==10)&(summary.cost=='base')].iloc[0]
    assert raw.events==3 and raw.n==2 and raw.pending==1
    assert raw.missing==0 and raw.unexecutable==0


def test_budget_failure_is_retained_as_unexecutable():
    panel=diagnostic_panel(145)
    result=event_outcome(panel,'ETF',120,10,EtfSignalConfig(),budget=1)
    assert result['status']=='unexecutable'
    assert result['reason']=='budget_below_one_lot'


def test_known_entry_failure_is_not_hidden_by_pending_exit():
    panel=diagnostic_panel(125)
    panel['amount'].iloc[121]=0
    assert event_outcome(panel,'ETF',120,20,EtfSignalConfig())['status']=='unexecutable'
    panel['amount'].iloc[121]=1e8
    panel['open'].iloc[121]=np.nan
    assert event_outcome(panel,'ETF',120,20,EtfSignalConfig())['status']=='missing'
    panel['open'].iloc[121]=10
    assert event_outcome(panel,'ETF',120,20,EtfSignalConfig(),budget=1)['status']=='unexecutable'
    assert event_outcome(panel,'ETF',124,20,EtfSignalConfig())['reason']=='entry_not_observed'


def test_low_coverage_dates_are_skipped_without_shortening_horizons():
    panel=diagnostic_panel(145)
    eligible=pd.Series(True,index=panel['close'].index)
    eligible.iloc[122]=False
    events=collect_diagnostics(panel,eligible_dates=eligible)
    assert 122 not in events.position.tolist()
    prior=events[events.position==121].iloc[0]
    assert prior.base_gross_10==pytest.approx(panel['open'].iloc[132,0]/panel['open'].iloc[122,0]-1)


def research_script():
    script=Path(__file__).resolve().parents[1]/'scripts'/'research_signal_value.py'
    spec=importlib.util.spec_from_file_location('research_signal_value',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_rejects_missing_hash_changed_source_config_and_protocol(tmp_path,monkeypatch):
    script=research_script()
    monkeypatch.setattr(script,'PROJECT_ROOT',tmp_path)
    hashes={}
    for relative in script.REQUIRED_FROZEN_SOURCES:
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(relative,encoding='utf-8')
        hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    cfg=EtfSignalConfig()
    manifest=dict(source_sha256=hashes,config=json.loads(json.dumps(asdict(cfg))),
                  event_protocol=script.EVENT_PROTOCOL.copy(),universe=dict(codes=['A'],count=1),
                  latest_completed_date_cap='2026-09-17')
    script.validate_manifest(manifest,cfg)
    altered=json.loads(json.dumps(manifest))
    altered['config']['score_threshold']=.7
    with pytest.raises(ValueError,match='config'):
        script.validate_manifest(altered,cfg)
    altered=json.loads(json.dumps(manifest))
    altered['event_protocol']['main_horizon']=11
    with pytest.raises(ValueError,match='protocol'):
        script.validate_manifest(altered,cfg)
    altered=json.loads(json.dumps(manifest))
    altered['source_sha256'].pop(script.REQUIRED_FROZEN_SOURCES[0])
    with pytest.raises(ValueError,match='missing required'):
        script.validate_manifest(altered,cfg)
    (tmp_path/script.REQUIRED_FROZEN_SOURCES[0]).write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='source changed'):
        script.validate_manifest(manifest,cfg)


def test_frozen_panel_uses_fixed_codes_cap_and_keeps_missing_sessions(monkeypatch):
    script=research_script()
    dates=pd.bdate_range('2026-01-01',periods=25)
    frame=pd.DataFrame(10.,index=dates,columns=['A','B','EXTRA'])
    frame=frame.drop(dates[5])
    frame.loc[dates[10],'B']=np.nan
    frame.loc[dates[23],'B']=np.nan
    cap=str(dates[23].date())
    def load(store,codes,start,end,adjust):
        assert codes==['A','B'] and end==cap and adjust=='qfq'
        return {field:frame.loc[:end].copy() for field in ('open','high','low','close','amount')}
    class Store:
        def load_calendar(self,start,end):
            return dates[dates<=pd.Timestamp(end)].tolist()
    monkeypatch.setattr(script,'load_panel',load)
    panel,coverage,end=script.frozen_panel(Store(),dict(universe=dict(codes=['A','B']),latest_completed_date_cap=cap))
    assert panel['close'].columns.tolist()==['A','B']
    assert dates[5] in panel['close'].index and panel['close'].loc[dates[5]].isna().all()
    assert not coverage.loc[dates[5],'eligible'] and not coverage.loc[dates[10],'eligible']
    assert coverage.loc[dates[10],'coverage_fraction']==.5
    assert end==str(dates[22].date()) and panel['close'].index[-1]==dates[22]
    assert coverage.index[-1]==dates[23]


def test_frozen_mode_refuses_to_overwrite_report_snapshot(tmp_path,monkeypatch):
    script=research_script()
    report=tmp_path/'metadata.json'
    report.write_text('original snapshot',encoding='utf-8')
    monkeypatch.setattr('sys.argv',['research_signal_value.py','--output',str(tmp_path),
                                   '--freeze-manifest',str(tmp_path/'manifest.json')])
    with pytest.raises(ValueError,match='cannot be overwritten'):
        script.main()
    assert report.read_text(encoding='utf-8')=='original snapshot'
    assert script.file_sha256(report)==hashlib.sha256(report.read_bytes()).hexdigest()


def test_adverse_move_includes_exit_open_gap():
    opens=pd.DataFrame({'ETF':[10.]*25})
    panel={'open':opens,'amount':opens*1e8,'low':opens.copy()}
    panel['open'].iloc[11]=7
    result=event_return(panel,'ETF',0,10,EtfSignalConfig())
    assert result['mae']==pytest.approx(-.3)


def test_spaced_selection_is_per_group_and_first_in_time():
    rows=[]
    for pos in [0,10,21]:
        row=dict(date=f'2025-01-{pos+1:02}',position=pos,code='A',chan=True,pullback=True,trend=False)
        for cost in ('base','stress'):
            for horizon in (10,20):
                row.update({f'{cost}_net_{horizon}':.01,f'{cost}_mae_{horizon}':-.02,f'{cost}_edge_{horizon}':.005})
        rows.append(row)
    summary,selected=summarize(pd.DataFrame(rows))
    assert selected[selected.group=='chan'].position.tolist()==[0,21]
    assert selected[selected.group=='pullback'].position.tolist()==[0,21]
    line=summary[(summary.group=='chan')&(summary['sample']=='spaced')&(summary.year=='all')&(summary.horizon==10)&(summary.cost=='base')].iloc[0]
    assert line.n==2 and line.dates==2 and line.etfs==1
