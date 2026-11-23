"""E211: full-history 20/60-day per-asset heads and causal meta ensembles."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments import e206_momentum3_tsfm as E
from src.analysis import tsfm_momentum_study as N
from src.prediction import meta_resid as MR
from src.cluster import hosts, manager, budget

ROOT = REPO/'results/reports/crypto/e211_cross_target_tsfm'
SCR = Path('/scratch/danielco/e211_cross_target_tsfm')
SOURCE = REPO/'results/reports/crypto/e206_momentum3_tsfm'
SOURCE_SCR = Path('/scratch/danielco/e206_momentum3_tsfm')
MODELS = ['chronos2', 'momentum', 't0', 'timestral']
NORMS = {'chronos2': 'rolling', 't0': 'rolling', 'timestral': 'robust'}
WALLS = {20: 35, 60: 91}
PYTHON = '/home/danielco/miniconda3/bin/python3'
save, parquet = E.save, E.parquet


def fold_panel(panel, start, horizon):
    """Forecast block with both tuning and final fitting behind label walls."""
    dates = np.sort(panel.date.unique())
    origin = pd.Timestamp(dates[start])
    last = origin-pd.Timedelta(days=WALLS[horizon])
    past = dates[dates <= last.to_datetime64()]
    if len(past) < 63:
        return None
    validation = pd.Timestamp(past[-63])
    training_end = validation-pd.Timedelta(days=WALLS[horizon])
    lower = pd.Timestamp(dates[max(0, start-504)])
    out = panel.copy()
    out['train'] = out.date.between(lower, training_end)
    out['validation'] = out.date.between(validation, last)
    out['final_fit'] = out.date.between(lower, last)
    out['test'] = out.date.isin(dates[start:start+130])
    protocol = dict(origin=origin, lower=lower, training_last=training_end,
                    validation_first=validation, last_resolved=last,
                    target=horizon, wall_calendar_days=WALLS[horizon])
    return out, protocol


def definitions():
    return [members for size in range(1, 6) for members in itertools.combinations(range(16), size)]


def prepare():
    if (ROOT/'PREPARED.json').exists():
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    SCR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(SOURCE/'panel.parquet')
    labels = pd.read_parquet(REPO/'data/crypto/v2.0/train.parquet',
                             columns=['symbol', 'date', 'target_binned_return_60'],
                             filters=[('date', '<', pd.Timestamp('2026-02-20'))]).reset_index()
    assert not labels.duplicated(['symbol', 'date']).any()
    joined = panel.merge(labels[['symbol', 'date', 'target_binned_return_60']],
                         on=['symbol', 'date'], how='left', validate='1:1', sort=False)
    pd.testing.assert_frame_equal(panel, joined[panel.columns])
    assert joined.date.max() < pd.Timestamp('2026-02-20')
    assert joined.target_binned_return_60.notna().all()
    parquet(ROOT/'panel.parquet', joined)
    metadata = json.loads((SOURCE/'PREPARED.json').read_text())
    metadata.update(experiment='E211', panel_hash=E.sha(ROOT/'panel.parquet'),
                    target60_label_hash=hashlib.sha256(pd.util.hash_pandas_object(labels, index=True).values.tobytes()).hexdigest(),
                    temporal_protocol='Full-history walk-forward; trailing504/validation63/forecast130 panel dates;35/91 calendar-day label purges; no final holdout split.',
                    label_source='v2.0 filtered date<2026-02-20 before reading labels; joined by symbol/date; no target60 imputation',
                    models=MODELS, label_walls=WALLS,
                    residualization='Existing prediction residualization: strength0.5, latest meta snapshot<=D-5 calendar days',
                    selection='All2..5 equal-rank combinations of16 raw/residual streams; top5 per evaluation target by descriptive same-history payout',
                    host=socket.gethostname())
    save(ROOT/'PREPARED.json', metadata)
    save(ROOT/'status.json', dict(stage='extracting_target60', time=time.time()))


def collect():
    old = E.ROOT, E.SCR, N.MODELS
    try:
        E.ROOT, E.SCR, N.MODELS = ROOT, SCR, ['chronos2', 't0', 'timestral']
        E.collect()
    finally:
        E.ROOT, E.SCR, N.MODELS = old
    common = np.load(SCR/'common_rows.npy') & np.load(SOURCE_SCR/'common_rows.npy')
    np.save(SCR/'common_rows.npy', common)
    save(ROOT/'CROSS_TARGET_READY.json', dict(target60_models=['chronos2', 't0', 'timestral'],
                                           common_rows=int(common.sum()), source20=str(SOURCE_SCR)))


def fit_stream(model, horizon):
    prepare()
    name = f'{model}_t{horizon}'
    destination = ROOT/'fits'/name
    if (destination/'COMPLETE.json').exists():
        return
    panel = pd.read_parquet(ROOT/'panel.parquet')
    target = f'target_binned_return_{horizon}'
    outputs = []
    fallbacks = []
    for block, start in enumerate(range(200, panel.step.max()+1, 130)):
        part = fold_panel(panel, start, horizon)
        if part is None:
            continue
        q, protocol = part
        root = ROOT/'blocks'/name/f'f{block:02d}'
        scratch = SCR/'blocks'/name/f'f{block:02d}'
        if (root/'COMPLETE.json').exists():
            outputs.append(pd.read_parquet(root/'predictions.parquet'))
            if model == 'momentum':
                fallbacks.append(pd.read_parquet(root/'momentum_baseline.parquet'))
            continue
        root.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        for link, actual in [('common_rows.npy', SCR/'common_rows.npy'),
                             ('embeddings', (SOURCE_SCR if horizon == 20 else SCR)/'embeddings')]:
            if not (scratch/link).exists():
                (scratch/link).symlink_to(actual, target_is_directory=link == 'embeddings')
        parquet(root/'panel.parquet', q)
        save(root/'protocol.json', protocol)
        E.ROOT, E.SCR, E.TARGET, E.RUN = root, scratch, target, f't{horizon}'
        os.environ['E206_SCOPE_ID'] = 'e211'
        if model == 'momentum':
            E.momentum_baseline()
            forecast = pd.read_parquet(root/'momentum_baseline.parquet')
            fallbacks.append(forecast)
        else:
            norm = NORMS[model]
            choices = []
            common = np.load(SCR/'common_rows.npy')
            train_counts = q.loc[common & q.train].groupby('ucid').size()
            val_counts = q.loc[common & q.validation].groupby('ucid').size().reindex(train_counts.index, fill_value=0)
            eligible = ((train_counts >= 60) & (val_counts >= 20)).any()
            windows = (N.ROLLING if norm == 'rolling' else [126]) if eligible else []
            for window in windows:
                pred, records, params, info = E.solve_assets(model, norm, window, 'C', 'selection', q)
                if records:
                    choices.append((float(np.mean([r['mse'] for r in records])), window, records, params, info))
            forecast = q.loc[q.test, ['row', 'ucid', 'symbol', 'date', target, 'meta_model']].copy()
            forecast['prediction'] = np.nan
            if choices:
                loss, window, records, params, info = min(choices, key=lambda v: (v[0], v[1]))
                save(root/'selection.json', dict(window=window, loss=loss, assets=records, design=info,
                     candidates=[dict(window=c[1], loss=c[0]) for c in choices]))
                pred, records, _, info = E.solve_assets(model, norm, window, 'C', 'final', q, params)
                forecast['prediction'] = pred[q.test.to_numpy()]
                save(root/'parameters.json', dict(assets=records, design=info))
            else:
                save(root/'selection.json', dict(status='warmup_no_eligible_asset_models'))
        parquet(root/'predictions.parquet', forecast)
        save(root/'COMPLETE.json', dict(rows=len(forecast), predictions=int(forecast.prediction.notna().sum())))
        outputs.append(forecast)
        print('BLOCK', name, block, protocol['origin'], forecast.prediction.notna().sum(), flush=True)
    assert outputs
    result = pd.concat(outputs, ignore_index=True)
    assert result.row.is_unique
    parquet(destination/'predictions.parquet', result)
    if fallbacks:
        parquet(destination/'baseline.parquet', pd.concat(fallbacks, ignore_index=True))
    save(destination/'COMPLETE.json', dict(task=dict(tag=name, model=model), horizon=horizon,
                                          blocks=len(outputs), predictions=int(result.prediction.notna().sum()), time=time.time()))


def lagged_snapshots(meta, date, symbols, cache):
    snapshot = MR.snapshot_for(meta.date.unique(), date)
    if snapshot is None:
        return np.full(len(symbols), np.nan), snapshot
    if snapshot not in cache:
        cache[snapshot] = meta.loc[meta.date.eq(snapshot)].drop_duplicates('symbol', keep='last').set_index('symbol').prediction
    assert snapshot <= date-pd.Timedelta(days=5)
    return cache[snapshot].reindex(symbols).to_numpy(float), snapshot


def score_matrix(predictions, target20, target60, meta):
    """Same input columns and rows, independently scored on both stored targets."""
    return {h: E.matrix_scores(predictions, target, meta)
            for h, target in [(20, target20), (60, target60)]}


def score():
    if (ROOT/'COMPLETE.json').exists():
        return
    panel = pd.read_parquet(ROOT/'panel.parquet').sort_values(['date', 'ucid']).reset_index(drop=True)
    assert np.array_equal(panel.row, np.arange(len(panel)))
    names = [f'{model}_t{h}' for h in [20, 60] for model in MODELS]
    predictions = []
    for name in names:
        frame = pd.read_parquet(ROOT/'fits'/name/'predictions.parquet')
        assert frame.row.is_unique
        predictions.append(frame.set_index('row').prediction.reindex(panel.row).to_numpy())
    raw = np.column_stack(predictions)
    causal = np.isfinite(raw).all(axis=1) & np.load(SCR/'common_rows.npy')
    all_predictions = np.full((len(panel), 16), np.nan, np.float64)
    all_predictions[:, :8] = raw
    names += [name+'_r' for name in names.copy()]
    meta = pd.read_parquet(REPO/'data/crypto/v2.0/historical_meta_models.parquet').reset_index()[['date', 'symbol', 'prediction']]
    meta = meta.loc[meta.date < pd.Timestamp('2026-02-20')].copy()
    snapshots, cache, groups = [], {}, []
    scored = causal & np.isfinite(panel.target_binned_return_20) & np.isfinite(panel.target_binned_return_60) & np.isfinite(panel.meta_model)
    baseline = np.full(len(panel), .5)
    mean_control = pd.read_parquet(ROOT/'fits/momentum_t20/baseline.parquet').set_index('row')
    base_rows = panel.row.isin(mean_control.index)
    baseline[base_rows] = mean_control.baseline.reindex(panel.loc[base_rows, 'row']).to_numpy()
    routed = np.empty((len(panel), 16), np.float64)
    for date, indices in panel.groupby('date', sort=True).indices.items():
        eligible = indices[causal[indices]]
        past, snapshot = lagged_snapshots(meta, pd.Timestamp(date), panel.loc[eligible, 'symbol'], cache)
        passthrough = 0
        if len(eligible):
            for column in range(8):
                residual, beta = MR.residualise(raw[eligible, column], past)
                all_predictions[eligible, column+8] = residual
                passthrough += int(np.isnan(beta))
            reference, ref_snapshot = MR.lagged_meta(meta, panel.loc[eligible, 'symbol'], date)
            assert snapshot == ref_snapshot
            np.testing.assert_array_equal(past, reference)
        snapshots.append(dict(date=date, snapshot=snapshot, eligible=len(eligible),
                              meta_covered=int(np.isfinite(past).sum()), passthrough=passthrough))
        base_ranks = E.rank_normalise(baseline[indices])
        for column in range(16):
            candidates = np.where(causal[indices], all_predictions[indices, column], np.nan)
            routed[indices, column] = E.route_slots(base_ranks, candidates)
        rows = indices[np.asarray(scored)[indices]]
        if len(rows) >= 10:
            groups.append((pd.Timestamp(date), rows))
    assert groups
    assert np.isfinite(routed).all()
    parquet(ROOT/'standalone_forecasts.parquet', panel[['row', 'ucid', 'symbol', 'date']].join(pd.DataFrame(all_predictions, columns=names)))
    parquet(SCR/'routed_standalone.parquet', panel[['row', 'date']].join(pd.DataFrame(routed, columns=names)))
    parquet(ROOT/'coverage.parquet', panel[['row', 'ucid', 'date']].assign(causal_common=causal, scored=scored))
    parquet(ROOT/'residualization_snapshots.parquet', pd.DataFrame(snapshots))
    combos = definitions()
    assert len(combos) == 6884
    configs = [dict(id=f'e211_trial_{i:04d}', members=[names[j] for j in members], size=len(members))
               for i, members in enumerate(combos)]
    save(ROOT/'trials.json', configs)
    scores = {h: np.lib.format.open_memmap(SCR/f'scores_t{h}.npy', mode='w+', dtype='float64', shape=(len(groups), len(combos), 2)) for h in WALLS}
    weight = np.zeros((16, len(combos)), np.float64)
    for column, members in enumerate(combos):
        weight[list(members), column] = 1/len(members)
    for d, (date, rows) in enumerate(groups):
        values = all_predictions[rows].astype(float)
        ranks = (rankdata(values, method='average', axis=0)-.5)/len(rows)
        for first in range(0, len(combos), 128):
            last = min(first+128, len(combos))
            ensemble = ranks @ weight[:, first:last]
            by_target = score_matrix(ensemble, panel.target_binned_return_20.to_numpy()[rows],
                                      panel.target_binned_return_60.to_numpy()[rows], panel.meta_model.to_numpy()[rows])
            for h, (corr, mmc) in by_target.items():
                scores[h][d, first:last, 0], scores[h][d, first:last, 1] = corr, mmc
        if d % 10 == 0:
            for array in scores.values():
                array.flush()
            save(ROOT/'status.json', dict(stage='scoring_ensembles', completed_dates=d+1, total_dates=len(groups), time=time.time()))
    summaries, daily_pay = [], {}
    for h, array in scores.items():
        array.flush()
        pay = .1*array[:, :, 0]+array[:, :, 1]
        daily_pay[h] = pay
        for i, config in enumerate(configs):
            p = pay[:, i]
            cum = np.r_[0, p.cumsum()]
            summaries.append(dict(**config, fit_targets=','.join(sorted({str(20 if '_t20' in name else 60) for name in config['members']})),
                                  evaluation_target=h, kind='standalone' if config['size'] == 1 else 'ensemble',
                                  dates=len(groups), corr=array[:, i, 0].mean(), mmc=array[:, i, 1].mean(), payout=p.mean(),
                                  **{'sharpe.p': p.mean()/p.std(ddof=0) if p.std(ddof=0) > 0 else np.nan,
                                     'dd.p': (np.maximum.accumulate(cum)-cum).max()}))
    table = pd.DataFrame(summaries).sort_values(['evaluation_target', 'payout', 'id'], ascending=[True, False, True])
    table['members'] = table.members.map(' + '.join)
    table.to_csv(ROOT/'all_trials.csv', index=False)
    report(panel, table, configs, groups, daily_pay)
    # experiment_db indexes the eight head streams; scores are payout-on20 summaries.
    controls = table[(table.evaluation_target == 20) & table.kind.eq('standalone') & ~table.members.str.endswith('_r')].copy()
    controls['tag'] = controls.members
    controls.rename(columns={'dates': 'n_dates'}).to_csv(ROOT/'all_model_window_rows.csv', index=False)
    save(ROOT/'audit.json', dict(outcome='PASS WITH CONDITIONS', coverage='Single common causal predictor sample; identical finite20/60-label and meta scored rows for every trial',
         residualization='Native strength0.5/D-5 snapshot helper; cached inputs matched direct live helper for every scored origin',
         routing='Full-panel finite standalone forecasts; exact uncovered baseline ranks and covered slot set asserted by route_slots',
         limits=['Current-vendor momentum histories and pretrained checkpoint overlap uncertified.',
                 'Same-history ensemble selection is exploratory; no independent confirmation, native-feature marginal or live deployment claim.',
                 'Payout-score research only; unresolved source-price anomaly is not treated as investment performance.']))
    save(ROOT/'COMPLETE.json', dict(status='success', heads=8, standalone_variants=16, ensembles=6868, trials=len(table),
                                    dates=len(groups), report=str(ROOT/'E211_cross_target_tsfm.pdf'), time=time.time()))


def report(panel, table, configs, groups, payouts):
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    document = '<html><head><meta charset="utf-8"></head><body><h1>E211 — Cross-target TSFM and momentum ensembles</h1>'
    document += ('<p>Full-history chronological predictions, joined across forecast blocks; no reserved final test. '
                 'Original source dates without eligible forecasts are warm-up, not in-sample predictions. '
                 'Cumulative payout uses the common scored dates. t20/t60 denote head training target; _r denotes '
                 'existing strength0.5 prediction residualization against the meta snapshot at least5 days old. '
                 'Ensembles average member ranks equally, then rescore. Payout=0.1CORR+historical-meta proxyMMC; '
                 'sharpe.p=mean/population standard deviation, no annualization. Top5 chosen on the same displayed history; exploratory.</p>')
    dates = [date for date, _ in groups]
    index = {config['id']: i for i, config in enumerate(configs)}
    for horizon in WALLS:
        section = table[table.evaluation_target == horizon]
        top = section[section.kind.eq('ensemble')].head(5)
        controls = section[section.kind.eq('standalone')]
        fig, ax = plt.subplots(figsize=(15, 9), layout='constrained')
        for row in pd.concat([top, controls]).itertuples():
            ax.plot(dates, payouts[horizon][:, index[row.id]].cumsum(), label=row.members,
                    linewidth=2.8 if row.kind == 'ensemble' else .85,
                    alpha=1 if row.kind == 'ensemble' else .6,
                    linestyle='-' if row.kind == 'ensemble' else '--')
        ax.set(title=f'Target{horizon}: top5 ensembles plus16 standalone variants', xlabel='Date', ylabel='Cumulative payout score')
        ax.grid(alpha=.25)
        ax.legend(loc='upper left', fontsize=7, ncol=2)
        plot = f'top5_and_standalones_t{horizon}.png'
        fig.savefig(ROOT/plot, dpi=170)
        plt.close(fig)
        document += f'<h2>Evaluation target{horizon}</h2><img style="width:100%" src="{plot}">'
        document += top.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    document += '<h2>All trials — target then payout descending</h2>'
    document += table.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    document += '<h2>Frozen protocol</h2><pre>'+html.escape((ROOT/'PREPARED.json').read_text())+'</pre></body></html>'
    page = ROOT/'E211_cross_target_tsfm.html'
    page.write_text(document)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lovelace', 'python3', str(REPO/'src/analysis/html_pdf.py'), str(page)], check=True, timeout=300)


def supervise():
    prepare()
    state_path = ROOT/'supervisor.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(owned={}, attempts={}, launches=[])
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    panel = pd.read_parquet(ROOT/'panel.parquet')
    try:
        while not (ROOT/'COMPLETE.json').exists():
            probes = hosts.reachable(hosts.probe_all(timeout=3))
            snapshots = manager.status(names=[h.name for h in probes], timeout=3)
            active, failed = E.unit_states(snapshots)
            old = E.SCR
            E.SCR = SCR
            E.recover_locks()
            E.SCR = old
            for unit in list(state['owned']):
                if unit not in active:
                    del state['owned'][unit]
            jobs = []
            ready = (ROOT/'CROSS_TARGET_READY.json').exists()
            if not ready:
                complete = []
                for model in ['chronos2', 't0', 'timestral']:
                    wanted = panel.groupby('ucid').covered.any()
                    directory = SCR/'embeddings'/model
                    remaining = [a for a in wanted.index[wanted] if not (directory/f'asset_{a}.npz').exists()]
                    if not remaining:
                        complete.append(model)
                        continue
                    if not any(not (directory/f'asset_{a}.lock').exists() for a in remaining):
                        continue
                    for worker in range(2):
                        name = f'e211_{model}_extract_cpu{worker}'
                        command = ('env '+shlex.join([f'TSFM_EMBED_ROOT={ROOT}', f'TSFM_EMBED_SCR={SCR}',
                                   'TSFM_EMBED_TARGET=target_binned_return_60', 'TSFM_EMBED_WALL=91'])+
                                   f' {E.VENVS[model]} -u -m src.data_mani.e206_tsfm_embeddings --model {model} --device cpu')
                        jobs.append(budget.Job(name, command, cores=2, ram_gb=12))
                save(ROOT/'status.json', dict(stage='extracting_target60', completed_models=complete, time=time.time()))
                if len(complete) == 3:
                    collect()
                    ready = True
            if ready:
                for horizon in WALLS:
                    for model in MODELS:
                        if (ROOT/'fits'/f'{model}_t{horizon}'/'COMPLETE.json').exists():
                            continue
                        name = f'e211_{model}_fit_t{horizon}'
                        jobs.append(budget.Job(name, f'{PYTHON} -u experiments/e211_cross_target_tsfm.py fit --model {model} --target {horizon}',
                                               cores=2 if model == 'momentum' else 4, ram_gb=32))
                if not jobs:
                    score()
                    break
                save(ROOT/'status.json', dict(stage='fitting_heads', completed_streams=len(list((ROOT/'fits').glob('*/COMPLETE.json'))), time=time.time()))
            pending = []
            for job in jobs:
                if 'nmr-'+job.name in active:
                    continue
                if state['attempts'].get(job.name, 0) >= 3:
                    raise RuntimeError('Three attempts exhausted: '+job.name+'; inspect logs/cluster/'+job.name+'.out')
                pending.append(job)
            placements, _ = budget.distribute(pending, probes, share=budget.SHARE)
            for placement in placements:
                subprocess.run(['ssh', '-o', 'BatchMode=yes', placement.host,
                                'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user reset-failed '+placement.unit],
                               capture_output=True, timeout=15)
                ok, message = manager.launch(placement)
                if ok:
                    record = dict(host=placement.host, unit=placement.unit, cpus=list(placement.cpus))
                    state['owned'][placement.unit] = record
                    state['launches'].append(record)
                    state['attempts'][placement.job.name] = state['attempts'].get(placement.job.name, 0)+1
                print('LAUNCH', ok, message, flush=True)
            state['updated'] = time.time()
            save(state_path, state)
            time.sleep(30)
    except Exception as error:
        save(ROOT/'FAILED.json', dict(error=str(error), traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        for host in {row['host'] for row in state['launches']}:
            manager.stop(host, list({row['unit'] for row in state['launches'] if row['host'] == host}), timeout=20)
        save(ROOT/'resource_lease.json', dict(default_share=budget.SHARE, active_share=budget.SHARE,
                                             status='restored', global_default_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'supervise', 'fit', 'score'])
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--target', type=int, choices=[20, 60])
    args = parser.parse_args()
    if args.command == 'fit':
        fit_stream(args.model, args.target)
    else:
        globals()[args.command]()
