"""E210: exhaustive equal-rank ensembles of the five recommended E206 rows."""
import hashlib
import itertools
import json
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'experiments')]
from experiments.e206_momentum3_tsfm import TARGET, matrix_scores
from src.analysis.asset_momentum import route_slots
from src.live.submit import rank_normalise

SOURCE = REPO/'results/reports/crypto/e206_momentum3_tsfm'
ROOT = REPO/'results/reports/crypto/e210_tsfm_ensembles'
SCRATCH = Path('/scratch/danielco/e206_momentum3_tsfm')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SOURCE/'recommended_common_results.csv').sort_values('model')
    assert len(selected) == 5 and selected.model.is_unique
    baseline = pd.read_parquet(SOURCE/'momentum_baseline.parquet').sort_values(['date', 'ucid']).reset_index(drop=True)
    assert baseline.row.is_unique and np.isfinite(baseline.baseline_rank).all()
    mask_path = SCRATCH/'portfolios'/selected.iloc[0].tag/'common'
    mask_frame = pd.read_parquet(mask_path, columns=['row', 'eligible'])
    assert mask_frame.row.is_unique
    common_rows = set(mask_frame.loc[mask_frame.eligible, 'row'])
    mask = baseline.row.isin(common_rows).to_numpy()
    expected_dates = pd.read_csv(SOURCE/'recommended_common_daily_payout.csv', usecols=['date']).date
    expected_dates = pd.to_datetime(expected_dates)
    finite_labels = np.isfinite(baseline[TARGET]) & np.isfinite(baseline.meta_model)
    scoring_mask = mask & finite_labels & baseline.date.isin(expected_dates)
    pd.DataFrame({'row': baseline.row, 'common': mask, 'scored': scoring_mask}).to_parquet(ROOT/'coverage.parquet', index=False)
    source_hashes = {}
    raw = []
    member_names = []
    source_scores = {}
    for member in selected.itertuples():
        path = (SOURCE/'momentum_baseline.parquet' if member.model == 'momentum'
                else SOURCE/'fits'/member.tag/'predictions.parquet')
        frame = pd.read_parquet(path, columns=['row', 'prediction'])
        assert frame.row.is_unique
        prediction = frame.set_index('row').prediction.reindex(baseline.row).to_numpy()
        assert np.isfinite(prediction[mask]).all()
        raw.append(prediction)
        member_names.append(member.model)
        source_hashes[str(path)] = digest(path)
        source_scores[member.model] = pd.read_parquet(SOURCE/'scores'/f'{member.tag}_common.parquet').set_index('date')
    raw = np.column_stack(raw)
    ranks = np.full_like(raw, np.nan)
    groups = list(baseline.groupby('date', sort=True).indices.items())
    for _, indices in groups:
        covered = indices[mask[indices]]
        for column in range(5):
            ranks[covered, column] = rank_normalise(raw[covered, column])

    definitions = [(i,) for i in range(5)]
    definitions += [members for n in range(2, 6) for members in itertools.combinations(range(5), n)]
    assert len(definitions) == 31
    stats = []
    all_daily = []
    maximum_replay_error = 0.
    for members in definitions:
        names = [member_names[i] for i in members]
        tag = 'e210_' + '_'.join(names)
        label = ' + '.join(names)
        candidate = (raw[:, members[0]].copy() if len(members) == 1
                     else ranks[:, members].mean(axis=1))
        candidate[~mask] = np.nan
        routed = baseline.baseline_rank.to_numpy().copy()
        daily = []
        for date, indices in groups:
            routed[indices] = route_slots(baseline.baseline_rank.to_numpy()[indices], candidate[indices])
            scored = indices[np.asarray(scoring_mask)[indices]]
            if len(scored) < 10:
                continue
            corr, mmc = matrix_scores(candidate[scored, None], baseline[TARGET].to_numpy()[scored],
                                      baseline.meta_model.to_numpy()[scored])
            corr, mmc = float(corr[0]), float(mmc[0])
            if len(members) == 1:
                reference = source_scores[names[0]].loc[date]
                assert len(scored) == reference['n']
                error = np.abs(np.array([corr, mmc])-reference[['corr', 'mmc']].to_numpy(float))
                maximum_replay_error = max(maximum_replay_error, float(error.max()))
                np.testing.assert_allclose([corr, mmc], reference[['corr', 'mmc']].to_numpy(float), rtol=1e-6, atol=1e-8)
            daily.append(dict(date=date, n=len(scored), corr=corr, mmc=mmc, payout=.1*corr+mmc))
        daily = pd.DataFrame(daily)
        assert np.array_equal(daily.date.to_numpy(), expected_dates.to_numpy())
        pay = daily.payout.to_numpy()
        cumulative = np.r_[0., pay.cumsum()]
        drawdown = float((np.maximum.accumulate(cumulative)-cumulative).max())
        std = pay.std(ddof=0)
        stats.append(dict(tag=tag, members=label, size=len(members), kind='standalone' if len(members) == 1 else 'ensemble',
                          dates=len(pay), corr=daily['corr'].mean(), mmc=daily.mmc.mean(), payout=pay.mean(),
                          **{'sharpe.p': pay.mean()/std if std > 0 else np.nan, 'dd.p': drawdown}))
        (ROOT/'scores').mkdir(exist_ok=True)
        daily.to_parquet(ROOT/'scores'/f'{tag}.parquet', index=False)
        (ROOT/'forecasts').mkdir(exist_ok=True)
        forecasts = baseline[['row', 'ucid', 'date']].copy()
        forecasts['candidate'] = candidate
        forecasts['routed'] = routed
        assert np.isfinite(routed).all()
        np.testing.assert_array_equal(routed[~mask], baseline.baseline_rank.to_numpy()[~mask])
        forecasts.to_parquet(ROOT/'forecasts'/f'{tag}.parquet', index=False)
        all_daily.append(daily.assign(tag=tag, members=label))
    table = pd.DataFrame(stats).sort_values(['payout', 'members'], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, 'rank', np.arange(1, len(table)+1))
    table.to_csv(ROOT/'consolidated_results.csv', index=False)
    pd.concat(all_daily, ignore_index=True).to_parquet(ROOT/'daily_scores.parquet', index=False)
    top = table[table.kind.eq('ensemble')].head(2)
    plotted = pd.concat([top, table[table.kind.eq('standalone')]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 6), layout='constrained')
    for row in plotted.itertuples():
        daily = pd.read_parquet(ROOT/'scores'/f'{row.tag}.parquet')
        ax.plot(daily.date, daily.payout.cumsum(), label=row.members,
                linewidth=2.6 if row.kind == 'ensemble' else 1.3,
                linestyle='-' if row.kind == 'ensemble' else '--')
    ax.set(title='Top two ensembles and five standalone models — common assets',
           xlabel='Date', ylabel='Cumulative payout score (0.1 CORR + proxy MMC)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    plot = 'e210_top_ensembles_cumulative_payout.png'
    fig.savefig(ROOT/plot, dpi=180)
    plt.close(fig)
    shown = table.drop(columns='tag')
    intro = ('<h2>E210 — Consolidated standalone and ensemble search</h2>'
             '<p>All 26 equal-rank ensembles: 10 pairs, 10 triples, 5 quadruples and 1 quintuple, '
             'plus the five standalone controls. Member predictions are ranked within each date, '
             'averaged equally and freshly scored on the identical common assets and 297 dates. '
             'Rows sorted by mean payout; sharpe.p = mean/std(ddof=0), without annualization; '
             'dd.p is maximum drawdown of cumulative payout scores. '
             'Top two ensembles are selected by payout on this same history: exploratory search, '
             'not independent confirmation or investment-return performance.</p>')
    section = intro + shown.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    section += '<h2>Top two ensembles alongside the five standalone models</h2>'
    section += f'<img style="width:100%" src="{plot}">'
    (ROOT/'E210_tsfm_ensembles.html').write_text('<html><head><meta charset="utf-8"></head><body>'+section+'</body></html>')
    shutil.copy2(ROOT/plot, SOURCE/plot)
    page = SOURCE/'E206_momentum3_tsfm.html'
    document = re.sub(r'<!-- E210 START -->.*?<!-- E210 END -->', '', page.read_text(), flags=re.S)
    document = document.replace('</body>', '<!-- E210 START -->'+section+'<!-- E210 END --></body>')
    page.write_text(document)
    manifest = dict(experiment='E210', host=socket.gethostname(), members=selected.to_dict('records'),
                    source_hashes=source_hashes, coverage_hash=digest(ROOT/'coverage.parquet'),
                    ensembles=26, controls=5, top_two=top.members.tolist(),
                    selection='Exploratory maximum mean payout on the same displayed history; no new labels or fits.',
                    audit=dict(outcome='PASS WITH CONDITIONS', standalone_replay_max_error=maximum_replay_error,
                               common_assets_and_dates_verified=True, uncovered_baseline_ranks_exact=True,
                               covered_rank_slots_exact=True,
                               conditions=['Prior E205/E206 signal and configuration selection on overlapping history.',
                                           'Current-vendor histories and pretrained checkpoint overlap are not certified point-in-time.',
                                           'Standalone binned-target research only; no marginal-native-feature or live-parity claim.',
                                           'Source-price anomaly remains unresolved; no actual-return claims are made here.']))
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    for path in [page, ROOT/'E210_tsfm_ensembles.html']:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', 'lovelace', 'python3',
                        str(REPO/'src/analysis/html_pdf.py'), str(path)], check=True, timeout=300)
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(status='success', ensembles=26, controls=5,
                                                    report=str(ROOT/'E210_tsfm_ensembles.pdf')), indent=2)+'\n')
    print(top[['members', 'payout', 'sharpe.p']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
