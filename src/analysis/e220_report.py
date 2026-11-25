"""Matched H versus H+momentum portfolio comparison; no ensemble selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.analysis.e218_ktz_report import metrics


def report(root,source):
    from experiments import e206_momentum3_tsfm as E
    from src.analysis.asset_momentum import route_slots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=pd.read_parquet(source/'momentum_baseline.parquet').sort_values(['date','ucid']).reset_index(drop=True)
    frames=sorted((root/'fits').glob('*/predictions.parquet'));assert len(frames)==24
    names=[p.parent.name for p in frames]
    raw=np.column_stack([pd.read_parquet(p).set_index('row').prediction.reindex(base.row).to_numpy(float) for p in frames])
    valid=np.isfinite(raw).all(1)&np.isfinite(base[E.TARGET])&np.isfinite(base.meta_model)
    dates=[];scores=[];counts=[];routed=np.empty_like(raw)
    for date,idx in base.groupby('date',sort=True).indices.items():
        for j in range(24):routed[idx,j]=route_slots(base.baseline_rank.to_numpy()[idx],raw[idx,j])
        rows=idx[valid.to_numpy()[idx]]
        if len(rows)<10:continue
        c,m=E.matrix_scores(raw[rows],base.loc[rows,E.TARGET].to_numpy(),base.loc[rows,'meta_model'].to_numpy())
        dates.append(date);scores.append(np.column_stack([c,m]));counts.append(len(rows))
    assert np.isfinite(routed).all()
    np.savez_compressed(root/'forecasts.npz',row=base.row.to_numpy(),raw=raw,routed=routed,common=valid.to_numpy())
    scores=np.stack(scores);table=pd.DataFrame(dict(tag=names,n_dates=len(dates),mean_coverage=np.mean(counts),**metrics(scores))).sort_values('payout',ascending=False)
    table.to_csv(root/'all_model_window_rows.csv',index=False);np.savez_compressed(root/'daily_scores.npz',date=np.asarray(dates),scores=scores)
    pay=.1*scores[:,:,0]+scores[:,:,1]
    with PdfPages(root/'E220_JA_heads.pdf') as pdf:
        for variant in ['H','Hmom']:
            t=table[table.tag.str.contains('_'+variant+'_')]
            fig,ax=plt.subplots(figsize=(16,9));ax.axis('off');ax.set_title('E220 — '+variant+' per-asset models; sorted by payout')
            columns=['tag','corr','mmc','payout','sharpe.p','maxdd.p']
            cells=[[row['tag']]+[f'{row[k]:.6f}' for k in columns[1:]] for _,row in t.iterrows()]
            tab=ax.table(cellText=cells,colLabels=columns,loc='center');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.8)
            fig.text(.07,.07,f'Matched common sample: {len(dates)} dates, {sum(counts)} asset/date rows. Payout=0.1 CORR + historical-meta proxy MMC.\nHistorical research; inherited data-vintage and pretrained-checkpoint limitations. No native-feature marginal or live claims.')
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(16,10))
        for j,n in enumerate(names):ax.plot(dates,pay[:,j].cumsum(),label=n)
        ax.set(ylabel='Cumulative payout',title='E220 — H versus H plus own momentum');ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    E.save(root/'audit.json',dict(outcome='PASS WITH CONDITIONS',common_keys=int(valid.sum()),dates=len(dates),full_panel_finite=True,uncovered_rank_slots_preserved=True,conditions=['Historical standalone model comparison, no production/marginal attribution','Original source vintage/pretraining limitations retained']))
