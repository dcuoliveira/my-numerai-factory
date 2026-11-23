"""Publish E215 historical artifacts under the production registry's run paths."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    source = root/'results/reports/crypto/e215_book_fixed_ensembles'
    keys = pd.read_parquet(source/'matched_predictions.parquet')
    predictions = np.load(source/'trial_predictions.npz')
    np.testing.assert_array_equal(keys.row, predictions['rows'])
    daily = pd.read_parquet(source/'daily_scores.parquet')
    for index, run in [(3, 'e215_jaci01'), (8, 'e215_cu_sa_ia_bo_ma_ja01')]:
        frame = keys[['symbol', 'date', 'target_binned_return_20', 'meta_model']].copy()
        frame['prediction'] = predictions['predictions'][:, index]
        for path, artifact in [
            (root/'results/forecast/crypto/meta_resid'/f'{run}.csv', frame),
            (root/'results/per_period/crypto/meta_resid'/f'{run}.csv', daily[daily.id.eq(index)].drop(columns='id')),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            artifact.to_csv(path, index=False)


if __name__ == '__main__':
    main()
