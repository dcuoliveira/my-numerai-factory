"""Explicit design-C inputs and fresh per-asset native KTZ estimators."""
import copy
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

ALIASES = {'lgbm_ktz': 'lgb_ktz', 'mlp_ktz': 'mlp_ktz', 'catboost_ktz': 'catboost_ktz'}


def native(value):
    return json.loads(json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else list(x)))


def wrapper(family, cores=1):
    from src.prediction.models import MODELS
    return MODELS[ALIASES[family]](task='regression', seed=0, n_jobs=cores)


def candidates(family):
    w = wrapper(family)
    safe = {k: list(v)[0] for k, v in w.param_grid.items()}
    if family == 'mlp_ktz': safe['mlp__alpha'] = .1
    result = [native(safe)]
    for value in ParameterSampler(w.param_grid, n_iter=20, random_state=218):
        value = native(value)
        if value not in result: result.append(value)
        if len(result) == 4: break
    return result


def materialize(h, assets, steps, valid, vocabulary, count, path):
    """No projection: [date, training-asset slot x hidden coordinate]."""
    assert len(set(map(str,vocabulary)))==len(vocabulary)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npy')
    phi = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                  shape=(count, len(vocabulary)*h.shape[1]))
    phi[:] = 0
    assets = np.asarray(assets).astype(str)
    for j, asset in enumerate(vocabulary):
        rows = np.flatnonzero(valid & (assets == str(asset)))
        assert len(np.unique(steps[rows])) == len(rows)
        phi[steps[rows], j*h.shape[1]:(j+1)*h.shape[1]] = h[rows]
    phi.flush(); del phi; temporary.replace(path)
    return np.load(path, mmap_mode='r')


class FrozenAffine(RegressorMixin, BaseEstimator):
    """Exact inference representation of a fitted identity MLP, not a new fit."""
    def __init__(self, coef, intercept):
        self.coef = coef; self.intercept = intercept

    def predict(self, x):
        return np.asarray(x) @ self.coef + self.intercept

    def fit(self, x, y=None):
        raise TypeError('Inference snapshot only; train a fresh native MLP instead')

    def __sklearn_is_fitted__(self):
        return True


def snapshot(estimator, family, query):
    if family != 'mlp_ktz': return estimator
    result = copy.copy(estimator)
    result.steps = list(estimator.steps)
    name, net = estimator.steps[-1]
    assert net.activation == 'identity' and net.out_activation_ == 'identity'
    coef = net.coefs_[-1]
    intercept = net.intercepts_[-1]
    for j in range(len(net.coefs_)-2, -1, -1):
        intercept = net.intercepts_[j] @ coef + intercept
        coef = net.coefs_[j] @ coef
    result.steps[-1] = (name, FrozenAffine(coef[:, 0], float(intercept[0])))
    np.testing.assert_allclose(result.predict(query), estimator.predict(query), rtol=1e-7, atol=1e-8)
    return result


def fit_asset(family, grid, train_x, train_y, val_x, val_y, final_x, final_y, query, cores, destination):
    with threadpool_limits(limits=cores):
        return _fit_asset(family,grid,train_x,train_y,val_x,val_y,final_x,final_y,query,cores,destination)


def _fit_asset(family, grid, train_x, train_y, val_x, val_y, final_x, final_y, query, cores, destination):
    """Each candidate and refit gets a newly cloned estimator and new weights."""
    started = time.time()
    w = wrapper(family, cores)
    keep = np.std(train_x, axis=0, dtype=np.float64) >= 1e-8
    if not keep.any(): raise ValueError('No nonconstant training inputs')
    tx = np.asarray(train_x[:, keep], dtype=np.float64 if family == 'mlp_ktz' else np.float32)
    vx = np.asarray(val_x[:, keep], dtype=tx.dtype)
    choices = []
    for parameters in grid:
        estimator = clone(w.ModelClass).set_params(**parameters)
        try:
            estimator.fit(tx, train_y)
            prediction = estimator.predict(vx)
            if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite validation prediction')
            loss = float(np.mean((prediction-val_y)**2))
            choices.append(dict(parameters=parameters, mse=loss))
        except Exception as error:
            choices.append(dict(parameters=parameters, error=str(error)))
        del estimator
    successful = [v for v in choices if 'mse' in v]
    if not successful: raise RuntimeError('Every candidate failed: '+str(choices))
    selected = min(successful, key=lambda v: v['mse'])
    del tx, vx
    keep = np.std(final_x, axis=0, dtype=np.float64) >= 1e-8
    fx = np.asarray(final_x[:, keep], dtype=np.float64 if family == 'mlp_ktz' else np.float32)
    qx = np.asarray(query[:, keep], dtype=fx.dtype)
    estimator = clone(w.ModelClass).set_params(**selected['parameters'])
    estimator.fit(fx, final_y)
    prediction = estimator.predict(qx)
    if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite final prediction')
    frozen = snapshot(estimator, family, qx[:min(10, len(qx))])
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    joblib.dump(dict(estimator=frozen, columns=np.flatnonzero(keep)), temporary, compress=3)
    temporary.replace(destination)
    return prediction, dict(selected=selected, choices=choices, n_train=len(train_y), n_validation=len(val_y),
                            n_final=len(final_y), columns=int(keep.sum()), input_columns=keep.size,
                            seconds=time.time()-started, sklearn_native_initialization=family == 'mlp_ktz')
