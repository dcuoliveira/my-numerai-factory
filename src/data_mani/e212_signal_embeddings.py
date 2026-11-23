"""E212 dataset-specific frozen pre-head representations; only numpy/pandas/torch in model envs.

Each cache unit contains one asset. Independent origins of that asset may be
batched; no other asset is ever passed to the frozen model in that call.
"""
import os
for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TABPFN_DISABLE_TELEMETRY']:
    os.environ[key] = '1'
os.environ['HF_HOME'] = '/scratch/danielco/hf'
os.environ.setdefault('TSFM_CHUNK', '4')
os.environ['TSFM_AMP'] = 'fp32'
import argparse
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('TSFM_EMBED_ROOT', str(REPO/'results/reports/crypto/e212_volatility99_tsfm')))
SCR = Path(os.environ.get('TSFM_EMBED_SCR', str(REPO/'results/reports/crypto/e212_volatility99_tsfm/cache/t20')))
TARGET = os.environ.get('TSFM_EMBED_TARGET', 'target_binned_return_20')
CTX, MINIMUM, WALL = 256, 32, int(os.environ.get('TSFM_EMBED_WALL', '35'))
CHECKPOINTS = {
    'timestral': ('mistralai/Timestral-209M', '9874d252ab0aae16f7f7c999f98f98d6ad7d8365'),
    't0': ('theforecastingcompany/t0-gamma', 'fef67abad3eed4f227c7b536a6e8ee4e9ab1080d'),
    'chronos2': ('amazon/chronos-2', '29ec3766d36d6f73f0696f85560a422f50e8498c'),
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{socket.gethostname()}_{os.getpid()}.tmp.json'); tmp.write_text(json.dumps(data, indent=2, default=str)); tmp.replace(path)


def inputs(y, x, grid, step):
    wall = int(np.searchsorted(grid, grid[step]-np.timedelta64(WALL, 'D'), side='right')-1)
    if wall < 0:
        return None
    lo = max(0, wall-CTX+1)
    if np.isfinite(y[lo:wall+1]).sum() < MINIMUM or not np.isfinite(x[step]).all():
        return None
    yy = np.full((1, CTX), np.nan, np.float32)
    xx = np.full((1, x.shape[1], CTX), np.nan, np.float32)
    yy[0, -(wall+1-lo):] = y[lo:wall+1]
    xx[0, :, -(wall+1-lo):] = x[lo:wall+1].T
    future = x[wall+1:step+1].T[None].astype(np.float32)
    return yy, xx, future, step-wall, np.array([grid[wall]], dtype='datetime64[D]')


class ChronosExtractor:
    def __init__(self, checkpoint, device):
        import torch
        from chronos import Chronos2Pipeline
        self.torch, self.device = torch, device
        self.model = Chronos2Pipeline.from_pretrained(checkpoint, device_map=device,
                                                    torch_dtype=torch.float32).model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(self.model.config.d_model)

    def states(self, y, x, f, horizon, last_dates):
        t = self.torch
        batch,variates=len(y),1+x.shape[1]
        context = np.concatenate([y[:, None, :], x], axis=1).reshape(-1, CTX)
        future = np.full((context.shape[0], horizon), np.nan, np.float32)
        for b in range(batch):future[b*variates+1:(b+1)*variates] = f[b]
        patches = (horizon+15)//16
        with t.inference_mode():
            kwargs = dict(context=t.from_numpy(context).to(self.device),
                          group_ids=t.arange(batch,device=self.device).repeat_interleave(variates),
                          future_covariates=t.from_numpy(np.nan_to_num(future, nan=0)).to(self.device),
                          future_covariates_mask=t.from_numpy(np.isfinite(future).astype(np.float32)).to(self.device),
                          num_output_patches=patches)
            enc = self.model.encode(**kwargs)[0]
            hidden = enc[0] if isinstance(enc, tuple) else enc.last_hidden_state
            return hidden[t.arange(batch,device=self.device)*variates, -patches+(horizon-1)//16].float().cpu().numpy()


def pack(arguments):
    assert len({a[3] for a in arguments})==1
    return tuple(np.concatenate([a[j] for a in arguments],axis=0) if j!=3 else arguments[0][3] for j in range(5))


def safe_states(ext,arguments):
    """Reduce only compute batch size after OOM; representation definition unchanged."""
    try:
        return ext.states(*pack(arguments))
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower() or len(arguments)==1:raise
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
        mid=len(arguments)//2
        return np.concatenate([safe_states(ext,arguments[:mid]),safe_states(ext,arguments[mid:])])


def adapter(model, device):
    if model in CHECKPOINTS:
        repository, revision = CHECKPOINTS[model]
        path = Path('/scratch/danielco/hf/hub')/('models--'+repository.replace('/', '--'))/'snapshots'/revision
        if not path.exists():
            raise FileNotFoundError('Pinned checkpoint unavailable: '+str(path))
        if model == 'chronos2':
            ext = ChronosExtractor(str(path), device)
        else:
            module = importlib.import_module('src.data_mani.tsfm_state_adapters.'+model)
            module.CKPT = str(path)
            ext = module.Extractor(device)
        checkpoint = dict(repository=repository, revision=revision, path=str(path))
    else:
        module = importlib.import_module('src.data_mani.tsfm_state_adapters.tabpfnts')
        ext = module.Extractor(device)
        if 'inference_precision' in ext.reg.get_params():
            import torch
            ext.reg.set_params(inference_precision=torch.float32)
        checkpoint = dict(path=module.CKPT_DEFAULT, sha256=module.CKPT_SHA256)
    for name in ['model']:
        net = getattr(ext, name, None)
        if net is not None:
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
    descriptions = {
        'timestral': 'model.head forward-pre-hook; target row; decoder ll+(h-1)//patch_size',
        't0': 'model.decoder forward-pre-hook; target row; context_patches-1+(h-1)//patch_size',
        'chronos2': 'encode last hidden state consumed by output_patch_embedding; target row; output patch (h-1)//16',
        'tabpfnts': 'get_embeddings(data_source=test), one D query row, n_estimators=1; contextual in-context fit, not weight training',
    }
    manifest = dict(model=model, checkpoint=checkpoint, width=int(ext.dim), extraction=descriptions[model],
                    context=CTX, label_wall_calendar_days=WALL, min_resolved=MINIMUM,
                    channels=f"one asset: {TARGET} plus{len(json.loads((ROOT/'PREPARED.json').read_text())['features'])} supplied dataset signals; uncovered target contexts masked; past missing covariates native/masked",
                    origin=f'D; covariates only through D, labels only through D-{WALL}; no reassignment of tokens to earlier dates',
                    dtype='float32 inference requested, float16 stored; hardware execution recorded per asset', device=device,
                    checkpoint_pretraining_limit='Frozen weights may have been pretrained after historical test dates; no pretraining-overlap guarantee')
    manifest['extractor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prepared=json.loads((ROOT/'PREPARED.json').read_text())
    manifest['panel_sha256']=prepared['panel_hash']
    versions={}
    for package in ['numpy','pandas','torch','transformers','chronos-forecasting','tabpfn','tabpfn-time-series','timestral','t0']:
        try:versions[package]=package_metadata.version(package)
        except package_metadata.PackageNotFoundError:pass
    manifest['package_versions']=versions
    manifest['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES','')
    manifest['dataset']=prepared['dataset']
    manifest['features']=prepared['features']
    manifest['target']=TARGET
    manifest['fingerprint']=hashlib.sha256(json.dumps({k:manifest[k] for k in ['panel_sha256','dataset','features','target','checkpoint','extraction','context','label_wall_calendar_days','extractor_sha256','package_versions']},sort_keys=True).encode()).hexdigest()
    if model == 'tabpfnts':
        manifest['channel_difference'] = 'Tabular context X=[true panel-step index, dataset features], resolved covered y; one origin-query embedding rather than a temporal patch'
    return ext, manifest


def audit(ext, y, x, grid, steps):
    checks = []
    for s in [steps[0], steps[len(steps)//2], steps[-1]]:
        args = inputs(y, x, grid, s)
        if args is None:
            continue
        first = ext.states(*args)
        altered_y, altered_x = y.copy(), x.copy()
        wall = np.searchsorted(grid, grid[s]-np.timedelta64(WALL, 'D'), side='right')-1
        altered_y[wall+1:] = 1000.
        altered_x[s+1:] = -1000.
        changed = ext.states(*inputs(altered_y, altered_x, grid, s))
        truncated = ext.states(*inputs(y[:s+1], x[:s+1], grid[:s+1], s))
        np.testing.assert_allclose(first, changed, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(first, truncated, rtol=1e-5, atol=1e-5)
        assert first.shape == (1, ext.dim) and np.isfinite(first).all()
        checks.append(dict(origin=str(grid[s]), max_future_change=float(np.max(np.abs(first-changed))),
                           max_prefix_change=float(np.max(np.abs(first-truncated))), width=int(ext.dim)))
    args=inputs(y,x,grid,steps[-1]);single=ext.states(*args)
    other=tuple(a.copy() if isinstance(a,np.ndarray) else a for a in args)
    other[0][:]+=1
    batch=safe_states(ext,[args,other])
    np.testing.assert_allclose(single,batch[:1],rtol=1e-4,atol=1e-4)
    checks.append(dict(batch_independence_max_change=float(np.max(np.abs(single-batch[:1]))),batch=2))
    return checks


def run(model, device, smoke=False):
    directory = SCR/'embeddings'/model
    directory.mkdir(parents=True, exist_ok=True)
    ext, manifest = adapter(model, device)
    save(directory/'extractor.json', manifest)
    panel_path=ROOT/'panel.parquet'
    assert hashlib.sha256(panel_path.read_bytes()).hexdigest()==manifest['panel_sha256'], 'Frozen panel changed'
    p = pd.read_parquet(panel_path)
    assert [c for c in p if c.startswith('feature_')]==manifest['features']
    features = [c for c in p if c.startswith('feature_')]
    grid = np.sort(p.date.to_numpy().astype('datetime64[D]').take(np.arange(len(p))))
    grid = np.unique(grid)
    device_audit=directory/('audit_'+socket.gethostname()+'_'+os.environ.get('CUDA_VISIBLE_DEVICES','cpu').replace(',','_')+'_'+device.replace(':','_')+'.json')
    done_audit = device_audit.exists() and json.loads(device_audit.read_text()).get('batch_verified',False)
    for asset, frame in p.groupby('ucid', sort=True):
        if not frame.covered.any():
            continue
        dest = directory/f'asset_{asset}.npz'
        if dest.exists():
            info=json.loads(dest.with_suffix('.json').read_text())
            assert info['fingerprint']==manifest['fingerprint'], 'Cache identity mismatch'
            continue
        lock = directory/f'asset_{asset}.lock'
        try:
            fd = os.open(lock, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
            os.write(fd, json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), time=time.time())).encode()); os.close(fd)
        except FileExistsError:
            continue
        started = time.time()
        try:
            positions = np.searchsorted(grid, frame.date.to_numpy().astype('datetime64[D]'))
            y = np.full(len(grid), np.nan, np.float32)
            x = np.full((len(grid), len(features)), np.nan, np.float32)
            y[positions] = np.where(frame.covered,frame[TARGET],np.nan).astype(np.float32)
            x[positions] = frame[features].to_numpy(np.float32)
            rows = frame.row.to_numpy()
            keep = [(int(pos), int(row)) for pos, row, covered in zip(positions, rows, frame.covered) if covered and inputs(y, x, grid, int(pos)) is not None]
            if not keep:
                with open(dest.with_suffix('.tmp.npz'), 'wb') as f:
                    np.savez_compressed(f, rows=np.array([], np.int64), states=np.empty((0, ext.dim), np.float16))
                dest.with_suffix('.tmp.npz').replace(dest)
                save(dest.with_suffix('.json'),dict(rows=0,seconds=time.time()-started,asset=int(asset),device=device,host=socket.gethostname(),fingerprint=manifest['fingerprint'],status='empty_eligible_history'))
                continue
            if not done_audit or smoke:
                checks = audit(ext, y, x, grid, [s for s, _ in keep])
                record=dict(checks=checks, asset=int(asset), passed=True,batch_verified=True,device=device,
                            extractor_sha256=manifest['extractor_sha256'])
                save(device_audit,record);save(directory/'audit.json',record)
                done_audit = True
            if smoke:
                save(directory/'SMOKE_COMPLETE.json',dict(model=model,target=TARGET,fingerprint=manifest['fingerprint'],device=device,host=socket.gethostname(),seconds=time.time()-started,checks=checks,passed=True))
                print(json.dumps(dict(model=model, manifest=manifest, audit='PASS', seconds=time.time()-started)), flush=True)
                return
            ids, states = [], []
            groups={}
            for step,row in keep:
                arg=inputs(y,x,grid,step);groups.setdefault(arg[3],[]).append((step,row,arg))
            batch_size=int(os.getenv('E206_BATCH','4'))
            for group in groups.values():
                for start in range(0,len(group),batch_size):
                    items=group[start:start+batch_size]
                    state=safe_states(ext,[item[2] for item in items])
                    if state.shape!=(len(items),ext.dim) or not np.isfinite(state).all():raise ValueError('Invalid pre-head representation')
                    stored=state.astype(np.float16)
                    if not np.isfinite(stored).all():raise FloatingPointError('Embedding overflows declared float16 cache precision')
                    ids.extend(item[1] for item in items);states.extend(stored)
                    if start%32==0:
                        save(directory/f'asset_{asset}.progress.json',dict(completed=len(ids),total=len(keep),
                             origin=str(grid[items[-1][0]]),elapsed_seconds=time.time()-started,updated=time.time(),host=socket.gethostname(),device=device))
            with open(dest.with_suffix('.tmp.npz'), 'wb') as f:
                np.savez_compressed(f, rows=np.array(ids, np.int64), states=np.stack(states))
            dest.with_suffix('.tmp.npz').replace(dest)
            save(directory/f'asset_{asset}.json', dict(rows=len(ids), seconds=time.time()-started,
                 asset=int(asset), host=socket.gethostname(), device=device, fingerprint=manifest['fingerprint'], physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES',''), precision='fp32 requested; float16 storage', status='success'))
            print(model, asset, len(ids), round(time.time()-started, 1), 'seconds', flush=True)
        except Exception:
            save(directory/f'asset_{asset}.error.json', dict(traceback=traceback.format_exc(), host=socket.gethostname(), time=time.time()))
            raise
        finally:
            lock.unlink(missing_ok=True)
    print('CLAIM_WORKER_DONE', model, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(a.model, a.device, a.smoke)
