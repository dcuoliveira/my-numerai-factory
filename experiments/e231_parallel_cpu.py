"""E231 CPU queue: existing asset shards, independent panel candidates and refit."""
import json
import time
from experiments import e231_tabnet_ensembles as T
from experiments.e231_supervisor_cpu import probes
from experiments import e212_volatility99_tsfm as S
from src.cluster import budget,manager

def main():
    T.prepare();owned={};stopped={}
    tasks={f'shard_{j:02d}':('e231_shard_'+f'{j:02d}',T.PY+' -m experiments.e231_tabnet_ensembles fit --shard '+str(j)) for j in range(T.SHARDS)}
    tasks.update({f'case_{j}':('e231_panel_case_'+str(j),T.PY+' -m experiments.e231_parallel_panel case --candidate '+str(j)) for j in range(4)})
    tasks['panel']=('e231_panel_final',T.PY+' -m experiments.e231_parallel_panel final')
    previous=T.ROOT/'parallel_placements.json'
    initial=json.loads(previous.read_text()) if previous.exists() else {k:v for k,v in json.loads((T.ROOT/'placements.json').read_text()).items() if k.startswith('shard_')}
    for name,v in initial.items():
        jobname,cmd=tasks[name];owned[name]=budget.Placement(budget.Job(jobname,cmd,cores=1,ram_gb=5),v['host'],tuple(v['cpus']))
    S.save(T.ROOT/'panel_parallel_manifest.json',dict(sources={str(p):S.sha(p) for p in [__file__,__import__('pathlib').Path(__file__).with_name('e231_parallel_panel.py')]},reason='Independent panel candidates run in parallel; scientific grid, splits, seed, complete40 epochs and fresh selected refit unchanged. Incomplete original serial panel search stopped before any final model publication.',cores_per_panel_candidate=1))
    def complete(name):
        return (T.ROOT/'panel_cases'/name.split('_')[1]/'choice.json').exists() if name.startswith('case_') else (T.ROOT/'workers'/name/'COMPLETE.json').exists()
    while True:
        finished=[name for name in tasks if complete(name)]
        if len(finished)==len(tasks):break
        snapshots=manager.status(T.HOSTS);active={r['unit'].removesuffix('.service') for states in snapshots.values() for r in states if r['state'].startswith('active')}
        for name,v in owned.items():
            if name not in finished and v.unit not in active:
                stopped.setdefault(name,time.time())
                if time.time()-stopped[name]>90:raise RuntimeError('Worker stopped without completion: '+name)
            else:stopped.pop(name,None)
        pending=[budget.Job(jobname,cmd,cores=1,ram_gb=5) for name,(jobname,cmd) in tasks.items() if name not in finished and name not in owned and (name!='panel' or all('case_'+str(j) in finished for j in range(4)))]
        if pending:
            placements,unplaced=budget.distribute(pending,probes())
            for place in placements:
                okay,note=manager.launch(place)
                if not okay:raise RuntimeError(note)
                name=next(n for n,(job,_) in tasks.items() if job==place.job.name);owned[name]=place
        S.save(previous,{name:dict(host=v.host,cpus=v.cpus,unit=v.unit) for name,v in owned.items()})
        S.save(T.ROOT/'status.json',dict(stage='fitting',completed_workers=finished,total_workers=len(tasks),running_workers=[n for n,v in owned.items() if v.unit in active and n not in finished],queued_workers=[name for name in tasks if name not in finished and name not in owned],time=time.time()))
        time.sleep(20)
    from experiments.e231_results import finish
    finish()

if __name__=='__main__':main()
