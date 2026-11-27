"""Run E231 scheduler with machine-wide probes outside its pinned CPU affinity."""
from concurrent.futures import ThreadPoolExecutor
from experiments import e231_tabnet_ensembles as T
from experiments import e212_volatility99_tsfm as S
from src.cluster import hosts

def probes():
    with ThreadPoolExecutor(max_workers=len(T.HOSTS)) as pool:
        return hosts.reachable(list(pool.map(lambda name:hosts.probe(name,local_name='__remote_probe__'),T.HOSTS)))

if __name__=='__main__':
    S.save(T.ROOT/'scheduler_override.json',dict(source=__file__,sha256=S.sha(__file__),probe_mode='ssh on every host, including self',reason='Pinned supervisor affinity limits nproc in a local child; SSH measures full machine outside that affinity.'))
    T.probes=probes
    T.supervise()
