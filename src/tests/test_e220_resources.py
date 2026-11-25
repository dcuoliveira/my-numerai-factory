from experiments import e220_ja_heads as M
from src.cluster.budget import Host


def test_capacity_uses_remote_probes_and_vision_only(monkeypatch):
    calls=[]
    def probe(name,timeout,local_name):
        calls.append((name,local_name));return Host(name=name,cores=64,ram_gb=251)
    monkeypatch.setattr(M.hosts,'probe',probe)
    found=M.capacity()
    assert len(found)==8 and all(h.cores==64 for h in found)
    assert all(local=='__remote_probe__' for _,local in calls)
    assert not {'westeros','imeusp'}&{name for name,_ in calls}
