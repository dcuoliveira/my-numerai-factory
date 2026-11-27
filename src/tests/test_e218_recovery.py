import pytest
from src.cluster.budget import Host
from experiments import e218_asset_ktz_states as M


def test_probe_uses_full_machine_and_excludes_nonshared_hosts(monkeypatch):
    calls=[]
    def probe(name,timeout,local_name):
        calls.append((name,local_name));return Host(name=name,cores=64,ram_gb=251)
    monkeypatch.setattr(M.hosts,'probe',probe)
    assert len(M.placement_probes())==8
    assert not {'westeros','imeusp'}&{n for n,_ in calls}
    assert all(local=='__remote_probe__' for _,local in calls)


def test_temporary_io_error_does_not_stop_workers(monkeypatch,tmp_path):
    monkeypatch.setattr(M,'ROOT',tmp_path)
    monkeypatch.setattr(M,'prepare',lambda:None)
    monkeypatch.setattr(M,'save',lambda *a:None)
    monkeypatch.setattr(M,'placement_probes',lambda:(_ for _ in ()).throw(OSError('storage unavailable')))
    messages=[];monkeypatch.setattr(M,'log_failure',messages.append)
    monkeypatch.setattr(M.time,'sleep',lambda _:(_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(M.signal,'signal',lambda *a:None)
    stopped=[];monkeypatch.setattr(M.manager,'stop',lambda *a,**kw:stopped.append(a))
    with pytest.raises(KeyboardInterrupt):M.supervise()
    assert len(messages)==1 and 'storage unavailable' in str(messages[0])
    assert stopped==[]
