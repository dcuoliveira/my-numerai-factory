"""Run one registered E226 head; atomically publish the identical shared mask."""
import argparse
import os
from pathlib import Path
from experiments import e226_volatility_cs_ridge as R


def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['CS99','TS_CS198']);ap.add_argument('horizon',type=int,choices=[20,60]);a=ap.parse_args()
    original_save=R.np.save
    shared=R.ROOT/'cache/common_rows.npy'
    def atomic_common(path,values,*args,**kwargs):
        if Path(path)!=shared:return original_save(path,values,*args,**kwargs)
        temporary=shared.with_name(f'common_rows.{os.getpid()}.npy')
        original_save(temporary,values,*args,**kwargs);temporary.replace(shared)
    R.np.save=atomic_common
    try:R.fit(a.arm,a.horizon)
    finally:R.np.save=original_save

if __name__=='__main__':main()
