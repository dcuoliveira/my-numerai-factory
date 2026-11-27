"""E230 concatenated134 feature dataset and four fresh supervised learners."""
import argparse
from experiments import e230_dataset as D

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','fit','finish']);ap.add_argument('arm',nargs='?',choices=D.ARMS);args=ap.parse_args()
    if args.action=='prepare':D.prepare()
    elif args.action=='fit':
        assert args.arm
        from experiments.e230_models import fit
        fit(args.arm)
    else:
        from experiments.e230_results import finish
        finish()

if __name__=='__main__':main()
