"""Choose the fixed epoch count (rule R19) by leave-one-take-out over the recorded real takes (user, 12 Oct).

For each training stage and each recorded take: train on the other takes (originals, plus their approved Cosmos clips in
the Cosmos stage) and score the held-out take's two original phone clips at every epoch (train.validation()'s
real_direct: nearest sim validation clip's task). The epoch count from CHOICES with the best mean joint accuracy over
all held-out clips is used for that stage's final model on all takes; ties go to fewer epochs. The test
recordings are never involved.
Usage: python -m handsoff.model.choose_epochs [--stages sim sim_real sim_real_cosmos] [--takes 1 2 ...] [--device cuda]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT/'data/sim/intent_tasks_train_v3'
REAL = ROOT/'data/real/demos3_approved_rgb_v3'
SPLIT = ROOT/'config/real_split.json'
OUT = ROOT/'results/epoch_choice'
CHOICES = (25, 50, 100, 200)


def run(stage, take, takes, device, seed):
    out = OUT/'runs'/stage/f'D{take:02d}'
    done = out/'results.json'
    if done.exists() and json.loads(done.read_text())['complete']:
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    split = out/'split.json'
    split.write_text(json.dumps(dict(folds=dict(train=[t for t in takes if t != take], validation=[take], test=[]),
                                     note=f'leave-one-take-out: D{take:02d} held out'), indent=1))
    cmd = [sys.executable, '-m', 'handsoff.model.train', str(SIM), '--out', str(out), '--stage', stage,
           '--real-data', str(REAL), '--real-split', str(split), '--epochs', str(max(CHOICES)), '--seed', str(seed),
           '--device', device]
    with (out/'train.log').open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return json.loads(done.read_text())


def main():
    global SIM, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', nargs='+', default=['sim', 'sim_real', 'sim_real_cosmos'])
    ap.add_argument('--takes', type=int, nargs='*')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sim', type=Path, default=SIM); ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    SIM, OUT = a.sim.resolve(), a.out.resolve(); OUT.mkdir(parents=True, exist_ok=True)
    takes = json.loads(SPLIT.read_text())['folds']['train']
    summary = dict(choices=CHOICES, takes=takes, seed=a.seed, split=str(SPLIT.relative_to(ROOT)), stages={})
    for stage in a.stages:
        per_take = {}
        for take in a.takes or takes:
            t0 = time.time(); r = run(stage, take, takes, a.device, a.seed)
            scores = {e: r['history'][e-1]['validation']['real_direct'] for e in CHOICES}
            per_take[take] = scores
            print(f'{stage} D{take:02d} ({time.time()-t0:.0f}s): ' + ' '.join(f'{e}ep {s["joint"]:.2f}' for e, s in scores.items()), flush=True)
        table = {e: {k: float(np.mean([per_take[t][e][k] for t in per_take])) for k in ('translation', 'roll', 'joint')} for e in CHOICES}
        best = max(CHOICES, key=lambda e: (round(table[e]['joint'], 6), -e))
        summary['stages'][stage] = dict(per_take={f'D{t:02d}': {str(e): s for e, s in v.items()} for t, v in per_take.items()},
                                        mean=table, chosen_epochs=best, held_out_clips=2*len(per_take))
        print(f'{stage}: chosen {best} epochs; mean joint ' + ' '.join(f'{e}:{table[e]["joint"]:.3f}' for e in CHOICES), flush=True)
    if not a.takes:
        (OUT/'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
