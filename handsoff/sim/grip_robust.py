"""Search one grasp under randomised block mass and friction (CPU, cross-entropy method over the 9 GraspParams).

The grasps of config/grip_v3.json were searched at the nominal block (0.12 kg, friction 1.0); the roll-right one drops
a light or slippery block. Here every generation draws 16 cases (block mass and friction from the training levels below,
movement, start) and scores each candidate on all of them with the gentle objective of handsoff.sim.grip. Each
generation's best and the last elite are then compared on a fixed validation set (every mass x friction level, 2 starts,
movements cycled) and the most successful is kept as 'final'. The training levels differ from the levels used to test
robustness (scripts/robustness.sh).
The roll-right and upright grasps of config/grip_v4.json come from this search (seed 0, warm start config/grip_v3.json);
its roll-left grasp is grip_v3's, which already holds under the same changes.
Usage: python -m handsoff.sim.grip_robust --grasp right --out results/grip_robust_right.json   (about 17 min on 8 cores)
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from handsoff.model.tasks import category, eligible
from handsoff.policy import plans
from handsoff.sim import primitive as P
from handsoff.sim.scene import build

ROOT = Path(__file__).resolve().parents[2]
TRAIN_MASS = [0.045, 0.065, 0.10, 0.15, 0.21, 0.28]     # kg (nominal 0.12)
TRAIN_FRIC = [0.45, 0.55, 0.65, 0.85, 1.1, 1.4]         # (nominal 1.0)
#               off_x  off_y  off_z  yaw  tilt thumb_yaw thumb_flex finger_flex pre_open
LO = np.array([-.06, -.06, -.08, -.4, -.6, 0., 0., 0., 0.])
HI = np.array([.04, .06, -.02, .4, .6, 1., 1., 1., .6])
KEY = dict(right='params', upright='upright_params')
ROLL = dict(right=-1, upright=0)
_models = {}


def init_worker(scene):
    os.nice(10); plans.configure(scene)


def model(mass, friction):
    key = (round(mass, 5), round(friction, 5))
    if key not in _models:
        if len(_models) > 4:
            _models.clear()
        _models[key] = build(replace(plans.CFG, block_mass=mass, friction=friction))
    return _models[key]


def case(params, c):
    """One scripted run of grasp `params` on case c; success as handsoff.policy.plans.execute."""
    gp = P.GraspParams.from_array(params); cfg = replace(plans.CFG, block_mass=c['mass'], friction=c['friction'])
    act = P.Action(dy=plans.DY[c['move']], roll=plans.ROLL[c['roll']])
    o = P.run(act, gp, cfg, block_xy=c['start'][:2], block_yaw=c['start'][2], model=model(c['mass'], c['friction']))
    out = np.r_[o.disp[:2], o.roll]
    ok = bool((category(out) == np.array([c['move'], c['roll']])).all()) and bool(eligible(out)) and \
        bool(P.placement_success(o, cfg))
    return o, ok


def score(args):
    params, cases = args
    outs = [case(params, c) for c in cases]
    o_ = [o for o, _ in outs]
    peaks = np.array([o.peak_off_axis_error for o in o_]) / np.deg2rad(10)
    positions = np.array([np.linalg.norm(o.disp[:2] - o.action.as_array()[:2]) for o in o_]) / .03
    angles = np.array([o.orientation_error for o in o_]) / np.deg2rad(20)
    lifts = np.array([o.max_lift for o in o_])
    oks = np.array([ok for _, ok in outs])
    gaps = np.array([o.commanded_release_gap for o in o_]) * 1000
    pen = np.array([max(0., -o.min_table_clearance*1000 - 1) if o.clearance_checked else 0. for o in o_])
    touch = np.array([o.touchdown_speed if np.isfinite(o.touchdown_speed) else 1.5 for o in o_])
    value = (8*oks.mean() - peaks.mean() - .5*peaks.max() - .3*np.minimum(positions, 10).mean() - .2*np.minimum(angles, 10).mean()
             - (lifts < .049).mean() - np.minimum(pen, 20).mean() + 8*np.minimum(lifts/.15, 1).mean()
             - 3*np.minimum(np.maximum(gaps, 0)/50, 5).mean() - np.minimum(touch/.3, 5).mean())
    return dict(score=float(value), success=float(oks.mean()), lift_ok=float((lifts >= .049).mean()))


def draw(rng, n, roll):
    return [dict(mass=float(rng.choice(TRAIN_MASS)), friction=float(rng.choice(TRAIN_FRIC)), move=int(rng.choice([-1, 0, 1])),
                 roll=roll, start=[float(rng.uniform(-.03, .03)), float(rng.uniform(-.03, .03)), float(rng.uniform(-.25, .25))])
            for _ in range(n)]


def validate(job):
    return case(np.array(job['params']), job['case'])[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grasp', choices=list(KEY), required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--warm-start', type=Path, default=ROOT/'config/grip_v3.json')
    ap.add_argument('--scene', type=Path, default=ROOT/'weights/scene.json', help='scene and motion (its grip is not used)')
    ap.add_argument('--iters', type=int, default=14); ap.add_argument('--pop', type=int, default=48)
    ap.add_argument('--cases', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); a.out.parent.mkdir(parents=True, exist_ok=True)
    plans.configure(a.scene)
    base = np.array(json.loads(a.warm_start.read_text())[KEY[a.grasp]][:9]); roll = ROLL[a.grasp]
    hist, elite, t0 = [], [], time.time()
    sd = .08*(HI-LO)
    with Pool(8, initializer=init_worker, initargs=(a.scene,)) as pool:
        for it in range(a.iters+1):
            cases = draw(rng, a.cases, roll)      # common random cases within a generation
            if it == 0:
                X = np.vstack([base, np.clip(base + rng.normal(0, 1, (a.pop-1, 9))*sd, LO, HI)])
            else:   # elites are re-scored on the new cases, so a lucky draw does not persist
                E = np.array([e['params'] for e in elite[:12]])
                mean = E.mean(0); sd = E.std(0) + .01*(HI-LO)
                X = np.vstack([E[0], np.clip(np.vstack([mean + sd*rng.normal(size=(a.pop//2, 9)),
                                                        E[0] + sd*rng.normal(size=(a.pop//2-1, 9))]), LO, HI)])
            S = pool.map(score, [(x, cases) for x in X], chunksize=1)
            elite = sorted([dict(params=x.tolist(), **s) for x, s in zip(X, S)], key=lambda e: e['score'], reverse=True)
            hist.append(dict(it=it, params=elite[0]['params'], best=elite[0]['score'], best_success=elite[0]['success'],
                             mean_success=float(np.mean([s['success'] for s in S])), seconds=time.time()-t0))
            print(json.dumps(hist[-1]), flush=True)
        # validation: every training level x 2 starts, movements cycled; the warm start is scored for reference
        cands = [h['params'] for h in hist] + [e['params'] for e in elite[:12]] + [base.tolist()]
        cases = [dict(mass=m, friction=f, move=[-1, 0, 1][(i + k) % 3], roll=roll, start=s.tolist())
                 for i, m in enumerate(TRAIN_MASS) for k, f in enumerate(TRAIN_FRIC) for s in plans.starts(2, 21)]
        jobs = [dict(c=ci, params=p, case=c) for ci, p in enumerate(cands) for c in cases]
        ok = pool.map(validate, jobs, chunksize=4)
    val = [float(np.mean([o for j, o in zip(jobs, ok) if j['c'] == ci])) for ci in range(len(cands))]
    best = int(np.argmax(val[:-1]))
    a.out.write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()}, base=base.tolist(), history=hist,
                                     validation=dict(n_cases=len(cases), success=val, base_success=val[-1]),
                                     final=dict(params=cands[best], validation_success=val[best]))))
    print(f'validation success: warm start {val[-1]:.2f}, chosen {val[best]:.2f} -> {a.out}')


if __name__ == '__main__':
    main()
