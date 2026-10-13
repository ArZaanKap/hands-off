"""Search the human-like roll-left grasp (rule R20; CPU, cross-entropy method over GraspParams + bank_extra).

The user's idea as the warm start: the roll-right side grasp rotated 180 deg about the block's forward axis, so the
wrist stays on the person's side. The search may also tilt the palm onto the block's upper edge (bank_extra) with the
fingers over the top, like the user's own roll-left grip. Each candidate is scored on four roll-left cases by twist,
position, lift, table penetration, release height and touchdown speed (the full gentle objective).
The roll-right and upright grasps in config/grip_v2.json came from earlier searches of the same kind.
The chosen candidate went into config/grip_v3.json as left_params; sim/dataset.py's physics stage then checks every
start x task with it.
Usage: python -m handsoff.sim.grip --warm-start config/grip_v2.json --out results/grip_search.json [--iters 8]
"""
import argparse
import json
import signal
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from handsoff.sim.primitive import Action, GraspParams, placement_success, run, success
from handsoff.sim.scene import SceneConfig, build

SIZE = (.062, .038, .086)
MOTION = 'clearance_side_human'
# roll-left cases: three start yaws at the centre, and move left + roll from an offset start
CASES = [(Action(roll=np.deg2rad(90)), (0., 0.), np.deg2rad(y)) for y in (-30, 0, 30)]
CASES += [(Action(dx=0, dy=.2, roll=np.deg2rad(90)), (.02, -.02), .17)]
#               off_x  off_y  off_z  yaw  tilt thumb_yaw thumb_flex finger_flex pre_open bank_extra
LO = np.array([-.06, -.06, -.08, -.4, -.6, 0., 0., 0., 0., -1.2])
HI = np.array([.04, .02, -.02, .4, .6, 1., 1., 1., .6, 1.2])
TIMEOUT_S = 90   # a candidate whose simulation does not finish counts as a failure
_state = {}


def init_worker():
    _state['cfg'] = SceneConfig(block_size=SIZE, collision_geometry='full')
    _state['model'] = build(_state['cfg'])


def _timeout(*_):
    raise TimeoutError


def score(params):
    signal.signal(signal.SIGALRM, _timeout); signal.alarm(TIMEOUT_S)
    try:
        return _score(params)
    except TimeoutError:
        n = len(CASES)
        return dict(score=-100., successes=[False]*n, peak_deg=[180.]*n, position_mm=[1e3]*n, lift_mm=[0.]*n,
                    penetration_over_1mm=[0.]*n, timed_out=True)
    finally:
        signal.alarm(0)


def _score(params):
    gp, cfg = GraspParams.from_array(params), _state['cfg']
    outs = [run(a, gp, cfg, block_xy=xy, block_yaw=yaw, model=_state['model'], motion=MOTION)
            for a, xy, yaw in CASES]
    peaks = np.array([o.peak_off_axis_error for o in outs]) / np.deg2rad(10)
    positions = np.array([np.linalg.norm(o.disp[:2] - o.action.as_array()[:2]) for o in outs]) / .03
    angles = np.array([o.orientation_error for o in outs]) / np.deg2rad(20)
    lifts = np.array([o.max_lift for o in outs])
    oks = [bool(success(o, cfg)) for o in outs]
    gentle = [placement_success(o, cfg) for o in outs]
    gaps = np.array([o.commanded_release_gap for o in outs])*1000
    penetrations = [max(0., -o.min_table_clearance*1000 - 1) if o.clearance_checked else 0. for o in outs]
    value = (2 * np.mean(oks) - np.mean(peaks) - .5 * np.max(peaks)
             - .3 * np.minimum(positions, 10).mean() - .2 * np.minimum(angles, 10).mean() - np.mean(lifts < .049))
    value -= np.minimum(penetrations, 20).mean()
    value += 4 * np.minimum(lifts/.15, 1).mean()   # prevent an untouched block winning exploration
    value += 3*np.mean(oks) + 4*np.minimum(lifts/.15, 1).mean() + 3*np.mean(gentle)
    value -= 3*np.minimum(np.maximum(gaps, 0)/50, 5).mean()
    value -= np.minimum(np.array([o.touchdown_speed for o in outs])/.3, 5).mean()
    return dict(score=float(value), successes=oks, peak_deg=np.rad2deg(peaks*np.deg2rad(10)).tolist(),
                position_mm=(positions*.03*1000).tolist(), lift_mm=(lifts*1000).tolist(), penetration_over_1mm=penetrations,
                gentle_successes=gentle, commanded_release_gap_mm=gaps.tolist(), release_gap_mm=[o.release_gap*1000 for o in outs],
                touchdown_speed_m_s=[o.touchdown_speed if np.isfinite(o.touchdown_speed) else None for o in outs])


def save(path, result):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(result, indent=2)); tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--warm-start', type=Path, required=True, help='grip file whose roll-right grasp seeds the search')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--iters', type=int, default=8); ap.add_argument('--pop', type=int, default=48)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed); t0 = time.time(); lo, hi = LO, HI
    result = dict(schema='rectangle_grip_search_v1', complete=False, seed=a.seed, population=a.pop, iterations=a.iters,
                  workers=a.workers, block_size=SIZE, names=GraspParams.NAMES, bounds=dict(lo=lo.tolist(), hi=hi.tolist()),
                  collision_geometry='full', motion=MOTION, gentle=True,
                  cases=[dict(action=x.__dict__, xy=xy, yaw=yaw) for x, xy, yaw in CASES], home=True, roll_frame='block',
                  history=[], candidates=[])
    if a.out.exists():
        old = json.loads(a.out.read_text())
        if old['seed'] != a.seed or old['motion'] != MOTION:
            raise ValueError('Resume config differs')
        result = dict(old, iterations=a.iters); rng.bit_generator.state = old['rng_state']
    with Pool(a.workers, initializer=init_worker) as pool:
        for stage in range(a.iters+1):
            name = 'init' if stage == 0 else f'cem{stage}'
            if any(h['stage'] == name for h in result['history']):
                continue
            if not result['candidates']:
                # the user's warm start: roll-right grasp rotated about the forward axis, hand height kept
                right = np.r_[json.loads(a.warm_start.read_text())['params'], 0.]
                mirrored = right.copy(); mirrored[1] = -right[1]; mirrored[3] = -right[3]
                seeds = [np.clip(mirrored, lo, hi), np.clip(right, lo, hi)]
                result['warm_start'] = dict(file=str(a.warm_start), roll_right=right.tolist(), mirrored=mirrored.tolist())
                # half local proposals around the seeds, half broad exploration
                local = [np.clip(seeds[i % len(seeds)] + rng.normal(0, .10, len(lo))*(hi-lo), lo, hi) for i in range(a.pop)]
                X = [*seeds, *local, *(lo+rng.random((a.pop, len(lo)))*(hi-lo))]
            else:
                elite = sorted(result['candidates'], key=lambda c: c['score'], reverse=True)[:12]
                E = np.asarray([c['params'] for c in elite]); sd = E.std(0) + .01*(hi-lo)
                # half proposals around the best, to avoid averaging distinct grips
                X = np.clip(np.vstack([E.mean(0)+sd*rng.normal(size=(a.pop//2, len(lo))),
                                       E[0]+sd*rng.normal(size=(a.pop-a.pop//2, len(lo)))]), lo, hi)
            scores = pool.map(score, X)
            result['candidates'].extend([dict(params=np.asarray(x).tolist(), **s) for x, s in zip(X, scores)])
            best = max(result['candidates'], key=lambda c: c['score'])
            result['history'].append(dict(stage=name, best=best['score'], successes=sum(best['successes']),
                                          peak_max_deg=max(best['peak_deg']), evaluations=len(result['candidates'])))
            result['seconds'] = result.get('seconds', 0) + time.time()-t0; t0 = time.time()
            result['rng_state'] = rng.bit_generator.state
            save(a.out, result); print(result['history'][-1], flush=True)
    result['complete'] = True; result['best'] = max(result['candidates'], key=lambda c: c['score'])
    save(a.out, result); print('best', result['best'], flush=True)


if __name__ == '__main__':
    main()
