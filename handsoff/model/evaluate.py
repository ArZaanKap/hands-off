"""Your phone clip -> trained intent head -> task -> the floating Inspire hand does it in MuJoCo.

Predictor: the nearest clip in the sim gallery of train.validation() (the second half of the validation starts, head
view in the other look, track noise with generator seed 1000 + model seed), the same predictor that chose the epoch
count. Real clips use their measured SAM3 tracks. Only original phone clips are queried (no Cosmos clips).
Execution: the predicted task's motion (one of nine) runs once at a declared fixed start (block at xy 0, yaw 0) with the
dataset's scene, grip and motion; identical predictions share it.

Stages: execute (physics + predictions -> results.json), render (robot MP4s + your-clip/robot sheets, both phones),
report (summary.md/json: per stage, per task type and per clip).
Usage: python -m handsoff.model.evaluate {execute|render|report} --queries DATA --out DIR [--runs RUN ...] [--purpose TEXT]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from handsoff.model.data import FEAT, TrackNoise, expected, load_queries, load_sim, read_inputs
from handsoff.model.head import Head, embed_all
from handsoff.model.tasks import category, eligible, name
from handsoff.sim.primitive import Action, run, placement_success, task_grip
from handsoff.sim.scene import SceneConfig, build
from handsoff.real.cosmos import encode

SIM = ROOT/'data/sim/intent_tasks_train_v3'
DY = {-1: -.15, 0: 0., 1: .15}; ROLL_RAD = {-1: -np.pi/2, 0: 0., 1: np.pi/2}   # category -> sim command
key_of = lambda c: f'move{int(c[0]):+d}_roll{int(c[1]):+d}'  # noqa: E731


def save(path, value):
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=1)); tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gallery(D, rows, folds, head, seed):
    """Exactly train.validation()'s gallery."""
    val = folds['validation']; groups = sorted({rows[i]['start_group'] for i in val})
    ids = [int(i) for i in val if rows[i]['start_group'] not in groups[:len(groups)//2]]
    x = {k: v[val].clone() for k, v in D['x'].items()}
    x['track'] = TrackNoise()(x['track'], torch.Generator().manual_seed(1000+seed))
    zv = embed_all(x, head, 'cpu')
    z = np.zeros((len(rows), *zv.shape[1:]), zv.dtype); z[val] = zv
    views = [int(np.flatnonzero(D['head'][i] & (D['look'][i] != D['look'][i, 0]))[0]) for i in ids]
    return z[ids, views], np.array([rows[i]['cat'] for i in ids]), ids, views


def execute(a):
    torch.set_num_threads(8); a.out.mkdir(parents=True, exist_ok=True); start = time.time()
    manifest = json.loads((SIM/'manifest.json').read_text())
    D, metas, rows, folds = load_sim(SIM)   # refuses data that breaks a user rule
    keys, x, qmetas = load_queries(a.queries)
    original = [i for i, (e, v) in enumerate(keys) if not (qmetas[e]['cameras'][v].get('generated')
                or qmetas[e]['cameras'][v].get('look_name') not in (None, 'original'))]
    keys = [keys[i] for i in original]; x = {k: v[original] for k, v in x.items()}
    truth = np.stack([expected(qmetas[e]['label']) for e, v in keys])
    # Nine robot executions at the declared start, once each.
    cfg = SceneConfig(**manifest['scene']); model = build(cfg); grip = manifest['grip']; executions = {}
    for d in (-1, 0, 1):
        for r in (-1, 0, 1):
            k = key_of((d, r)); directory = a.out/'robot'/k; directory.mkdir(parents=True, exist_ok=True)
            if (directory/'execution.json').exists():
                executions[k] = json.loads((directory/'execution.json').read_text()); continue
            act = Action(dy=DY[d], roll=ROLL_RAD[r])
            gp = task_grip(grip, ROLL_RAD[r])
            t = time.time(); o = run(act, gp, cfg, block_xy=[0., 0.], block_yaw=0., model=model, record_hz=10,
                                     motion=manifest['motion'])
            outcome = np.r_[o.disp[:2], o.roll]
            executions[k] = dict(key=k, task=name((d, r)), action=act.__dict__, fixed_start=dict(xy=[0., 0.], yaw=0.),
                                 outcome=outcome.tolist(), category=category(outcome).tolist(), clean=bool(eligible(outcome)),
                                 gentle_physics_success=bool(placement_success(o, cfg)),
                                 peak_twist_deg=float(np.degrees(o.peak_off_axis_error)), seconds=time.time()-t)
            np.savez_compressed(directory/'traj.npz', **o.log); save(directory/'execution.json', executions[k])
            print('executed', name((d, r)), executions[k]['category'], executions[k]['gentle_physics_success'], flush=True)
    rows_out, models = [], {}
    for run_dir in a.runs:
        r = json.loads((run_dir/'results.json').read_text()); p = r['provenance']
        if not r['complete'] or p.get('select') != 'last':
            raise ValueError(f'{run_dir}: need a complete final-epoch (R19) model')
        head = Head(FEAT); head.load_state_dict(torch.load(run_dir/'head.pt', map_location='cpu')); head.eval()
        models[run_dir.name] = dict(stage=p['stage'], seed=p['seed'], head_sha256=digest(run_dir/'head.pt'),
                                    real_training_keys=len(p['real_training_keys']))
        G, gcat, gids, gviews = gallery(D, rows, folds, head, p['seed'])
        with torch.no_grad():
            q = head(x).numpy()
        for j, (e, v) in enumerate(keys):
            k = int((q[j] @ G.T).argmax()); pred = gcat[k]; ex = executions[key_of(pred)]
            done = np.array(ex['category'])
            rows_out.append(dict(model=run_dir.name, stage=p['stage'], seed=p['seed'], take=int(qmetas[e].get('parent_episode', e)),
                                 episode=int(e), view=int(v), camera=qmetas[e]['cameras'][v].get('kind'),
                                 expected=truth[j].tolist(), predicted=pred.tolist(),
                                 retrieved_episode=int(D['eps'][gids[k]]), retrieved_view=gviews[k],
                                 recognised=bool((pred == truth[j]).all()), executed=done.tolist(),
                                 robot_did_task=bool((done == truth[j]).all() and ex['clean'] and ex['gentle_physics_success'])))
    result = dict(schema='execute_v2', complete=True, purpose=a.purpose, queries=str(a.queries), sim=str(SIM),
                  sim_manifest_sha256=digest(SIM/'manifest.json'), models=models, executions=executions, rows=rows_out,
                  clips=len(keys), seconds=time.time()-start,
                  protocol='Original phone clips only. Predictor = nearest clip in the train.validation() gallery '
                           '(the E41 fair-test predictor). Nine robot motions at a declared fixed start, shared across selections.')
    save(a.out/'results.json', result)
    for stage in ('sim', 'sim_real', 'sim_real_cosmos'):
        rs = [r for r in rows_out if r['stage'] == stage]
        if rs:
            print(f"{stage}: recognised {np.mean([r['recognised'] for r in rs]):.3f}, robot did the task "
                  f"{np.mean([r['robot_did_task'] for r in rs]):.3f} ({len(rs)} clip x model)", flush=True)


def render(a):
    import mujoco
    from handsoff.sim.render import MaskedRenderer, mjv_camera
    result = json.loads((a.out/'results.json').read_text())
    manifest = json.loads((SIM/'manifest.json').read_text())
    model = build(SceneConfig(**manifest['scene'])); data = mujoco.MjData(model); renderer = MaskedRenderer(model, 320)
    specs = [dict(kind='overhead', lookat=[-.05, 0, .1], azimuth=0, elevation=-60, distance=.8),
             dict(kind='side', lookat=[-.05, 0, .12], azimuth=90, elevation=-15, distance=.9)]
    cams = [mjv_camera(c) for c in specs]; robot_frames = {}
    for k, ex in result['executions'].items():
        directory = a.out/'robot'/k; traj = np.load(directory/'traj.npz'); frames = []
        for t, qp in zip(traj['t'], traj['qpos']):
            data.qpos[:] = qp; mujoco.mj_forward(model, data)
            panel = np.hstack([cv2.cvtColor(renderer.render(data, c)[0], cv2.COLOR_RGB2BGR) for c in cams])
            frame = np.zeros((panel.shape[0]+40, panel.shape[1], 3), np.uint8); frame[40:] = panel
            cv2.putText(frame, f"robot: {ex['task']}  t={t:.1f}s  physics ok={ex['gentle_physics_success']}", (6, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            frames.append(frame)
        if not (directory/'robot.mp4').exists():
            encode(directory/'robot.mp4', frames, 10)
        robot_frames[k] = [frames[i] for i in np.linspace(0, len(frames)-1, 4).round().astype(int)]
    renderer.close()
    # One sheet per model stage and phone (seed 0): each original clip, what the model chose, what the robot did.
    for stage, camera in [(s, c) for s in ('sim', 'sim_real', 'sim_real_cosmos') for c in ('head', 'second')]:
        rs = sorted([r for r in result['rows'] if r['stage'] == stage and r['seed'] == 0 and r['camera'] == camera],
                    key=lambda r: r['take'])
        if not rs:
            continue
        lines = []
        for r in rs:
            _, masked = read_inputs(Path(result['queries'])/f"ep{r['episode']:05d}", r['view'])
            raw = read_inputs(Path(result['queries'])/f"ep{r['episode']:05d}", r['view'])[0]
            yours = [cv2.resize(cv2.cvtColor(raw[i], cv2.COLOR_RGB2BGR), (180, 180)) for i in (0, 3, 5, 7)]
            robot = [cv2.resize(f, (300, 180)) for f in robot_frames[key_of(r['predicted'])]]
            label = np.full((180, 330, 3), 255, np.uint8)
            colour = (0, 140, 0) if r['robot_did_task'] else (0, 0, 200)
            for n, text in enumerate([f"{'D' if r['take'] < 100 else 'T'}{r['take']:02d} (your {camera} phone)",
                                      f"you did: {name(r['expected'])}",
                                      f"model chose: {name(r['predicted'])}", 'robot did it: ' + ('YES' if r['robot_did_task'] else 'NO')]):
                cv2.putText(label, text, (6, 35+n*38), cv2.FONT_HERSHEY_SIMPLEX, .5, colour if n == 3 else (0, 0, 0), 1)
            lines.append(np.hstack([*yours, label, *robot]))
        sheet = f'{stage}_seed0_your_clip_to_robot.jpg' if camera == 'head' else f'{stage}_seed0_second_phone_to_robot.jpg'
        cv2.imwrite(str(a.out/sheet), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print('sheet', stage, camera, len(lines), flush=True)


def report(a):
    """Tables from results.json only (no model is run): stage x phone, stage x task type, and every clip at seed 0."""
    result = json.loads((a.out/'results.json').read_text()); rows = result['rows']
    stages = [s for s in ('sim', 'sim_real', 'sim_real_cosmos') if any(r['stage'] == s for r in rows)]
    pct = lambda rs, k: 100*np.mean([r[k] for r in rs]) if rs else float('nan')
    part = lambda rs, i: 100*np.mean([r['predicted'][i] == r['expected'][i] for r in rs]) if rs else float('nan')
    seeds = lambda rs: '/'.join(f"{pct([r for r in rs if r['seed'] == s], 'recognised'):.0f}"
                                for s in sorted({r['seed'] for r in rs}))
    summary = dict(purpose=result['purpose'], queries=result['queries'], clips=result['clips'], stages={})
    md = [f"# {result['purpose'] or 'Own clips -> intent model -> robot'}", '',
          f"Queries `{result['queries']}` ({result['clips']} original phone clips); sim `{result['sim']}`. "
          'Each number is over every clip x the 3 model seeds; "per seed" lists the seeds separately. '
          'Joint = movement and roll both right; robot = the robot then did that task with gentle physics.', '',
          '| Model | Phone | Clips | Joint % (per seed) | Movement % | Roll % | Robot did the task % |', '|---|---|---|---|---|---|---|']
    for st in stages:
        for cam in ('all', 'head', 'second'):
            rs = [r for r in rows if r['stage'] == st and cam in ('all', r['camera'])]
            if not rs:
                continue
            n = len({(r['episode'], r['view']) for r in rs})
            md.append(f"| {st} | {cam} | {n} | {pct(rs, 'recognised'):.0f} ({seeds(rs)}) | {part(rs, 0):.0f} | "
                      f"{part(rs, 1):.0f} | {pct(rs, 'robot_did_task'):.0f} |")
            summary['stages'].setdefault(st, {})[cam] = dict(clips=n, joint=pct(rs, 'recognised'), movement=part(rs, 0),
                                                             roll=part(rs, 1), robot=pct(rs, 'robot_did_task'))
    md += ['', '## Per task type (joint %, both phones, all seeds)', '',
           '| Task you did | Clips | ' + ' | '.join(stages) + ' |', '|---|---|' + '---|'*len(stages)]
    for d in (-1, 0, 1):
        for rr in (-1, 0, 1):
            rs = [r for r in rows if r['expected'] == [d, rr]]
            n = len({(r['episode'], r['view']) for r in rs})
            cells = [f"{pct([r for r in rs if r['stage'] == st], 'recognised'):.0f}" if n else '-' for st in stages]
            md.append(f"| {name((d, rr))} | {n} | " + ' | '.join(cells) + ' |')
            summary.setdefault('tasks', {})[key_of((d, rr))] = dict(
                clips=n, joint={st: pct([r for r in rs if r['stage'] == st], 'recognised') for st in stages} if n else None)
    md += ['', '## Every clip, seed 0 (hits and misses)', '',
           '| Take | Phone | You did | ' + ' | '.join(f'{st} chose' for st in stages) + ' |', '|---|---|---|' + '---|'*len(stages)]
    for (take, view, cam), exp in sorted({(r['take'], r['view'], r['camera']): tuple(r['expected']) for r in rows}.items()):
        cells = []
        for st in stages:
            r = next((r for r in rows if r['stage'] == st and r['seed'] == 0 and r['take'] == take and r['view'] == view), None)
            cells.append('-' if r is None else ('HIT ' if r['recognised'] else 'miss: ') + name(r['predicted']))
        md.append(f"| {'D' if take < 100 else 'T'}{take:02d} | {cam} | {name(exp)} | " + ' | '.join(cells) + ' |')
    sheets = sorted(p.name for p in a.out.glob('*_seed0_*.jpg'))
    md += ['', '## Sheets and robot videos', '', *[f'- {a.out.resolve()/s}' for s in sheets],
           *[f"- {(a.out/'robot'/k/'robot.mp4').resolve()} ({ex['task']})" for k, ex in result['executions'].items()]]
    (a.out/'summary.md').write_text('\n'.join(md)+'\n'); save(a.out/'summary.json', summary)
    print('\n'.join(md[:8+4*len(stages)]), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['execute', 'render', 'report'])
    ap.add_argument('--queries', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--runs', type=Path, nargs='+', default=sorted((ROOT/'results/final_v3').glob('*_seed*')))
    ap.add_argument('--purpose', default='')
    ap.add_argument('--sim', type=Path, default=SIM, help='sim dataset whose validation gallery and robot settings are used')
    a = ap.parse_args()
    SIM = a.sim.resolve()
    {'execute': execute, 'render': render, 'report': report}[a.stage](a)
