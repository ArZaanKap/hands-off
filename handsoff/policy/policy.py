"""The policy: intent vector z + block start -> the 14 hand waypoints, executed open-loop in MuJoCo physics.

Choose, then move (Behavior Transformer style, Shafiullah et al. 2022): k-means (K=16, seed 0) on the teacher plans'
set-down waypoint (position / 5 cm, 6-D rotation) gives 16 motion bins; each bin's centre is its members' mean plan.
One MLP sees [z, start] and picks the bin; a second MLP sees only the start and gives each bin's offset, so an unusual
z can change which motion runs but never distort it. Act = centre of the most likely bin + its offset.
Training pairs each teacher plan (2000 starts x 9 tasks) with a random z of the same task from a training-start sim
clip (any view), drawn afresh every batch; fixed schedule (6000 steps), final weights.

Tests (success = right task + clean + gentle placement, measured in physics):
  sim    every clip of the held-out validation starts (all views): z -> robot at that clip's own start
  arith  the composition queries z(a) - z(b) + z(c) (normalised) of the validation starts -> robot at c's start
  real   original phone clips (both phones) of a real dataset: z -> robot at the fixed start (xy 0, yaw 0)

Stages: plans | embed | train | eval      (.venv-spider; CPU; physics on 8 workers)
Usage:
  python -m handsoff.policy.policy plans --sim DATA --plans FILE      (also writes scene.json and durations.npz beside it)
  python -m handsoff.policy.policy embed --sim DATA --head RUN --z FILE [--real DIR]
  python -m handsoff.policy.policy embed --real-only --head RUN --z FILE --real DIR      (no sim dataset needed)
  python -m handsoff.policy.policy train --z FILE --plans FILE --out DIR [--seed S]
  python -m handsoff.policy.policy eval  --sim DATA --z FILE --plans FILE --out DIR [--policy FILE] [--sets sim arith real]
--sim also accepts the scene.json file and --plans the durations.npz file (eval of real clips without the sim dataset).
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from handsoff.model.data import TrackNoise, expected, load_queries, load_sim
from handsoff.model.head import Head, embed_all
from handsoff.model.tasks import category, composition_queries, name
from handsoff.policy import plans as pl

D_Y = pl.N_WP * pl.D_WP; K_BINS = 16; STEPS = 6000; SET_DOWN = 8
FIXED = np.zeros(3)


def start_feats(s):
    s = np.atleast_2d(s)
    return np.c_[s[:, :2] / .03, np.cos(s[:, 2]), np.sin(s[:, 2]) / np.sin(.25)].astype(np.float32)


def mlp(d_in, d_out, d=256):
    return torch.nn.Sequential(torch.nn.Linear(d_in, d), torch.nn.GELU(), torch.nn.Linear(d, d), torch.nn.GELU(),
                               torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, d_out))


class Policy(torch.nn.Module):
    """Condition c = [z (64), start features (4)] -> normalised waypoints (294)."""

    def __init__(self, centres, d_z=64):
        super().__init__()
        self.pick = mlp(d_z + 4, K_BINS)             # z + start -> which motion bin
        self.offset = mlp(4, K_BINS * D_Y)           # start only -> each bin's offset
        self.register_buffer('centres', torch.as_tensor(centres, dtype=torch.float32))

    def heads(self, c):
        return self.pick(c), self.offset(c[:, -4:]).reshape(len(c), K_BINS, D_Y)

    def loss(self, c, y, b):
        logits, off = self.heads(c)
        return torch.nn.functional.cross_entropy(logits, b) + \
            torch.nn.functional.mse_loss(off[torch.arange(len(c)), b], y - self.centres[b])

    @torch.no_grad()
    def act(self, c):
        logits, off = self.heads(c); b = logits.argmax(1)
        return self.centres[b] + off[torch.arange(len(c)), b]


def make_plans(a):
    t0 = time.time()
    with Pool(8, initializer=pl.init_worker, initargs=(a.sim,)) as pool:
        np.savez(a.plans, **pl.build_set(a.n_starts, a.plan_seed, pool))
    np.savez(a.plans.parent/'durations.npz', durations=np.load(a.plans)['durations'])
    man = json.loads((a.sim/'manifest.json').read_text())
    (a.plans.parent/'scene.json').write_text(json.dumps({k: man[k] for k in ('scene', 'grip', 'motion')}, indent=1))
    print(f'plans: {a.n_starts} starts x 9 tasks in {time.time() - t0:.0f}s -> {a.plans}')


def embed(a):
    """z of every sim clip (one fixed draw of track noise, generator 1000 + head seed), the composition queries of the
    validation starts, and (with --real) the original phone clips of a real dataset; --real-only skips the sim parts."""
    head = Head(); head.load_state_dict(torch.load(a.head/'head.pt', map_location='cpu')); head.eval()
    out = embed_real(a, head) if a.real else {}
    if not a.real_only:
        out.update(embed_sim(a, head))
    a.z.parent.mkdir(parents=True, exist_ok=True); np.savez(a.z, **out)
    print(f"embedded {len(out.get('z', []))} sim clips, {len(out.get('za', []))} arithmetic queries, "
          f"{len(out.get('zr', []))} real clips -> {a.z}")


def embed_sim(a, head):
    D, metas, rows, folds = load_sim(a.sim)
    seed = json.loads((a.head/'results.json').read_text())['provenance']['seed']
    x = {k: v.clone() for k, v in D['x'].items()}
    x['track'] = TrackNoise()(x['track'], torch.Generator().manual_seed(1000 + seed))
    z = embed_all(x, head, 'cpu')
    va = folds['validation']; groups = sorted({rows[i]['start_group'] for i in va})
    source = [int(i) for i in va if rows[i]['start_group'] in groups[:len(groups)//2]]
    qs = composition_queries(rows, source)
    za = np.stack([z[q['a'], 0] - z[q['b'], 0] + z[q['c'], 0] for q in qs]); za /= np.linalg.norm(za, axis=1, keepdims=True)
    starts = np.array([[*m['start']['xy'], m['start']['yaw']] for m in metas])
    return dict(z=z, cats=np.array([r['cat'] for r in rows]), split=np.array([m['split'] for m in metas]), starts=starts,
                eps=D['eps'], za=za, a_target=np.array([q['target'] for q in qs]), a_family=np.array([q['family'] for q in qs]),
                a_start=starts[[q['c'] for q in qs]], a_ids=np.array([[q['a'], q['b'], q['c']] for q in qs]))


def embed_real(a, head):
    keys, xr, rmetas = load_queries(a.real)
    orig = [i for i, (e, v) in enumerate(keys) if not (rmetas[e]['cameras'][v].get('generated')
            or rmetas[e]['cameras'][v].get('look_name') not in (None, 'original'))]
    with torch.no_grad():
        zr = head({k: v[orig] for k, v in xr.items()}).numpy()
    rkeys = [keys[i] for i in orig]
    return dict(zr=zr, rtruth=np.stack([expected(rmetas[e]['label']) for e, _ in rkeys]), rkeys=np.array(rkeys),
                rtake=np.array([rmetas[e].get('parent_episode', e) for e, _ in rkeys]),
                rview=np.array([rmetas[e]['cameras'][v]['kind'] for e, v in rkeys]))


def train(a):
    from sklearn.cluster import KMeans
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    p = np.load(a.plans); d = np.load(a.z); tr = d['split'] == 'train'
    pool = {t: d['z'][tr & (d['cats'] == t).all(1)].reshape(-1, 64) for t in pl.TASKS}
    mu, sd = p['y'].mean(0), p['y'].std(0) + 1e-6
    rows = np.arange(len(p['y'])); y = (p['y'] - mu) / sd; sf = start_feats(p['s'])
    w = p['y'].reshape(len(p['y']), pl.N_WP, pl.D_WP)
    km = KMeans(K_BINS, n_init=4, random_state=0).fit(np.c_[w[:, SET_DOWN, :3] / .05, w[:, SET_DOWN, 3:9]])
    bins = km.labels_.astype(np.int64); centres = np.stack([y[bins == k].mean(0) for k in range(K_BINS)])
    print('bins: sizes', np.bincount(bins, minlength=K_BINS).tolist(), 'tasks per bin',
          [len(set(p['t'][bins == k])) for k in range(K_BINS)])
    model = Policy(centres)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); t0 = time.time()
    for step in range(STEPS):
        i = rng.choice(rows, 512)
        zb = np.stack([pool[pl.TASKS[t]][rng.integers(len(pool[pl.TASKS[t]]))] for t in p['t'][i]])
        loss = model.loss(torch.tensor(np.c_[zb, sf[i]], dtype=torch.float32), torch.tensor(y[i], dtype=torch.float32),
                          torch.tensor(bins[i]))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == STEPS - 1:
            print(f'step {step} loss {loss.item():.5f}', flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(net=model.state_dict(), mu=mu, sd=sd, steps=STEPS, seed=a.seed, z=str(a.z)), a.out/'policy.pt')
    print(f'trained in {time.time() - t0:.0f}s -> {a.out/"policy.pt"}')


def evaluate(a):
    d = np.load(a.z); ck = torch.load(a.policy or a.out/'policy.pt', weights_only=False)
    durations = np.load(a.plans)['durations']
    model = Policy(ck['net']['centres']); model.load_state_dict(ck['net']); model.eval()

    def act(z, s):
        return model.act(torch.tensor(np.c_[z, start_feats(s)], dtype=torch.float32)).numpy() * ck['sd'] + ck['mu']
    sets = {}
    if 'z' in d:
        va = np.flatnonzero(d['split'] == 'validation'); V = d['z'].shape[1]
        sets = {'sim': (d['z'][va].reshape(-1, 64), np.repeat(d['starts'][va], V, 0), np.repeat(d['cats'][va], V, 0)),
                'arith': (d['za'], d['a_start'], d['a_target'])}
    if 'zr' in d:
        sets['real'] = (d['zr'], np.repeat(FIXED[None], len(d['zr']), 0), d['rtruth'])
    report = dict(protocol=__doc__, policy_seed=ck['seed'], steps=ck['steps'], z=str(a.z))
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(8, initializer=pl.init_worker, initargs=(a.sim,)) as workers:
        for k, (z, s, truth) in ((k, v) for k, v in sets.items() if k in a.sets):
            t0 = time.time(); y = act(z, s)
            runs = workers.map(pl.execute, [(tuple(map(int, t)), si, yi, durations) for t, si, yi in zip(truth, s, y)], chunksize=4)
            (a.out/f'runs_{k}.json').write_text(json.dumps(runs))
            ok = np.array([r['success'] for r in runs])
            report[k] = dict(n=len(runs), success=float(ok.mean()), right_task=float(np.mean([r['right_task'] for r in runs])),
                             gentle=float(np.mean([r['gentle'] for r in runs])),
                             per_task={name(t): float(ok[(truth == t).all(1)].mean()) for t in pl.TASKS if (truth == t).all(1).any()},
                             seconds=time.time() - t0)
            if k == 'arith':
                report[k]['per_family'] = {f: float(ok[d['a_family'] == f].mean()) for f in np.unique(d['a_family'])}
            if k == 'real' and 'rview' in d:    # one line per clip
                for r, take, view in zip(runs, d['rtake'], d['rview']):
                    r.update(take=int(take), phone=str(view))
                    c = category(np.array(r['outcome'])); did = name(c) if abs(c[1]) <= 1 else 'over-turned'
                    print(f"  take {take} {view:6s}  demonstrated: {name(r['task']):22s} robot did: {did:22s} "
                          f"{'success' if r['success'] else 'fail'}", flush=True)
                (a.out/f'runs_{k}.json').write_text(json.dumps(runs))
            print(k, json.dumps(report[k]), flush=True)
    (a.out/'report.json').write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['plans', 'embed', 'train', 'eval'])
    ap.add_argument('--sim', type=Path, default=Path('data/sim/intent_tasks_train_v3'))
    ap.add_argument('--plans', type=Path, default=Path('results/policy/plans.npz'))
    ap.add_argument('--n-starts', type=int, default=2000); ap.add_argument('--plan-seed', type=int, default=100)
    ap.add_argument('--head', type=Path, help='embed: a trained head run (head.pt + results.json)')
    ap.add_argument('--real', type=Path, help='embed: a prepared real dataset (e.g. the test takes)')
    ap.add_argument('--real-only', action='store_true', help='embed: only the real clips (no sim dataset needed)')
    ap.add_argument('--z', type=Path, help='embed output / train and eval input')
    ap.add_argument('--out', type=Path, help='train/eval: output directory')
    ap.add_argument('--policy', type=Path, help='eval: policy.pt (default OUT/policy.pt)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sets', nargs='+', choices=['sim', 'arith', 'real'], default=['sim', 'arith', 'real'])
    a = ap.parse_args(); torch.set_num_threads(8)
    if a.stage == 'plans':
        a.plans.parent.mkdir(parents=True, exist_ok=True)
    dict(plans=make_plans, embed=embed, train=train, eval=evaluate)[a.stage](a)


if __name__ == '__main__':
    main()
