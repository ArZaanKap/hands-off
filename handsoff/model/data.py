"""Loading model inputs: sim and real clips as frozen Cosmos-Embed1 features + 2D block tracks, task labels, folds.

Features per clip (view): `frames` (8 x 1408 frame CLS tokens), `proj` (768 clip vector), `track` (8 x 5 block box
u, v, w, h, visible; sim tracks are exact, real ones come from SAM3, so sim tracks get SAM3-like noise in training).
"""
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from handsoff.model.tasks import category, eligible
from handsoff.real.review_gate import REVIEW_PATH, read_reviews, require_dataset_approval, view_approved

ROOT = Path(__file__).resolve().parents[2]
FEAT = ('cls', 'track')
NOISE = ROOT/'config/track_noise_real.npz'        # SAM3 tracks of sim clips ...
NOISE_TRUTH = ROOT/'config/track_noise_sim.npz'   # ... and the exact tracks of the same clips


def load_features(dataset):
    """Sim dataset -> dict(x={frames [E,V,8,1408], proj [E,V,768], track [E,V,8,5]}, outcome [E,3] (dx, dy, roll),
    look/head [E,V], ok [E], eps [E]). Episodes missing any view are dropped."""
    root = Path(dataset)
    require_dataset_approval(root)
    f = np.load(root/'cosmos_agent.npz')
    V = int(f['view'].max()) + 1
    idx = {(int(e), int(v)): i for i, (e, v) in enumerate(zip(f['ep'], f['view']))}
    keep = [e for e in np.unique(f['ep']) if all((int(e), v) in idx for v in range(V))]
    sel = np.array([[idx[(int(e), v)] for v in range(V)] for e in keep])
    meta = [json.loads((root/f'ep{int(e):05d}/meta.json').read_text()) for e in keep]
    t = np.load(root/'block_track.npz')
    tidx = {(int(e), int(v)): i for i, (e, v) in enumerate(zip(t['ep'], t['view']))}
    head = np.array([[c.get('kind') == 'head' for c in m['cameras']] for m in meta])
    head[:, 0] = True   # v0 is always the (look-0) head view
    x = dict(frames=torch.tensor(f['frame_cls'][sel], dtype=torch.float32),
             proj=torch.tensor(f['proj'][sel], dtype=torch.float32),
             track=torch.tensor(t['track'][[[tidx[(int(e), v)] for v in range(V)] for e in keep]]))
    return dict(x=x, outcome=np.array([[*m['outcome']['disp'][:2], m['outcome']['roll']] for m in meta], np.float32),
                look=np.array([[c.get('look', 0) for c in m['cameras']] for m in meta]), head=head,
                ok=np.array([m['success'] for m in meta]), eps=np.array(keep))


class TrackNoise:
    """Resample SAM3 track errors (residuals + misses, per frame index) measured against the sim ground truth."""

    def __init__(self, sam3_path=NOISE, gt_path=NOISE_TRUTH):
        s, g = np.load(sam3_path), np.load(gt_path)
        gi = {(int(e), int(v)): t for e, v, t in zip(g['ep'], g['view'], g['track'])}
        S = s['track']
        G = np.array([gi[(int(e), int(v))] for e, v in zip(s['ep'], s['view'])])
        gvis, svis = G[..., 4] > 0, S[..., 4] > 0
        self.miss = torch.tensor((~svis & gvis).sum(0) / np.maximum(gvis.sum(0), 1), dtype=torch.float32)  # [8]
        self.res = [torch.tensor((S - G)[:, k, :4][svis[:, k] & gvis[:, k]]) for k in range(S.shape[1])]

    def __call__(self, t, gen=None):
        """t [..., 8, 5] -> noisy copy."""
        t = t.clone()
        for k, r in enumerate(self.res):
            n = t[..., k, 0].numel()
            idx = torch.randint(len(r), (n,), generator=gen)
            t[..., k, :4] += r[idx].reshape(*t.shape[:-2], 4).to(t.device)
            miss = (torch.rand(n, generator=gen) < self.miss[k]).reshape(t.shape[:-2]).to(t.device)
            t[..., k, :][miss] = 0
        return t


def validate_folds(folds, need_validation=True):
    """Each take/start group has exactly one fold. Real splits may have no validation takes (R08)."""
    if set(folds) != {'train', 'validation', 'test'}:
        raise ValueError('Need explicit train/validation/test folds')
    membership = {}
    for fold, identities in folds.items():
        if len(identities) != len(set(identities)):
            raise ValueError(f'Duplicate identity within {fold}')
        for identity in identities:
            if identity in membership:
                raise ValueError(f'Split leakage: {identity} in {membership[identity]} and {fold}')
            membership[identity] = fold
    if not folds['train'] or (need_validation and not folds['validation']):
        raise ValueError('Training and validation folds must be nonempty')
    return membership


def sim_folds(metas):
    folds = {fold: [] for fold in ('train', 'validation', 'test')}; group_folds = {}
    for i, meta in enumerate(metas):
        fold = meta.get('split'); group = meta.get('start_group')
        if fold not in folds or group is None:
            raise ValueError('Sim training requires explicit whole-start train/validation/test folds')
        if group in group_folds and group_folds[group] != fold:
            raise ValueError(f'Start group {group} crosses folds')
        group_folds[group] = fold; folds[fold].append(i)
    validate_folds(folds)
    return {fold: np.asarray(ids, dtype=int) for fold, ids in folds.items()}


def load_sim(dataset):
    """Sim dataset that passes every user rule (docs/decisions.md) -> (features, metas, rows, folds)."""
    from handsoff import rules
    root = Path(dataset).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Need complete simulation data')
    rules.registry()
    failed = [f'{name}: {detail}' for name, (applies, fn) in rules.CHECKS.items() if 'sim' in applies.split()
              for ok, detail in [fn(sim=root, manifest=manifest)] if not ok]
    if failed:
        raise ValueError('Simulation data breaks user rules: ' + '; '.join(failed))
    D = load_features(root)
    D['x'] = {k: v.float() for k, v in D['x'].items()}
    metas = [json.loads((root/f'ep{e:05d}/meta.json').read_text()) for e in D['eps']]
    if len(metas) != len(list(root.glob('ep*/meta.json'))) or not D['ok'].all() or not eligible(D['outcome']).all():
        raise ValueError('Need complete, physically successful, category-eligible simulation clips')
    views = D['x']['proj'].shape[1]
    if any(len(m['cameras']) != views for m in metas) or any(not torch.isfinite(v).all() for v in D['x'].values()):
        raise ValueError('Incomplete camera/feature cache or nonfinite input')
    rows = [dict(domain='sim', cat=category(D['outcome'][i]).tolist(), outcome=D['outcome'][i].tolist(),
                 clip_dir=str(root/f'ep{e:05d}'), start=m['start'], size=m['scene']['block_size'],
                 start_group=m['start_group']) for i, (e, m) in enumerate(zip(D['eps'], metas))]
    folds = sim_folds(metas)
    for group in {m['start_group'] for m in metas}:
        cameras = [m['cameras'] for m in metas if m['start_group'] == group]
        if any(c != cameras[0] for c in cameras):
            raise ValueError('Matched-start task operands must share cameras/looks')
    return D, metas, rows, folds


def load_queries(qdir, review_path=REVIEW_PATH):
    """Real dataset -> (keys [(ep, view)], x {frames, proj, track} [N, ...], metas {ep: meta}). Generated (Cosmos)
    views without the user's approval are left out."""
    qdir = Path(qdir)
    f = np.load(qdir/'cosmos_agent.npz')
    all_keys = list(zip(f['ep'].tolist(), f['view'].tolist()))
    metas = {e: json.loads((qdir/f'ep{e:05d}/meta.json').read_text()) for e in sorted({k[0] for k in all_keys})}
    review = read_reviews(review_path)
    keep = np.array([view_approved(review, metas[e], v) for e, v in all_keys], dtype=bool)
    keys = [k for k, ok in zip(all_keys, keep) if ok]
    if not keep.all():
        print(f'Human review: excluded {(~keep).sum()} unapproved/stale cached variant(s) from {qdir}', flush=True)
    t = np.load(qdir/'block_track_sam3.npz')
    ti = {(int(e), int(v)): tr for e, v, tr in zip(t['ep'], t['view'], t['track'])}
    x = dict(frames=torch.tensor(f['frame_cls'][keep], dtype=torch.float32),
             proj=torch.tensor(f['proj'][keep], dtype=torch.float32),
             track=torch.tensor(np.array([ti[k] for k in keys])))
    return keys, x, metas


def expected(label):
    """Confirmed take label -> task category (movement, roll), each -1/0/+1 (right / none / left)."""
    return np.array([{'right': -1, 'stay': 0, 'left': 1}[label['translation']],
                     {'right': -1, 'upright': 0, 'left': 1}[label['roll']]])


def read_inputs(ep, view):
    """The 8 model frames of one clip: raw RGB and agent-masked (grey)."""
    rgb = iio.imread(ep/f'v{view}_rgb.mp4')
    mask = iio.imread(ep/f'v{view}_mask.mp4')
    mask = (mask[..., 0] if mask.ndim == 4 else mask) > 127
    idx = np.linspace(0, len(rgb)-1, 8).round().astype(int)
    raw = rgb[idx]; masked = raw.copy(); masked[mask[idx]] = 128
    return raw, masked
