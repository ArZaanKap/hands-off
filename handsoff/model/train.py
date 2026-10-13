"""Train the intent head on frozen video features, with explicit task composition.

Stages (same sim data and settings; only the real-video term differs):
  sim              sim clips only
  sim_real         + the user's original phone clips of the training takes, taught with their confirmed task labels
  sim_real_cosmos  + the user-approved Cosmos-Transfer appearance variants of those takes
Losses: cross-look contrastive task loss, movement/roll classifiers, composition ranking (z(a) - z(b) + z(c) should be
nearest a clip of the target task, from another start), and for real stages a labelled real-to-sim task loss.
Every sim camera trains (R17); real takes use their labels (R18); head.pt is the final epoch of a fixed schedule (R19).
Real folds come from the split JSON by whole take (R08).
Usage: python -m handsoff.model.train DATA --out DIR --stage STAGE [--real-data DIR] [--epochs N] [--seed S]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from handsoff.model.data import FEAT, NOISE, NOISE_TRUTH, TrackNoise, expected, load_queries, load_sim, validate_folds
from handsoff.model.head import Head, embed_all, view_pairs
from handsoff.model.tasks import category, composition_accuracy, composition_queries
from handsoff.real.review_gate import is_control_video, require_dataset_approval

ROOT = Path(__file__).resolve().parents[2]
STAGES = ('sim', 'sim_real', 'sim_real_cosmos')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt'); torch.save(value, tmp); tmp.replace(path)


class TaskHead(nn.Module):
    """The head plus linear movement/roll classifiers used only as training losses."""

    def __init__(self):
        super().__init__()
        self.encoder = Head(FEAT)
        self.translation = nn.Linear(64, 3)
        self.roll = nn.Linear(64, 3)

    def forward(self, x):
        return self.encoder(x)


def contrastive_task_loss(z1, z2, cats, tau=.1):
    positives = (cats[:, None] == cats[None, :]).all(-1)
    logits = z1 @ z2.T / tau
    def one(matrix):
        return (torch.logsumexp(matrix, -1) - torch.logsumexp(matrix.masked_fill(~positives, -torch.inf), -1)).mean()
    return .5 * (one(logits) + one(logits.T))


def factor_loss(model, z1, z2, cats):
    targets = cats.long()+1
    return sum(F.cross_entropy(layer(z), targets[:, i])
               for i, layer in enumerate((model.translation, model.roll)) for z in (z1, z2))/4


def composition_loss(za, zb, zc, gallery, gallery_cats, targets, allowed, tau=.1):
    """Rank all correct-category targets above negatives; source start group excluded."""
    good = (gallery_cats[None] == targets[:, None]).all(-1) & allowed
    if not allowed.any(-1).all() or not good.any(-1).all():
        raise ValueError('Every composition needs an eligible target outside its source group')
    query = F.normalize(za-zb+zc, dim=-1)
    scores = query @ gallery.T / tau
    return (torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), -1)
            - torch.logsumexp(scores.masked_fill(~good, -torch.inf), -1)).mean()


def real_task_loss(model, real_x, ids, real_cats, z_sim, sim_cats, tau=.1):
    """Labelled real clips: each is pulled toward same-task sim clips in the batch and pushed from other tasks,
    plus the same movement/roll classifiers as sim. Only training-fold takes are passed in."""
    device = z_sim.device
    zr = model({k: v[ids].to(device) for k, v in real_x.items()})
    positives = (real_cats[:, None] == sim_cats[None, :]).all(-1); keep = positives.any(-1)
    logits = zr @ z_sim.T / tau
    contrast = (torch.logsumexp(logits[keep], -1) - torch.logsumexp(logits.masked_fill(~positives, -torch.inf)[keep], -1)).mean()
    targets = real_cats.long()+1
    return contrast + (F.cross_entropy(model.translation(zr), targets[:, 0]) + F.cross_entropy(model.roll(zr), targets[:, 1]))/2


def is_generated(camera):
    return bool(camera.get('generated') or camera.get('look_name') not in (None, 'original'))


def real_training_clips(keys, metas, folds, stage):
    """Indices of the real clips this stage trains on: training-fold takes' originals (+ approved variants)."""
    membership = validate_folds(folds, need_validation=False); used = []
    for i, (episode, view) in enumerate(keys):
        meta = metas[episode]; parent = int(meta.get('parent_episode', episode)); camera = meta['cameras'][view]
        if parent not in membership:
            raise ValueError(f'Unlisted take {parent}; add it to a fold before use')
        if is_control_video(camera.get('source')) or camera.get('media_role') == 'control':
            raise ValueError('Control maps are never real model inputs')
        if membership[parent] != 'train' or (is_generated(camera) and stage != 'sim_real_cosmos'):
            continue
        if is_generated(camera) and (camera.get('task_preservation_pass') is not True
                                     or camera.get('task_preservation_source_sha256') != camera.get('source_sha256')):
            raise ValueError('Generated training views need hash-bound task-preservation checks as well as human approval')
        used.append(i)
    for parent in folds['train']:
        kinds = {metas[keys[i][0]]['cameras'][keys[i][1]].get('kind') for i in used
                 if int(metas[keys[i][0]].get('parent_episode', keys[i][0])) == parent and not is_generated(metas[keys[i][0]]['cameras'][keys[i][1]])}
        if not {'head', 'second'} <= kinds:
            raise ValueError(f'Take {parent} needs both original phone views')
    if stage == 'sim_real_cosmos' and not any(is_generated(metas[keys[i][0]]['cameras'][keys[i][1]]) for i in used):
        raise ValueError('Cosmos stage needs approved training-take RGB variants; originals alone are not that stage')
    return sorted(used)


@torch.no_grad()
def validation(model, D, rows, ids, device, real=None):
    """Held-out sim starts: direct task retrieval and composition accuracy against a gallery of other starts, head
    view in the other look. With real validation takes (choose_epochs), also their nearest-clip task accuracy."""
    groups = sorted({rows[i]['start_group'] for i in ids})
    if len(groups) < 2:
        raise ValueError('Validation needs disjoint source and gallery start groups')
    source = [int(i) for i in ids if rows[i]['start_group'] in groups[:len(groups)//2]]
    gallery_ids = [int(i) for i in ids if i not in source]
    queries = composition_queries(rows, source)
    encoded = embed_all({k: v[ids] for k, v in D['x'].items()}, model.encoder, device)
    z = np.zeros((len(rows), *encoded.shape[1:]), dtype=encoded.dtype); z[ids] = encoded
    gallery_views = []
    for i in gallery_ids:
        candidates = np.flatnonzero(D['head'][i] & (D['look'][i] != D['look'][i, 0]))
        if not len(candidates):
            raise ValueError('Validation gallery needs a head view in a different look')
        gallery_views.append(int(candidates[0]))
    gallery = z[gallery_ids, gallery_views]; cats = np.array([rows[i]['cat'] for i in gallery_ids])
    score = lambda predicted, truth: dict(n=len(truth), translation=float((predicted[:, 0] == truth[:, 0]).mean()),  # noqa: E731
                                          roll=float((predicted[:, 1] == truth[:, 1]).mean()),
                                          joint=float((predicted == truth).all(-1).mean()))
    truth = np.array([rows[i]['cat'] for i in source])
    result = dict(direct=score(cats[(z[source, 0] @ gallery.T).argmax(-1)], truth),
                  composition=composition_accuracy(queries, rows, z[:, 0], gallery, gallery_ids))
    scores = [result['direct']['joint'], result['composition']['joint']]
    if real is not None:
        x, keys, metas, selected = real
        vectors = model({k: v[selected].to(device) for k, v in x.items()}).cpu().numpy()
        truth = np.stack([expected(metas[keys[i][0]]['label']) for i in selected])
        result['real_direct'] = score(cats[(vectors @ gallery.T).argmax(-1)], truth)
        scores.append(result['real_direct']['joint'])
    result['score'] = float(np.mean(scores))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', choices=STAGES, default='sim'); ap.add_argument('--real-data', type=Path)
    ap.add_argument('--real-split', type=Path, default=ROOT/'config/real_split.json')
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=128); ap.add_argument('--composition-batch', type=int, default=48)
    ap.add_argument('--composition-weight', type=float, default=1.); ap.add_argument('--factor-weight', type=float, default=1.)
    ap.add_argument('--real-weight', type=float, default=1.); ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--resume', action='store_true', help='continue from OUT/last.pt')
    a = ap.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.composition_batch < 1 or min(a.composition_weight, a.factor_weight, a.real_weight) < 0:
        ap.error('Epochs/batches must be positive and loss weights nonnegative')
    if a.stage != 'sim' and not a.real_data:
        ap.error('Real stages need a prepared real dataset')
    torch.set_num_threads(8); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D, metas, rows, folds = load_sim(a.dataset); tr = folds['train']
    cats = torch.tensor(category(D['outcome']), dtype=torch.long, device=a.device)
    queries = composition_queries(rows, tr)
    if any(len(np.unique(D['look'][i])) < 2 for i in tr):
        raise ValueError('Every training episode needs two looks')
    real_x = None; real_used = []; real_validation = None; real_val_ids = []
    if a.real_data:
        require_dataset_approval(a.real_data); keys, real_x, real_metas = load_queries(a.real_data)
        real_x = {k: v.float() for k, v in real_x.items()}
        if any(not torch.isfinite(v).all() for v in real_x.values()):
            raise ValueError('Nonfinite real input features')
        split = json.loads(a.real_split.read_text())
        real_used = real_training_clips(keys, real_metas, split['folds'], a.stage) if a.stage != 'sim' else []
        membership = validate_folds(split['folds'], need_validation=False)
        real_val_ids = [i for i, (e, v) in enumerate(keys) if membership[int(real_metas[e].get('parent_episode', e))] == 'validation'
                        and not is_generated(real_metas[e]['cameras'][v])]
        real_validation = (real_x, keys, real_metas, real_val_ids) if real_val_ids else None
        real_cats = torch.tensor(np.stack([expected(real_metas[keys[i][0]]['label']) for i in real_used]) if real_used
                                 else np.zeros((0, 2)), dtype=torch.long, device=a.device)
    provenance = dict(dataset=str(a.dataset.resolve()), stage=a.stage, seed=a.seed,
        sim_manifest_sha256=digest(a.dataset/'manifest.json'), sim_features_sha256=digest(a.dataset/'cosmos_agent.npz'),
        sim_tracks_sha256=digest(a.dataset/'block_track.npz'), train_episodes=D['eps'][tr].tolist(),
        sim_metadata_sha256=hashlib.sha256(json.dumps(metas, sort_keys=True).encode()).hexdigest(),
        validation_episodes=D['eps'][folds['validation']].tolist(), test_episodes=D['eps'][folds['test']].tolist(),
        real_split_sha256=digest(a.real_split) if real_x is not None else None,
        real_features_sha256=digest(a.real_data/'cosmos_agent.npz') if real_x is not None else None,
        real_tracks_sha256=digest(a.real_data/'block_track_sam3.npz') if real_x is not None else None,
        real_metadata_sha256=hashlib.sha256(json.dumps(real_metas, sort_keys=True).encode()).hexdigest() if real_x is not None else None,
        real_training_keys=[list(keys[i]) for i in real_used], real_validation_keys=[list(keys[i]) for i in real_val_ids],
        batch_size=a.batch_size, composition_batch=a.composition_batch, device=a.device,
        optimizer=dict(lr=1e-3, weight_decay=1e-2), tau=.1,
        loss_weights=dict(composition=a.composition_weight, factor=a.factor_weight, real=a.real_weight),
        noise_sha256=digest(NOISE), noise_truth_sha256=digest(NOISE_TRUTH), feat=list(FEAT),
        held_out_views='none', real_loss='labelled', select='last', source_sha256=digest(__file__))
    model = TaskHead().to(a.device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    noise = TrackNoise()
    start_epoch = 0; history = []
    if a.resume:
        checkpoint = torch.load(a.out/'last.pt', map_location=a.device, weights_only=False)
        if checkpoint['provenance'] != provenance:
            raise ValueError('Resume data/split/config changed')
        model.load_state_dict(checkpoint['model']); opt.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state = checkpoint['numpy_rng']; torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if a.device == 'cuda':
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        start_epoch = checkpoint['epoch']+1; history = checkpoint['history']
    a.out.mkdir(parents=True, exist_ok=True); start = time.time(); group_ids = np.array([r['start_group'] for r in rows])
    # Validation clips get one fixed draw of track noise, as a stand-in for SAM3 tracks.
    eval_data = dict(D, x=dict(D['x'])); eval_data['x']['track'] = D['x']['track'].clone()
    eval_data['x']['track'][folds['validation']] = noise(D['x']['track'][folds['validation']], torch.Generator().manual_seed(1000+a.seed))
    views = np.arange(D['look'].shape[1])

    def encode(indices, views):
        x = {k: v[indices, views].to(a.device) for k, v in D['x'].items()}
        x['track'] = noise(x['track'])
        return model(x)
    for epoch in range(start_epoch, a.epochs):
        model.train(); losses = []
        order = rng.permutation(tr)
        for begin in range(0, len(tr), a.batch_size):
            ids = order[begin:begin+a.batch_size]; pairs = view_pairs(rng, ids, D['look'])
            z1, z2 = encode(ids, pairs[:, 0]), encode(ids, pairs[:, 1])
            invariant = contrastive_task_loss(z1, z2, cats[ids]); factors = factor_loss(model, z1, z2, cats[ids])
            selected = [queries[i] for i in rng.integers(len(queries), size=a.composition_batch)]
            sources = np.array([[q[k] for k in ('a', 'b', 'c')] for q in selected])
            source_views = np.array([rng.choice(views) for _ in selected])
            za, zb, zc = [encode(sources[:, i], source_views) for i in range(3)]
            gallery = encode(tr, np.array([rng.choice(views) for _ in tr]))
            allowed = torch.tensor(group_ids[sources[:, 0], None] != group_ids[tr][None], device=a.device)
            targets = torch.tensor([q['target'] for q in selected], device=a.device)
            comp = composition_loss(za, zb, zc, gallery, cats[tr], targets, allowed)
            real = z1.new_zeros(()) if a.stage == 'sim' else real_task_loss(model, real_x, real_used, real_cats, z1, cats[ids])
            loss = invariant+a.factor_weight*factors+a.composition_weight*comp+a.real_weight*real
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([float(v.detach()) for v in (loss, invariant, factors, comp, real)])
        model.eval(); val = validation(model, eval_data, rows, folds['validation'], a.device, real_validation)
        history.append(dict(epoch=epoch, loss=dict(zip(('total', 'invariance', 'factors', 'composition', 'real'),
                                                      np.mean(losses, axis=0).tolist())), validation=val))
        atomic_torch(a.out/'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch, history=history,
                                           provenance=provenance, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                           cuda_rng=torch.cuda.get_rng_state_all() if a.device == 'cuda' else []))
        if epoch+1 == a.epochs:
            atomic_torch(a.out/'head.pt', model.encoder.state_dict())
        atomic_json(a.out/'results.json', dict(schema='composition_head_training_v1', complete=epoch+1 == a.epochs,
                                              provenance=provenance, epochs_completed=epoch+1, history=history,
                                              seconds=time.time()-start))
        print(f'epoch{epoch+1}: loss={history[-1]["loss"]["total"]:.4f} validation={val["score"]:.3f}', flush=True)
    print('Saved head:', a.out/'head.pt', flush=True)


if __name__ == '__main__':
    main()
