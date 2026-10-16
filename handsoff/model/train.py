"""Train the intent head on frozen video features of sim clips, with explicit task composition.

Losses: a cross-look contrastive task loss (two views of an episode from different looks; same task = positive) and a
composition ranking loss (z(a) - z(b) + z(c) should be nearest a clip of the target task, from another start).
Sim tracks get SAM3-like noise. Every sim camera trains; head.pt is the final epoch of a fixed schedule.
The head never sees a real clip.
Usage: python -m handsoff.model.train DATA --out DIR [--epochs N] [--seed S] [--device cuda|cpu] [--resume]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from handsoff.model.data import NOISE, NOISE_TRUTH, TrackNoise, load_sim
from handsoff.model.head import Head, embed_all, view_pairs
from handsoff.model.tasks import category, composition_accuracy, composition_queries


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt'); torch.save(value, tmp); tmp.replace(path)


def contrastive_task_loss(z1, z2, cats, tau=.1):
    positives = (cats[:, None] == cats[None, :]).all(-1)
    logits = z1 @ z2.T / tau
    def one(matrix):
        return (torch.logsumexp(matrix, -1) - torch.logsumexp(matrix.masked_fill(~positives, -torch.inf), -1)).mean()
    return .5 * (one(logits) + one(logits.T))


def composition_loss(za, zb, zc, gallery, gallery_cats, targets, allowed, tau=.1):
    """Rank all correct-category targets above negatives; source start group excluded."""
    good = (gallery_cats[None] == targets[:, None]).all(-1) & allowed
    if not allowed.any(-1).all() or not good.any(-1).all():
        raise ValueError('Every composition needs an eligible target outside its source group')
    query = F.normalize(za-zb+zc, dim=-1)
    scores = query @ gallery.T / tau
    return (torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), -1)
            - torch.logsumexp(scores.masked_fill(~good, -torch.inf), -1)).mean()


@torch.no_grad()
def validation(model, D, rows, ids, device):
    """Held-out sim starts: direct task retrieval and composition accuracy against a gallery of other starts, head
    view in the other look."""
    groups = sorted({rows[i]['start_group'] for i in ids})
    if len(groups) < 2:
        raise ValueError('Validation needs disjoint source and gallery start groups')
    source = [int(i) for i in ids if rows[i]['start_group'] in groups[:len(groups)//2]]
    gallery_ids = [int(i) for i in ids if i not in source]
    queries = composition_queries(rows, source)
    encoded = embed_all({k: v[ids] for k, v in D['x'].items()}, model, device)
    z = np.zeros((len(rows), *encoded.shape[1:]), dtype=encoded.dtype); z[ids] = encoded
    gallery_views = []
    for i in gallery_ids:
        candidates = np.flatnonzero(D['head'][i] & (D['look'][i] != D['look'][i, 0]))
        if not len(candidates):
            raise ValueError('Validation gallery needs a head view in a different look')
        gallery_views.append(int(candidates[0]))
    gallery = z[gallery_ids, gallery_views]; cats = np.array([rows[i]['cat'] for i in gallery_ids])
    predicted, truth = cats[(z[source, 0] @ gallery.T).argmax(-1)], np.array([rows[i]['cat'] for i in source])
    result = dict(direct=dict(n=len(truth), translation=float((predicted[:, 0] == truth[:, 0]).mean()),
                              roll=float((predicted[:, 1] == truth[:, 1]).mean()), joint=float((predicted == truth).all(-1).mean())),
                  composition=composition_accuracy(queries, rows, z[:, 0], gallery, gallery_ids))
    result['score'] = float(np.mean([result['direct']['joint'], result['composition']['joint']]))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=128); ap.add_argument('--composition-batch', type=int, default=48)
    ap.add_argument('--composition-weight', type=float, default=1.)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--resume', action='store_true', help='continue from OUT/last.pt')
    a = ap.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.composition_batch < 1 or a.composition_weight < 0:
        ap.error('Epochs/batches must be positive and the loss weight nonnegative')
    torch.set_num_threads(8); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D, metas, rows, folds = load_sim(a.dataset); tr = folds['train']
    cats = torch.tensor(category(D['outcome']), dtype=torch.long, device=a.device)
    queries = composition_queries(rows, tr)
    if any(len(np.unique(D['look'][i])) < 2 for i in tr):
        raise ValueError('Every training episode needs two looks')
    provenance = dict(dataset=str(a.dataset.resolve()), seed=a.seed,
        sim_manifest_sha256=digest(a.dataset/'manifest.json'), sim_features_sha256=digest(a.dataset/'cosmos_agent.npz'),
        sim_tracks_sha256=digest(a.dataset/'block_track.npz'), train_episodes=D['eps'][tr].tolist(),
        sim_metadata_sha256=hashlib.sha256(json.dumps(metas, sort_keys=True).encode()).hexdigest(),
        validation_episodes=D['eps'][folds['validation']].tolist(), test_episodes=D['eps'][folds['test']].tolist(),
        batch_size=a.batch_size, composition_batch=a.composition_batch, device=a.device,
        optimizer=dict(lr=1e-3, weight_decay=1e-2), tau=.1, composition_weight=a.composition_weight,
        noise_sha256=digest(NOISE), noise_truth_sha256=digest(NOISE_TRUTH), select='last', source_sha256=digest(__file__))
    model = Head().to(a.device)
    torch.nn.Linear(64, 3), torch.nn.Linear(64, 3)   # two unused draws keep the seeds' random streams as in the reported runs
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
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
            invariant = contrastive_task_loss(z1, z2, cats[ids])
            selected = [queries[i] for i in rng.integers(len(queries), size=a.composition_batch)]
            sources = np.array([[q[k] for k in ('a', 'b', 'c')] for q in selected])
            source_views = np.array([rng.choice(views) for _ in selected])
            za, zb, zc = [encode(sources[:, i], source_views) for i in range(3)]
            gallery = encode(tr, np.array([rng.choice(views) for _ in tr]))
            allowed = torch.tensor(group_ids[sources[:, 0], None] != group_ids[tr][None], device=a.device)
            targets = torch.tensor([q['target'] for q in selected], device=a.device)
            comp = composition_loss(za, zb, zc, gallery, cats[tr], targets, allowed)
            loss = invariant + a.composition_weight*comp
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([float(v.detach()) for v in (loss, invariant, comp)])
        model.eval(); val = validation(model, eval_data, rows, folds['validation'], a.device)
        history.append(dict(epoch=epoch, loss=dict(zip(('total', 'invariance', 'composition'), np.mean(losses, axis=0).tolist())),
                            validation=val))
        atomic_torch(a.out/'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch, history=history,
                                           provenance=provenance, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                           cuda_rng=torch.cuda.get_rng_state_all() if a.device == 'cuda' else []))
        if epoch+1 == a.epochs:
            atomic_torch(a.out/'head.pt', model.state_dict())
        atomic_json(a.out/'results.json', dict(schema='intent_head_training_v2', complete=epoch+1 == a.epochs,
                                              provenance=provenance, epochs_completed=epoch+1, history=history,
                                              seconds=time.time()-start))
        print(f'epoch{epoch+1}: loss={history[-1]["loss"]["total"]:.4f} validation={val["score"]:.3f}', flush=True)
    print('Saved head:', a.out/'head.pt', flush=True)


if __name__ == '__main__':
    main()
