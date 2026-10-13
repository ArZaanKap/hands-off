"""Trainer: composition loss, fold leakage, which real clips train, gradient flow, test isolation and exact resume."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from handsoff import rules
from handsoff.model import train
from handsoff.model.data import sim_folds, validate_folds
from handsoff.model.head import Head
from handsoff.model.tasks import composition_queries
from tests.test_tasks import grid_rows


def factor_vectors(rows):
    """Ideal intent vectors: one-hot movement + one-hot roll."""
    z = torch.zeros(len(rows), 64)
    for i, row in enumerate(rows):
        d, r = row['cat']; z[i, d+1] = 1.; z[i, 3+r+1] = 1.
    return F.normalize(z, dim=-1)


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)

    def test_composition_loss_is_low_only_for_the_right_targets(self):
        rows = grid_rows(2); q = composition_queries(rows, list(range(9)))
        z = factor_vectors(rows); cats = torch.tensor([r['cat'] for r in rows])
        a, b, c = [torch.stack([z[p[k]] for p in q]) for k in ('a', 'b', 'c')]
        target = torch.tensor([p['target'] for p in q]); allowed = torch.ones(12, 9, dtype=torch.bool)
        self.assertTrue(torch.equal(cats[9:][(F.normalize(a-b+c, dim=-1) @ z[9:].T).argmax(-1)], target))
        correct = train.composition_loss(a, b, c, z[9:], cats[9:], target, allowed)
        wrong = target.clone(); wrong[:, 1] = torch.where(wrong[:, 1] == 0, torch.ones(12, dtype=torch.long), -wrong[:, 1])
        self.assertLess(float(correct), .1)
        self.assertGreater(float(train.composition_loss(a, b, c, z[9:], cats[9:], wrong, allowed)), float(correct)+4.)
        with self.assertRaisesRegex(ValueError, 'eligible target'):
            train.composition_loss(a[:1], b[:1], c[:1], z[9:], cats[9:], target[:1], torch.zeros(1, 9, dtype=torch.bool))

    def test_fold_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Split leakage'):
            validate_folds(dict(train=[1], validation=[1], test=[2]))
        with self.assertRaisesRegex(ValueError, 'crosses folds'):
            sim_folds([dict(split='train', start_group=0), dict(split='validation', start_group=0)])

    def test_only_training_takes_train_and_variants_need_checks(self):
        folds = dict(train=[1], validation=[2], test=[3]); keys = []; metas = {}
        for ep in (1, 2, 3):
            cameras = [dict(kind='head'), dict(kind='second'),
                       dict(kind='head', look_name='blue', generated=True, source='generated.mp4', source_sha256='hash',
                            task_preservation_pass=True, task_preservation_source_sha256='hash')]
            metas[ep] = dict(label=dict(translation='right', roll='upright'), cameras=cameras)
            keys.extend((ep, v) for v in range(3))
        self.assertEqual(train.real_training_clips(keys, metas, folds, 'sim_real_cosmos'), [0, 1, 2])
        self.assertEqual(train.real_training_clips(keys, metas, folds, 'sim_real'), [0, 1])
        metas[1]['cameras'][2]['task_preservation_source_sha256'] = 'stale'
        with self.assertRaisesRegex(ValueError, 'task-preservation'):
            train.real_training_clips(keys, metas, folds, 'sim_real_cosmos')
        metas[1]['cameras'][2]['task_preservation_source_sha256'] = 'hash'; metas[4] = metas[1]
        with self.assertRaisesRegex(ValueError, 'Unlisted take'):
            train.real_training_clips(keys+[(4, 0)], metas, folds, 'sim_real')
        metas[1]['cameras'][2]['source'] = 'generated_control_edge.mp4'
        with self.assertRaisesRegex(ValueError, 'Control maps'):
            train.real_training_clips(keys, metas, folds, 'sim_real_cosmos')

    def test_losses_update_encoder_and_both_readouts(self):
        torch.manual_seed(5); model = train.TaskHead()
        x = dict(frames=torch.randn(9, 8, 1408), proj=torch.randn(9, 768), track=torch.randn(9, 8, 5))
        cats = torch.tensor([r['cat'] for r in grid_rows()]); optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        z = model(x); before = float(train.factor_loss(model, z, z, cats).detach())
        for _ in range(5):
            z = model(x); loss = train.factor_loss(model, z, z, cats); optimizer.zero_grad(); loss.backward()
            for module in (model.encoder, model.translation, model.roll):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            optimizer.step()
        self.assertLess(float(train.factor_loss(model, model(x), model(x), cats).detach()), before)
        exported = Head(); exported.load_state_dict(model.encoder.state_dict()); exported.eval(); model.eval()
        self.assertTrue(torch.equal(exported(x), model(x)))   # head.pt alone reproduces the encoder

    def test_validation_never_sees_test_episodes(self):
        rows = grid_rows(5); count = len(rows); model = train.TaskHead().eval()
        x = dict(frames=torch.zeros(count, 4, 8, 1408), proj=torch.zeros(count, 4, 768), track=torch.zeros(count, 4, 8, 5))
        x['proj'][:, :, 0] = torch.arange(count, dtype=torch.float32)[:, None]
        D = dict(x=x, head=np.tile([True, False, True, False], (count, 1)), look=np.tile([0, 0, 1, 1], (count, 1)))
        observed = []
        handle = model.encoder.register_forward_pre_hook(lambda module, args: observed.extend(args[0]['proj'][:, 0].tolist()))
        train.validation(model, D, rows, np.arange(18, 36), 'cpu'); handle.remove()
        self.assertEqual(set(observed), set(range(18, 36)))

    def test_resume_matches_uninterrupted_training(self):
        # Feature-only synthetic fixture (random features): tests bookkeeping, not model capability.
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); dataset = d/'fixture'; dataset.mkdir(); rng = np.random.default_rng(7); metas = []
            for group in range(4):
                for direction in (-1, 0, 1):
                    for roll in (-1, 0, 1):
                        e = 40000+len(metas); ep = dataset/f'ep{e:05d}'; ep.mkdir()
                        meta = dict(episode=e, split='train' if group < 2 else 'validation', start_group=group,
                                    start=dict(xy=[group*.001, 0.], yaw=0.), scene=dict(block_size=[.062, .038, .086]),
                                    success=True, outcome=dict(disp=[0., direction*.15, 0.], roll=roll*np.pi/2),
                                    cameras=[dict(kind='head' if v % 2 == 0 else 'free', look=v//2) for v in range(4)])
                        (ep/'meta.json').write_text(json.dumps(meta)); metas.append(meta)
            (dataset/'manifest.json').write_text(json.dumps(dict(complete=True)))
            eps = np.repeat([m['episode'] for m in metas], 4); views = np.tile(np.arange(4), len(metas)); count = len(eps)
            np.savez(dataset/'cosmos_agent.npz', ep=eps, view=views, proj=rng.normal(size=(count, 768)).astype('float32'),
                     frame_cls=rng.normal(size=(count, 8, 1408)).astype('float32'))
            np.savez(dataset/'block_track.npz', ep=eps, view=views, track=rng.normal(size=(count, 8, 5)).astype('float32'))

            def run(out, epochs, resume=False, extra=()):
                args = ['train', str(dataset), '--out', str(out), '--device', 'cpu', '--epochs', str(epochs),
                        '--batch-size', '18', '--composition-batch', '4', *(['--resume'] if resume else []), *extra]
                with patch.object(sys, 'argv', args), patch.object(rules, 'CHECKS', {}), \
                        patch.object(rules, 'registry', lambda: {}), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
            run(d/'resumed', 1); run(d/'resumed', 2, True); run(d/'fresh', 2)
            resumed = torch.load(d/'resumed/last.pt', weights_only=False); fresh = torch.load(d/'fresh/last.pt', weights_only=False)
            self.assertEqual(resumed['history'], fresh['history'])
            for key in fresh['model']:
                self.assertTrue(torch.equal(resumed['model'][key], fresh['model'][key]), key)
            with self.assertRaisesRegex(ValueError, 'Resume data/split/config changed'):
                run(d/'resumed', 3, True, ['--composition-weight', '.2'])


if __name__ == '__main__':
    unittest.main()
