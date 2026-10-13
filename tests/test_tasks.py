"""Task categories and composition queries."""
import unittest

import numpy as np

from handsoff.model.tasks import category, composition_accuracy, composition_queries, eligible, matched_start


def grid_rows(groups=1):
    """All nine tasks from each of `groups` starts."""
    return [dict(domain='sim', cat=[d, r], outcome=[0., d*.15, r*np.pi/2], clip_dir=f'fixture/{g}/{d}/{r}',
                 start=dict(xy=[g*.001, 0.], yaw=0.), size=[.062, .038, .086], start_group=g)
            for g in range(groups) for d in (-1, 0, 1) for r in (-1, 0, 1)]


class Tasks(unittest.TestCase):
    def test_roll_and_movement_are_separate(self):
        outcomes = np.array([[0, -.12, -np.pi/2], [0, 0, -np.pi/2], [0, -.12, 0]])
        self.assertEqual(category(outcomes).tolist(), [[-1, -1], [0, -1], [-1, 0]])
        self.assertFalse(eligible([.08, 0, 0]))   # drifted forward: not a clean example

    def test_twelve_queries_per_start_cover_three_families_and_mirrors(self):
        q = composition_queries(grid_rows(), list(range(9)))
        self.assertEqual(len(q), 12)
        self.assertEqual({p['family'] for p in q}, {'extract_roll', 'extract_translation', 'transfer_roll'})
        rows = grid_rows()
        for p in q:   # the target is the arithmetic of the operands' categories
            np.testing.assert_array_equal(np.array(rows[p['a']]['cat']) - rows[p['b']]['cat'] + rows[p['c']]['cat'], p['target'])

    def test_operands_must_share_the_start(self):
        rows = grid_rows(); rows[1]['start'] = dict(xy=[.1, 0.], yaw=0.)
        with self.assertRaisesRegex(ValueError, 'exact start'):
            composition_queries(rows, list(range(9)))
        a = dict(start=dict(xy=[0, 0], yaw=0), size=[.05, .05, .08])
        self.assertFalse(matched_start(a, dict(a, start=dict(xy=[0, 0], yaw=np.deg2rad(30)))))

    def test_accuracy_excludes_operand_clips_from_the_gallery(self):
        rows = [dict(clip_dir=k, cat=c) for k, c in [('a', [-1, 1]), ('b', [0, 1]), ('c', [0, 0]), ('wrong', [1, 0]), ('good', [0, 1])]]
        z = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., 1.], [-1., 0.]])
        q = dict(a=0, b=1, c=2, family='extract_roll', target=[0, 1])
        # a - b + c points at 'wrong' (operand b is never a candidate); with 'wrong' gone the answer is 'good'
        self.assertEqual(composition_accuracy([q], rows, z, z[[1, 3, 4]], [1, 3, 4])['joint'], 0.)
        self.assertEqual(composition_accuracy([q], rows, z, z[[1, 4]], [1, 4])['joint'], 1.)


if __name__ == '__main__':
    unittest.main()
