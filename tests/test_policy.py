"""Policy: waypoint encoding round trip, and z can change which motion runs but never the motion's offset."""
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from handsoff.policy import plans as pl
from handsoff.policy.policy import D_Y, K_BINS, Policy, start_feats


class PolicyTest(unittest.TestCase):
    def test_waypoints_round_trip(self):
        rng = np.random.default_rng(0); c = np.array([.01, -.02, .043]); durations = np.arange(1., pl.N_WP + 1)
        wps = [(d, c + rng.normal(0, .05, 3), Rotation.random(random_state=k), rng.normal(0, 1, 12)) for k, d in enumerate(durations)]
        back = pl.decode(pl.encode(wps, c), c, durations)
        for (d, p, R, q), (d2, p2, R2, q2) in zip(wps, back):
            self.assertEqual(d, d2); self.assertTrue(np.allclose(p, p2) and np.allclose(q, q2))
            self.assertLess((R * R2.inv()).magnitude(), 1e-6)

    def test_offset_sees_only_the_start(self):
        torch.manual_seed(0); model = Policy(torch.randn(K_BINS, D_Y)).eval()
        s = torch.tensor(start_feats(np.array([.01, -.02, .1])))
        c1, c2 = torch.cat([torch.randn(1, 64), s], 1), torch.cat([torch.randn(1, 64), s], 1)
        (l1, o1), (l2, o2) = model.heads(c1), model.heads(c2)
        self.assertTrue(torch.equal(o1, o2))            # a different z: same per-bin motions
        self.assertFalse(torch.equal(l1, l2))           # ... but a different choice of bin
        b = l1.argmax(1)
        self.assertTrue(torch.allclose(model.act(c1), model.centres[b] + o1[0, b]))


if __name__ == '__main__':
    unittest.main()
