"""Arm: the shipped asset is the generated rig, and the arm runs a scripted plan like the floating hand."""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from handsoff.policy import plans as pl
from handsoff.sim import arm

SCENE = Path(__file__).resolve().parents[1] / 'weights/scene.json'


class ArmTest(unittest.TestCase):
    def test_asset_is_the_generated_rig(self):
        m0, m1 = arm.rig_spec().compile(), mujoco.MjModel.from_xml_path(str(arm.ASSET))
        self.assertEqual((m0.nq, m0.nu, m0.nbody, m0.ngeom), (m1.nq, m1.nu, m1.nbody, m1.ngeom))
        for f in ('body_mass', 'body_pos', 'dof_armature', 'actuator_gainprm', 'actuator_ctrlrange', 'geom_friction'):
            self.assertTrue(np.allclose(getattr(m0, f), getattr(m1, f)), f)

    def test_arm_matches_floating_hand(self):
        job = ((0, -1), np.zeros(3), None, None)          # stay + roll right, nominal start, scripted plan
        pl.init_worker(SCENE); floating = pl.execute(job)
        pl.init_worker(SCENE, 'arm'); on_arm = pl.execute(job)
        pl._arm = None
        self.assertTrue(floating['success'] and on_arm['success'])
        self.assertLess(np.abs(np.array(floating['outcome']) - on_arm['outcome']).max(), 0.002)


if __name__ == '__main__':
    unittest.main()
