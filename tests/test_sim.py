"""Scene and robot motion: block-frame roll, rectangular support, success gates, collision geometry, R20 grasp."""
import json
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from handsoff.sim.primitive import (Action, GraspParams, Outcome, block_roll, grasp_frame, grasp_variant, hz_after,
                                    orientation_errors, placement_success, plan, roll_turn, success, task_grip)
from handsoff.sim.scene import HAND_CT, TOP, Hand, SceneConfig, build

GRIP = json.loads((Path(__file__).resolve().parents[1]/'config/grip_v3.json').read_text())


def clean_outcome(**kw):
    return Outcome(Action(), np.zeros(3), 0, .07, True, orientation_error=0, off_axis_error=0, peak_off_axis_error=0,
                   clearance_checked=True, min_table_clearance=0, **kw)


class RollGeometry(unittest.TestCase):
    def test_turn_keeps_the_initial_forward_axis(self):
        for yaw in np.radians([-60, -30, 0, 30, 60]):
            r0 = Rotation.from_euler('z', yaw)
            for roll in np.radians([-90, -45, 45, 90]):
                r1 = roll_turn(roll, yaw) * r0
                np.testing.assert_allclose(r1.apply([1, 0, 0]), r0.apply([1, 0, 0]), atol=1e-12)
                self.assertAlmostEqual(block_roll(r1, yaw), roll)
                self.assertLess(max(orientation_errors(r1, yaw, roll)), 1e-12)

    def test_rolled_block_rests_on_its_sideways_face(self):
        self.assertAlmostEqual(hz_after(SceneConfig(), np.pi/2), .019)   # 38 mm wide, not 62 mm deep


class SuccessGates(unittest.TestCase):
    def test_heading_drift_or_twist_fails(self):
        cfg = SceneConfig()
        eo, ea = orientation_errors(Rotation.from_euler('z', np.radians(19)), 0, 0)
        o = clean_outcome(); o.orientation_error = eo; o.off_axis_error = o.peak_off_axis_error = ea
        self.assertFalse(success(o, cfg))
        o.orientation_error = o.off_axis_error = o.peak_off_axis_error = 0
        self.assertTrue(success(o, cfg))
        o.peak_off_axis_error = np.radians(12)
        self.assertFalse(success(o, cfg))

    def test_success_needs_checked_table_clearance(self):
        o = clean_outcome(); o.min_table_clearance = -.0005
        self.assertTrue(success(o, SceneConfig()))
        o.min_table_clearance = -.0011
        self.assertFalse(success(o, SceneConfig()))
        o.min_table_clearance = 0; o.clearance_checked = False
        self.assertFalse(success(o, SceneConfig()))

    def test_gentle_placement_checks_release_height_and_touchdown(self):
        o = clean_outcome(release_gap=.008, commanded_release_gap=.012, touchdown_speed=.2)
        self.assertTrue(placement_success(o, SceneConfig()))
        for key, value in (('touchdown_speed', 1.2), ('release_gap', .02), ('commanded_release_gap', .17)):
            bad = clean_outcome(release_gap=.008, commanded_release_gap=.012, touchdown_speed=.2)
            setattr(bad, key, value)
            self.assertFalse(placement_success(bad, SceneConfig()), key)


class Scene(unittest.TestCase):
    def test_hand_only_with_person_proxy_and_one_collider_per_link(self):
        m = build(SceneConfig())
        names = {m.geom(i).name for i in range(m.ngeom)}
        self.assertNotIn('person_forearm', names)
        self.assertTrue({'person_head', 'person_torso', 'table', 'block'} <= names)
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                self.assertEqual(m.geom_contype[g], HAND_CT if m.geom_group[g] == 0 else 0)
            if m.geom(g).name.startswith('collision_hand'):
                self.assertEqual(m.geom_contype[g], 0)
        with self.assertRaises(ValueError):
            build(SceneConfig(forearm=True))

    def test_geometry_height_and_signed_table_distance(self):
        m = build(SceneConfig()); d = mujoco.MjData(m); h = Hand(m)
        d.qpos[h.wrist_qadr] = h.wrist_targets([0, 0, .5], TOP); mujoco.mj_forward(m, d)
        z = h.minimum_geometry_z(d)
        self.assertGreater(h.table_clearance(d), 0)
        d.qpos[h.wrist_qadr[2]] += .03; mujoco.mj_forward(m, d)
        self.assertAlmostEqual(h.minimum_geometry_z(d) - z, .03)
        d.qpos[h.wrist_qadr] = 0; mujoco.mj_forward(m, d)
        self.assertLess(h.table_clearance(d), -.01)


class Grasp(unittest.TestCase):
    def test_each_task_uses_its_own_grasp(self):
        self.assertEqual(task_grip(GRIP, 0.).as_array().tolist(), GRIP['upright_params'])
        self.assertEqual(task_grip(GRIP, -np.pi/2).as_array().tolist(), GRIP['params'])
        self.assertEqual(task_grip(GRIP, np.pi/2).as_array().tolist(), GRIP['left_params'])

    def test_fingers_point_away_from_the_person_for_every_task(self):   # R20
        for roll in (-np.pi/2, 0., np.pi/2):
            gp, bank = grasp_variant(task_grip(GRIP, roll), roll)
            _, rot = grasp_frame(gp, np.zeros(3), 0., bank=bank)
            self.assertGreater(rot.apply([1, 0, 0])[0], .5, roll)

    def test_carried_block_gets_the_commanded_roll_and_move(self):
        cfg = SceneConfig(); h = Hand(build(cfg)); centre = np.array([.01, -.02, .043]); yaw = .4
        for roll in (-np.pi/2, np.pi/2):
            act = Action(dx=.03, dy=.15, roll=roll); gp = GraspParams()
            w = plan(h, act, gp, centre, yaw, cfg)
            turn = w[6][2] * w[3][2].inv()   # carry pose x grasp pose^-1
            self.assertLess((turn * roll_turn(roll, yaw).inv()).magnitude(), 1e-12)
            np.testing.assert_allclose(w[6][1] - turn.apply(w[3][1] - centre),
                                       centre + [act.dx, act.dy, .019 - .043 + cfg.carry_height], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
