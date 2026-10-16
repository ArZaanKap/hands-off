"""The Inspire hand on a UR5e arm: the robot asset and the controller that runs the policy's hand waypoints on it.

Robot (assets/ur5e_inspire.xml, written by `export`): UR5e (MuJoCo Menagerie) -> flange adapter (63 mm, 10 mm, 50 g)
-> Inspire right hand, one kinematic chain; the hand's six floating wrist joints are removed. The arm joints are torque
motors limited to the UR5e torques (150 / 28 N m); a static pedestal holds the base where the person stood. Contact,
solver and integrator settings are the floating-hand scene's.

Controller: the policy outputs hand poses, so the arm only has to move the hand the way the floating wrist would. The
floating wrist is a mass-spring-damper at the hand base (M_f = 1.48 kg / 1.0 kg m^2, K = 1000, D = 77 / 63). The arm
renders exactly these dynamics, M_f a = K e - D v + W, every 2 ms:
    F = Lambda [M_f^-1 (K e - D v + W) - Jdot qdot] - W,    tau = J^T F + Coriolis      (W = contact wrench on the hand)
Lambda is the arm's task-space inertia and W what a wrist force/torque sensor measures (here J^-T tau_contact of the
previous step). No inverse kinematics during the motion (only for the start pose). The empty-hand transit segments are
slowed 1.5x so the joints stay under the UR5e's 180 deg/s.

  python -m handsoff.sim.arm export        (rewrite assets/ur5e_inspire.xml from the two source models)
  python -m handsoff.policy.policy eval ... --robot arm
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from handsoff.sim import primitive as P
from handsoff.sim.scene import (CONTACT_SOLIMP, CONTACT_SOLREF, GRASP_POS_BASE, HAND_CT, HAND_XML, R_BASE_GRASP,
                                WORLD_CT, WRIST)

ROOT = Path(__file__).resolve().parents[2]
UR_XML = ROOT / "external/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
ASSET = ROOT / "assets/ur5e_inspire.xml"
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
TAU_LIMIT = np.array([150., 150., 150., 28., 28., 28.])
# Arm base where the person stood (x, y, z, yaw) and the IK branch (elbow/wrist configuration) it starts in; found by
# a placement search; random reachable bases on the person's side work as well (863/864 episodes).
BASE = (-0.6116004869732276, 0.13570011842623847, 0.2657524660600267, -0.22158320694169698)
BRANCH = np.array([1.734822528386509, 0.4003581693048215, -0.9118912591827604, 0.5116221679442434,
                   -0.05755666377147817, 1.5707072114805145])
ADAPTER = 0.01                           # flange-to-hand adapter (m)
R_FLANGE_HAND = Rotation.from_matrix(np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))  # fingers along flange z
FLOOR_Z = -0.75
K = np.r_[[1000.] * 3, [1000.] * 3]      # the floating wrist: stiffness, damping (critical), mass / inertia
D = np.r_[[76.9967793] * 3, [63.3] * 3]
M_F = np.r_[[1.482] * 3, [1.0] * 3]
TRANSIT, TRANSIT_SLOW = [1, 2, 11, 12], 1.5   # plan segments with the open hand high above the table


def _meshes_absolute(spec, xml):
    """Mesh paths from the source model's folder (not resolved, so the exported paths stay inside the repo)."""
    folder = Path(os.path.abspath(Path(xml).parent / (spec.meshdir or "")))
    for m in spec.meshes:
        m.file = str(folder / m.file)
    spec.meshdir = ""


def rig_spec(friction=1.0, base=BASE):
    """UR5e + adapter + hand + pedestal, without the scene."""
    hs = mujoco.MjSpec.from_file(str(HAND_XML)); _meshes_absolute(hs, HAND_XML)
    hs.default.name = "inspire"          # keeps the hand's defaults addressable once attached under the UR5e's
    for i, g in enumerate(hs.geoms):     # the floating-hand scene's collision set-up
        if g.name.startswith("collision_hand"):
            g.contype, g.conaffinity, g.condim = 0, 0, 3
        elif g.type == mujoco.mjtGeom.mjGEOM_MESH and g.group == 0:
            g.name = f"collision_visual_{i}"
            g.contype, g.conaffinity, g.condim = HAND_CT, 0, 3
            g.friction = [friction, 0.005, 0.0001]
            g.solref, g.solimp = CONTACT_SOLREF, CONTACT_SOLIMP
    for b in hs.bodies:
        if b.name.startswith("right_"):
            b.gravcomp = 1.0
    for n in WRIST:
        hs.delete(hs.actuator(f"{n}_position")); hs.delete(hs.joint(n))
    hand = hs.body("right_hand_base")
    hand.pos = [0, 0, ADAPTER]
    hand.quat = R_FLANGE_HAND.as_quat(scalar_first=True)
    hand.add_site(name="grasp", pos=GRASP_POS_BASE, quat=R_BASE_GRASP.as_quat(scalar_first=True), size=[0.005, 0, 0])
    adapter = hand.add_body(name="adapter", pos=[0, ADAPTER / 2, 0], gravcomp=1.0,      # fills hand-base y in [0, ADAPTER]
                            quat=Rotation.from_euler("x", -np.pi / 2).as_quat(scalar_first=True))
    adapter.add_geom(name="adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0315, ADAPTER / 2, 0], mass=0.05,
                     rgba=(0.62, 0.64, 0.66, 1), contype=0, conaffinity=0)

    s = mujoco.MjSpec.from_file(str(UR_XML)); _meshes_absolute(s, UR_XML)
    s.modelname = "ur5e_inspire"
    for k in list(s.keys):               # ur5e.xml's 'home' keyframe is for the bare arm
        s.delete(k)
    s.option.timestep, s.option.iterations, s.option.ls_iterations = 0.002, 50, 50
    s.option.cone, s.option.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, 10
    s.option.integrator = hs.option.integrator          # the hand asset's Euler integrator and flags, under which the
    s.option.disableflags = hs.option.disableflags      # grips were tuned (ur5e.xml sets implicitfast)
    s.option.enableflags = hs.option.enableflags
    ub = s.body("base")
    ub.pos = list(base[:3])
    ub.quat = (Rotation.from_euler("z", base[3]) * Rotation.from_quat([0, 0, 0, -1], scalar_first=True)).as_quat(
        scalar_first=True)
    for b in s.bodies:
        if b.name != "world":
            b.gravcomp = 1.0             # the controller compensates gravity, as for the floating hand
    for i, g in enumerate(s.geoms):
        if g.group == 3:                 # arm collision capsules: hit table and block, not each other or the hand
            g.name = g.name or f"arm_collision_{i}"
            g.contype, g.conaffinity = HAND_CT, 0
    for a, lim in zip(s.actuators, TAU_LIMIT):   # position servos -> torque motors
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
        a.gainprm[:] = 0; a.gainprm[0] = 1; a.biasprm[:] = 0
        a.ctrlrange, a.ctrllimited = [-lim, lim], True
        a.forcerange, a.forcelimited = [-lim, lim], True
    s.site("attachment_site").attach_body(hand, "", "")
    h = (base[2] - FLOOR_Z) / 2
    s.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, h - 0.001, 0],
                         pos=[base[0], base[1], base[2] - h], rgba=(0.3, 0.3, 0.32, 1),
                         contype=WORLD_CT, conaffinity=WORLD_CT)
    return s


def export(out=ASSET):
    """Write the asset (mesh paths relative to it) and check that it reloads to the same model."""
    s = rig_spec(); m0 = s.compile()
    root = ET.fromstring(s.to_xml())
    for el in root.iter("mesh"):
        el.set("file", os.path.relpath(el.get("file"), out.parent))
    ET.indent(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!-- UR5e + adapter + Inspire right hand on a pedestal. Generated by `python -m handsoff.sim.arm "
                   "export`; meshes from external/ (scripts/setup.sh). -->\n" + ET.tostring(root, encoding="unicode") + "\n")
    m1 = mujoco.MjModel.from_xml_path(str(out))
    same = (m0.nq, m0.nu, m0.nbody, m0.ngeom) == (m1.nq, m1.nu, m1.nbody, m1.ngeom) and \
        np.allclose(m0.body_mass, m1.body_mass) and np.allclose(m0.body_pos, m1.body_pos)
    print(f"wrote {out}: {m1.nbody} bodies, {m1.nu} actuators, {m1.body_mass.sum():.2f} kg; reload matches: {same}")


def build(cfg, xml=ASSET):
    """The robot asset in the floating-hand scene: table (top at z = 0), floor, light and the block at the origin."""
    s = mujoco.MjSpec.from_file(str(xml))
    for g in s.geoms:
        if g.contype == HAND_CT and g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.friction = [cfg.friction, 0.005, 0.0001]
    wb = s.worldbody
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 3], dir=[0.2, 0.1, -1],
                 castshadow=False)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.40, 0.65, 0.02], pos=[0.15, 0, -0.02],
                contype=WORLD_CT, conaffinity=WORLD_CT | HAND_CT, friction=[cfg.friction, 0.005, 0.0001],
                rgba=cfg.table_rgba)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.01], pos=[0, 0, FLOOR_Z],
                rgba=cfg.floor_rgba, contype=0, conaffinity=0)
    half = np.array(cfg.block_size) / 2
    block = wb.add_body(name="block", pos=[0, 0, half[2]])
    block.add_freejoint(name="block_free")
    block.add_geom(name="block", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=cfg.block_mass,
                   rgba=cfg.block_rgba, contype=WORLD_CT, conaffinity=WORLD_CT | HAND_CT, condim=4,
                   friction=[cfg.friction, 0.005, 0.0001], solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP)
    m = s.compile()
    m.vis.global_.fovy = cfg.camera_fovy
    return m


class ArmHand:
    """Indices into the arm scene (the fields of scene.Hand that the outcome measures use)."""

    def __init__(self, m):
        self.m = m
        joints = np.array([m.joint(n).id for n in ARM])
        self.arm_qadr, self.arm_dof = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
        self.arm_act = np.array([m.actuator(n.removesuffix("_joint")).id for n in ARM])
        self.finger_act = np.array([i for i in range(m.nu) if i not in self.arm_act])
        self.finger_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in self.finger_act])
        self.qlo, self.qhi = m.jnt_range[joints].T
        self.block_qadr = m.jnt_qposadr[m.joint("block_free").id]
        self.block_dof = m.jnt_dofadr[m.joint("block_free").id]
        self.hand_base = m.body("right_hand_base").id
        self.grasp_site = m.site("grasp").id
        self.block_geom, self.table_geom = m.geom("block").id, m.geom("table").id
        hand_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith("right_")}
        self.clearance_geoms = np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] in hand_bodies
                                         and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[i] == 0])
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.jpd, self.jrd = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.Minv = np.zeros((m.nv, m.nv))

    def ik(self, pos, rot: Rotation, q0, iters=300):
        """Damped least squares on the grasp site (used once, for the start pose)."""
        d, q = mujoco.MjData(self.m), np.array(q0, float)
        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
            e = np.r_[pos - d.site_xpos[self.grasp_site],
                      Rotation.from_matrix(rot.as_matrix() @ d.site_xmat[self.grasp_site].reshape(3, 3).T).as_rotvec()]
            if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
                break
            mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.grasp_site)
            J = np.r_[self.jp, self.jr][:, self.arm_dof]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            n = np.abs(dq).max()
            q = np.clip(q + (dq * 0.5 / n if n > 0.5 else dq), self.qlo, self.qhi)
        return q

    def block_pose(self, d):
        q = d.qpos[self.block_qadr:self.block_qadr + 7]
        return q[:3].copy(), Rotation.from_quat(q[3:], scalar_first=True)

    def table_clearance(self, d, distmax=0.1):
        return min(mujoco.mj_geomDistance(self.m, d, self.table_geom, int(g), distmax, None) for g in self.clearance_geoms)

    def torque(self, d, p_t, R_t):
        """Joint torques that make the hand base move like the floating wrist toward (p_t, R_t)."""
        m, b, dof = self.m, self.hand_base, self.arm_dof
        e = np.r_[p_t - d.xpos[b], Rotation.from_matrix(R_t.as_matrix() @ d.xmat[b].reshape(3, 3).T).as_rotvec()]
        mujoco.mj_jacBody(m, d, self.jp, self.jr, b)
        J = np.r_[self.jp, self.jr][:, dof]
        mujoco.mj_solveM(m, d, self.Minv, np.eye(m.nv))
        Lam = np.linalg.inv(J @ self.Minv[np.ix_(dof, dof)] @ J.T + 1e-6 * np.eye(6))
        W = np.linalg.solve(J.T, d.qfrc_constraint[dof])                 # contact wrench on the hand
        mujoco.mj_jacDot(m, d, self.jpd, self.jrd, d.xpos[b], b)
        Jdqd = np.r_[self.jpd, self.jrd][:, dof] @ d.qvel[dof]
        F = Lam @ ((K * e - D * (J @ d.qvel[dof]) + W) / M_F - Jdqd) - W
        return np.clip(J.T @ F + (d.qfrc_bias - d.qfrc_gravcomp)[dof], -TAU_LIMIT, TAU_LIMIT)


def run(wps, act, cfg, block_xy, block_yaw, model, record_hz=0.0):
    """primitive.run on the arm: same waypoints, interpolation, settle time and outcome measures; transit segments
    slowed 1.5x; the arm torque is recomputed every 2 ms physics step."""
    wps = [(w[0] * (TRANSIT_SLOW if k in TRANSIT else 1.0), *w[1:]) for k, w in enumerate(wps)]
    m = model; d = mujoco.MjData(m); h = ArmHand(m)
    a = h.block_qadr
    d.qpos[a:a + 3] = [block_xy[0], block_xy[1], cfg.block_size[2] / 2]
    d.qpos[a + 3:a + 7] = Rotation.from_euler("z", block_yaw).as_quat(scalar_first=True)
    d.qpos[h.arm_qadr] = h.ik(wps[0][1], wps[0][2], BRANCH)
    d.qpos[h.finger_qadr] = d.ctrl[h.finger_act] = wps[0][3]
    mujoco.mj_forward(m, d)
    p0, _ = h.block_pose(d)
    release_gap = touchdown_speed = np.inf
    release_stage, grasp_stage, lower_stage = 9, 3, 8
    turn = wps[lower_stage][2] * wps[grasp_stage][2].inv()
    expected = wps[lower_stage][1] - turn.apply(wps[grasp_stage][1] - p0)
    commanded_release_gap = float(expected[2] - P.hz_after(cfg, act.roll))
    durs = np.array([w[0] for w in wps]); edges = np.cumsum(durs)
    log = dict(t=[], block_pos=[], block_quat=[], qpos=[], seg=[]) if record_hz else None
    rec_every = int(round(1 / (record_hz * m.opt.timestep))) if record_hz else 0
    max_h, step, peak_off_axis, min_clearance = 0.0, 0, 0.0, h.table_clearance(d)
    while d.time < edges[-1] + 0.4:      # last 0.4 s: hold the retreat pose while the block settles
        k = min(int(np.searchsorted(edges, d.time, side="right")), len(wps) - 1)
        s = np.clip((d.time - (edges[k - 1] if k else 0)) / durs[k], 0, 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)
        prev = wps[k - 1] if k else wps[0]
        if step % P.CTRL_EVERY == 0:     # the hand target, every 10 ms as for the floating hand
            R_t = Slerp([0, 1], Rotation.concatenate([prev[2], wps[k][2]]))(s) * R_BASE_GRASP.inv()
            p_t = prev[1] + s * (wps[k][1] - prev[1]) - R_t.apply(GRASP_POS_BASE)
        d.ctrl[h.arm_act] = h.torque(d, p_t, R_t)
        d.ctrl[h.finger_act] = prev[3] + s * (wps[k][3] - prev[3])
        previous_vz = float(d.qvel[h.block_dof + 2])
        mujoco.mj_step(m, d)
        step += 1
        block_position, block_rotation = h.block_pose(d)
        max_h = max(max_h, block_position[2] - p0[2])
        if k == release_stage and not np.isfinite(release_gap):
            release_gap = float(block_position[2] - P.hz_after(cfg, act.roll))
        if max_h > .049 and not np.isfinite(touchdown_speed):
            for c in d.contact:
                if {int(c.geom1), int(c.geom2)} == {h.block_geom, h.table_geom}:
                    touchdown_speed = max(0., -previous_vz)
                    break
        if step % P.CTRL_EVERY == 0:
            peak_off_axis = max(peak_off_axis, P.orientation_errors(block_rotation, block_yaw, act.roll)[1])
            min_clearance = min(min_clearance, h.table_clearance(d))
        if log is not None and step % rec_every == 0:
            log["t"].append(d.time); log["block_pos"].append(block_position)
            log["block_quat"].append(block_rotation.as_quat(scalar_first=True))
            log["qpos"].append(d.qpos.copy()); log["seg"].append(k)
        if not np.all(np.isfinite(d.qpos)):
            return P.Outcome(act, np.zeros(3), 0.0, 0.0, False)
    p1, r1 = h.block_pose(d)
    eo, ea = P.orientation_errors(r1, block_yaw, act.roll)
    return P.Outcome(act, p1 - p0, P.block_roll(r1, block_yaw), float(max_h), True,
                     {k: np.array(v) for k, v in log.items()} if log else {},
                     orientation_error=eo, off_axis_error=ea, peak_off_axis_error=float(peak_off_axis),
                     min_table_clearance=float(min_clearance), clearance_checked=True, release_gap=release_gap,
                     commanded_release_gap=commanded_release_gap, touchdown_speed=touchdown_speed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export"])
    ap.parse_args()
    export()
