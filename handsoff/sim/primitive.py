"""Pick, carry, turn and place the tall block with the floating Inspire hand (one fixed motion per task).

An action is (dx, dy, roll): move the block by (dx, dy) on the table and roll it about the block's initial forward
axis (local x) through its centre. Positive roll tips the top toward local +y (the person's left at a square start).
Translation stays in the table frame (R02).

A grip (GraspParams) places the grasp frame relative to the block: hand orientation = block yaw * yaw * TOP * tilt
about the grasp y axis (tilt > 0 points the fingers down), and an offset of the grasp site from the block centre in that
hand frame (z < 0 = palm standing off the face). The grip is fixed to the block during the carry, so the hand turns
with the roll. Rolls use a side grasp; for roll left it is the roll-right grasp turned 180 deg about the forward axis,
so the wrist stays on the person's side like a human right hand (R20, motion "clearance_side_human").

The hand starts and ends at rest out of view, approaches from above, releases at the lowest height that clears the
table, and the reported outcome is measured from the block, not commanded.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from handsoff.sim.scene import TOP, Hand, SceneConfig, build

MOTION = "clearance_side_human"
LIFT = 0.07   # a real pick-up lifts the block centre at least 0.7 * LIFT (m)
# Rest pose out of view at the person's right, like the human hand before and after a take.
HOME = np.array([-0.30, -0.60, 0.15])
HOME_HOLD = 0.8  # s at rest before and after (the block is still, as in the human clips)
CTRL_EVERY = 5


@dataclass
class GraspParams:
    NAMES = ("off_x", "off_y", "off_z", "yaw", "tilt", "thumb_yaw", "thumb_flex", "finger_flex", "pre_open")
    off_x: float = -0.02        # along the fingers (m)
    off_y: float = 0.0          # toward the little finger (m)
    off_z: float = -0.05        # along the palm normal (m); negative = palm off the block face
    yaw: float = 0.0            # about world z (rad)
    tilt: float = 0.0           # about the grasp y axis (rad); > 0 points the fingers down
    thumb_yaw: float = 0.8      # fractions of each actuator's range
    thumb_flex: float = 0.5
    finger_flex: float = 0.5
    pre_open: float = 0.2       # finger and thumb flex during approach, as a fraction of the closed flex
    bank_extra: float = 0.0     # clearance_side_human roll left only: extra turn about the finger axis (rad), palm
                                # from the block's side face toward its upper edge (R20)

    def as_array(self):
        """Nine values; a tenth only for a roll-left grasp that uses bank_extra."""
        return np.array([getattr(self, n) for n in self.NAMES] + ([self.bank_extra] if self.bank_extra else []))

    @classmethod
    def from_array(cls, a):
        return cls(*map(float, a))


@dataclass
class Action:
    dx: float = 0.0
    dy: float = 0.0
    roll: float = 0.0    # rad about initial block x, + = top toward block +y

    def as_array(self):
        return np.r_[self.dx, self.dy, self.roll]


@dataclass
class Outcome:
    action: Action
    disp: np.ndarray        # (dx, dy, dz) of the block centre (m)
    roll: float             # measured tip in the selected initial frame (rad)
    max_lift: float
    ok_sim: bool
    log: dict = field(default_factory=dict)
    orientation_error: float = np.inf
    off_axis_error: float = np.inf
    peak_off_axis_error: float = np.inf
    min_table_clearance: float = np.inf
    clearance_checked: bool = False
    release_gap: float = np.inf
    commanded_release_gap: float = np.inf
    touchdown_speed: float = np.inf

    @property
    def vec(self):
        return np.r_[self.disp[:2], self.roll]


def hand_q(hand, gp: GraspParams, closed: bool):
    """12 finger targets: thumb yaw, then thumb pitch/intermediate/distal, then 4 x (proximal, intermediate)."""
    s = 1.0 if closed else gp.pre_open
    f = np.empty(len(hand.finger_names))
    for i, n in enumerate(hand.finger_names):
        if n == "thumb_proximal_yaw":
            f[i] = gp.thumb_yaw
        elif n.startswith("thumb"):
            f[i] = s * gp.thumb_flex
        else:
            f[i] = s * gp.finger_flex
    return hand.finger_q(f)


def grasp_frame(gp: GraspParams, centre, block_yaw, bank=0.):
    rot = Rotation.from_euler("z", block_yaw + gp.yaw) * TOP
    if bank:
        rot=rot*Rotation.from_euler('x',bank)
    rot=rot*Rotation.from_euler("y",gp.tilt)
    return centre + rot.apply([gp.off_x, gp.off_y, gp.off_z]), rot


def roll_turn(roll, block_yaw):
    """World transform for a turn about the initial block x axis."""
    r0 = Rotation.from_euler("z", block_yaw)
    return r0 * Rotation.from_euler("x", -roll) * r0.inv()


def hz_after(cfg, roll):
    """Height of the block centre after a roll about local x."""
    _, y, h = cfg.block_size
    return abs(np.cos(roll)) * h / 2 + abs(np.sin(roll)) * y / 2


def orientation_errors(R, block_yaw, command_roll):
    """Full goal-orientation error and non-roll error; both in radians, no Euler yaw."""
    r0 = Rotation.from_euler("z", block_yaw)
    goal = roll_turn(command_roll, block_yaw) * r0
    pure_measured = roll_turn(block_roll(R, block_yaw), block_yaw) * r0
    return float((R * goal.inv()).magnitude()), float((R * pure_measured.inv()).magnitude())


def grasp_variant(gp: GraspParams, roll):
    """Bank of the side grasp about the finger axis for a roll: roll right -90 deg; roll left +90 deg + bank_extra, the
    roll-right grasp turned about the forward axis (fingers still point away from the person, R20)."""
    if abs(roll) <= np.pi/4:
        return gp, 0.
    return gp, (np.pi/2 + gp.bank_extra if roll > 0 else -np.pi/2)


def task_grip(record, roll):
    """Grip parameters for one task from a grip record: upright top grasp, a separately tuned roll-left grasp when
    the record has one (R20), else the roll grasp."""
    if abs(roll) < np.pi/4 and 'upright_params' in record:
        return GraspParams.from_array(record['upright_params'])
    if roll > 0 and 'left_params' in record:
        return GraspParams.from_array(record['left_params'])
    return GraspParams.from_array(record['params'])


def plan(hand, act: Action, gp: GraspParams, centre, block_yaw, cfg):
    """Waypoints [(duration s, grasp-site pos, Rotation, finger q)]; the first is held for its duration."""
    gp, bank = grasp_variant(gp, act.roll)
    grasp_pos, rot0 = grasp_frame(gp, centre, block_yaw, bank=bank)
    turn = roll_turn(act.roll, block_yaw)
    goal_c = centre + [act.dx, act.dy, hz_after(cfg, act.roll) - cfg.block_size[2] / 2]
    held = lambda c, R: (c + R.apply(grasp_pos - centre), R * rot0)  # noqa: E731
    q_open, q_close = hand_q(hand, gp, False), hand_q(hand, gp, True)
    carry = np.array([0, 0, cfg.carry_height])
    p_lift, r_lift = held(centre + carry, Rotation.identity())
    p_mid, r_mid = held(goal_c + carry, turn)
    p_set, r_set = held(goal_c + [0, 0, 0.004], turn)
    # Release at the lowest height that clears the table for both open and closed finger geometry.
    probe = mujoco.MjData(hand.m)
    needed = 0.
    for q in (q_close, q_open):
        probe.qpos[hand.wrist_qadr] = hand.wrist_targets(p_set, r_set)
        probe.qpos[hand.finger_qadr] = q
        mujoco.mj_kinematics(hand.m, probe)
        needed = max(needed, .005 - hand.minimum_geometry_z(probe))
    p_set = p_set + [0, 0, needed]
    high = np.array([0, 0, cfg.approach_height])
    safe_rot = Rotation.from_euler("z", block_yaw) * TOP
    rest = centre + HOME + [0, 0, .15]
    return [(HOME_HOLD, rest, safe_rot, q_open),
            (1.0, grasp_pos + high, safe_rot, q_open),
            (.8, grasp_pos + high, rot0, q_open),
            (1.0, grasp_pos, rot0, q_open),
            (.6, grasp_pos, rot0, q_close),
            (.9, p_lift, r_lift, q_close),
            (1.4, p_mid, r_mid, q_close),
            (.3, p_mid, r_mid, q_close),
            (.9, p_set, r_set, q_close),
            (.7, p_set, r_set, q_open),
            (.9, p_set + high, r_set, q_open),
            (.8, p_set + high, safe_rot, q_open),
            (1.0, rest, safe_rot, q_open),
            (HOME_HOLD, rest, safe_rot, q_open)]


def block_roll(R: Rotation, block_yaw=0.0):
    """Signed tip in the initial block frame."""
    zb = (Rotation.from_euler("z", block_yaw).inv() * R).apply([0, 0, 1.0])
    return float(np.arctan2(zb[1], zb[2]))


def _unwrap(q, prev):
    q = q.copy()
    q[3:] = prev[3:] + (q[3:] - prev[3:] + np.pi) % (2 * np.pi) - np.pi
    return q


def run(act: Action, gp: GraspParams, cfg: SceneConfig = SceneConfig(), block_xy=(0.0, 0.0), block_yaw=0.0,
        model=None, record_hz=0.0, motion=MOTION):
    """Simulate one task; `record_hz` > 0 logs block pose, qpos and waypoint index at that rate."""
    if motion != MOTION:
        raise ValueError(f"Only the {MOTION} motion is supported (R20)")
    m = model if model is not None else build(cfg)
    d = mujoco.MjData(m)
    hand = Hand(m)
    a = hand.block_qadr
    d.qpos[a:a + 3] = [block_xy[0], block_xy[1], cfg.block_size[2] / 2]
    d.qpos[a + 3:a + 7] = Rotation.from_euler("z", block_yaw).as_quat(scalar_first=True)
    mujoco.mj_forward(m, d)
    p0, _ = hand.block_pose(d)
    wps = plan(hand, act, gp, p0.copy(), block_yaw, cfg)
    release_gap = touchdown_speed = np.inf
    release_stage, grasp_stage, lower_stage = 9, 3, 8
    turn = wps[lower_stage][2] * wps[grasp_stage][2].inv()
    expected = wps[lower_stage][1] - turn.apply(wps[grasp_stage][1] - p0)
    commanded_release_gap = float(expected[2] - hz_after(cfg, act.roll))
    block_vadr = m.jnt_dofadr[m.joint('block_free').id]

    q_w = hand.wrist_targets(wps[0][1], wps[0][2])
    d.qpos[hand.wrist_qadr], d.qpos[hand.finger_qadr] = q_w, wps[0][3]
    d.ctrl[hand.wrist_act], d.ctrl[hand.finger_act] = q_w, wps[0][3]
    mujoco.mj_forward(m, d)

    durs = np.array([w[0] for w in wps])
    edges = np.cumsum(durs)
    log = dict(t=[], block_pos=[], block_quat=[], qpos=[], seg=[]) if record_hz else None
    rec_every = int(round(1 / (record_hz * m.opt.timestep))) if record_hz else 0
    max_h, step, k = 0.0, 0, 0
    peak_off_axis = 0.0
    min_clearance = hand.table_clearance(d)
    while d.time < edges[-1] + 0.4:  # last 0.4 s: hold the retreat pose while the block settles
        k = min(int(np.searchsorted(edges, d.time, side="right")), len(wps) - 1)
        s = np.clip((d.time - (edges[k - 1] if k else 0)) / durs[k], 0, 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)
        prev = wps[k - 1] if k else wps[0]
        if step % CTRL_EVERY == 0:
            pos = prev[1] + s * (wps[k][1] - prev[1])
            rot = Slerp([0, 1], Rotation.concatenate([prev[2], wps[k][2]]))(s)
            q_w = _unwrap(hand.wrist_targets(pos, rot), q_w)
            d.ctrl[hand.wrist_act] = q_w
        d.ctrl[hand.finger_act] = prev[3] + s * (wps[k][3] - prev[3])
        previous_vz=float(d.qvel[block_vadr+2])
        mujoco.mj_step(m, d)
        step += 1
        block_position, block_rotation = hand.block_pose(d)
        max_h = max(max_h, block_position[2] - p0[2])
        if k == release_stage and not np.isfinite(release_gap):
            release_gap = float(block_position[2] - hz_after(cfg, act.roll))
        if max_h > .049 and not np.isfinite(touchdown_speed):
            for contact in d.contact:
                if {int(contact.geom1),int(contact.geom2)}=={hand.block_geom,hand.table_geom}:
                    touchdown_speed=max(0.,-previous_vz)
                    break
        if step % CTRL_EVERY == 0:
            peak_off_axis = max(peak_off_axis, orientation_errors(block_rotation, block_yaw, act.roll)[1])
            min_clearance = min(min_clearance, hand.table_clearance(d))
        if log is not None and step % rec_every == 0:
            bp, bq = hand.block_pose(d)
            log["t"].append(d.time); log["block_pos"].append(bp); log["block_quat"].append(bq.as_quat(scalar_first=True))
            log["qpos"].append(d.qpos.copy()); log["seg"].append(k)
        if not np.all(np.isfinite(d.qpos)):
            return Outcome(act, np.zeros(3), 0.0, 0.0, False)
    p1, r1 = hand.block_pose(d)
    if log is not None:
        log = {k: np.array(v) for k, v in log.items()}
    eo, ea = orientation_errors(r1, block_yaw, act.roll)
    return Outcome(act, p1 - p0, block_roll(r1, block_yaw), float(max_h), True, log or {},
                   orientation_error=eo, off_axis_error=ea, peak_off_axis_error=float(peak_off_axis),
                   min_table_clearance=float(min_clearance), clearance_checked=True, release_gap=release_gap,
                   commanded_release_gap=commanded_release_gap, touchdown_speed=touchdown_speed)


def error(o: Outcome):
    """(position error m, roll error rad) of the measured outcome vs the commanded action."""
    return float(np.linalg.norm(o.disp[:2] - [o.action.dx, o.action.dy])), float(abs(o.roll - o.action.roll))


def success(o: Outcome, cfg: SceneConfig = SceneConfig(), pos_tol=0.03, roll_tol=np.radians(20),
            orientation_tol=np.radians(20), off_axis_tol=np.radians(10)):
    """Task reached AND clean rotation through motion (sampled at 100 Hz), including upright moves."""
    ep, er = error(o)
    dz_ok = abs(o.disp[2] - (hz_after(cfg, o.action.roll) - cfg.block_size[2] / 2)) < 0.015
    lifted = o.max_lift > 0.7 * LIFT  # the block was really picked up, not pushed
    clean_rotation = o.orientation_error < orientation_tol and o.off_axis_error < off_axis_tol and o.peak_off_axis_error < off_axis_tol
    clearance_ok = o.clearance_checked and o.min_table_clearance >= -0.001
    return bool(o.ok_sim and lifted and ep < pos_tol and er < roll_tol and dz_ok and clean_rotation and clearance_ok)


def placement_success(o: Outcome, cfg: SceneConfig):
    """Existing task/clearance checks plus low release and slow first touchdown."""
    return bool(success(o,cfg) and o.release_gap<=.01 and o.commanded_release_gap<=.015
                and np.isfinite(o.release_gap) and np.isfinite(o.commanded_release_gap)
                and o.touchdown_speed<=.3)
