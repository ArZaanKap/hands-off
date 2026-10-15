"""Hand motions as waypoint vectors: the scripted teacher plans and their execution in MuJoCo physics.

A motion is 14 waypoints; each is stored relative to the block centre as position (3), 6-D rotation (6) and the 12
finger targets, so a plan is 294 numbers. Durations stay the script's. Starts are drawn like the sim dataset
(xy +-3 cm, yaw +-0.25 rad). Scene, grip and controller come from the sim dataset's manifest or its scene.json copy
(configure()).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from handsoff.model.tasks import category, eligible
from handsoff.sim import primitive as P
from handsoff.sim.scene import Hand, SceneConfig, build

DY = {-1: -.15, 0: 0., 1: .15}; ROLL = {-1: -np.pi/2, 0: 0., 1: np.pi/2}
TASKS = [(m, r) for m in (-1, 0, 1) for r in (-1, 0, 1)]
N_WP, D_WP = 14, 21
CFG = GRIP = MOTION = None
_hand = _model = None


def configure(source):
    """Scene, grip and motion of the sim dataset: its manifest.json, or a scene.json holding just those three."""
    global CFG, GRIP, MOTION
    source = Path(source); man = json.loads((source/'manifest.json' if source.is_dir() else source).read_text())
    CFG, GRIP, MOTION = SceneConfig(**man['scene']), man['grip'], man['motion']


def init_worker(source):
    """Pool initializer: one MuJoCo model per worker."""
    global _hand, _model
    configure(source); _model = build(CFG); _hand = Hand(_model)


def starts(n, seed):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-.03, .03, (n, 2)), rng.uniform(-.25, .25, n)]


def centre(s):
    return np.array([s[0], s[1], CFG.block_size[2] / 2])


def encode(wps, c):
    return np.concatenate([np.r_[p - c, R.as_matrix()[:, :2].T.ravel(), q] for _, p, R, q in wps])


def decode(y, c, durations):
    wps = []
    for k, w in enumerate(y.reshape(N_WP, D_WP)):
        a, b = w[3:6], w[6:9]                                 # Gram-Schmidt on the two predicted columns
        a = a / np.linalg.norm(a); b = b - a * (a @ b); b = b / np.linalg.norm(b)
        wps.append((durations[k], w[:3] + c, Rotation.from_matrix(np.c_[a, b, np.cross(a, b)]), w[9:]))
    return wps


def teacher(job):
    """(task, start) -> the scripted plan as a waypoint vector, and its durations."""
    t, s = job
    act = P.Action(dy=DY[t[0]], roll=ROLL[t[1]]); gp = P.task_grip(GRIP, ROLL[t[1]])
    wps = P.plan(_hand, act, gp, centre(s), s[2], CFG)
    return encode(wps, centre(s)), np.array([w[0] for w in wps])


def execute(job):
    """One physics run of waypoints y (None = the script itself) for task t at start s. Success = the right task
    (movement and roll category), a clean outcome and a gentle placement."""
    t, s, y, durations = job
    act = P.Action(dy=DY[t[0]], roll=ROLL[t[1]]); gp = P.task_grip(GRIP, ROLL[t[1]])
    original = P.plan
    if y is not None:
        P.plan = lambda *a, **k: decode(y, centre(s), durations)
    try:
        o = P.run(act, gp, CFG, block_xy=s[:2].tolist(), block_yaw=float(s[2]), model=_model, motion=MOTION)
    finally:
        P.plan = original
    out = np.r_[o.disp[:2], o.roll]
    right = bool((category(out) == np.array(t)).all())
    gentle = bool(P.placement_success(o, CFG))
    return dict(task=list(t), start=s.tolist(), outcome=out.tolist(), right_task=right, clean=bool(eligible(out)),
                gentle=gentle, success=right and bool(eligible(out)) and gentle)


def build_set(n, seed, pool):
    """Teacher plans for n starts x 9 tasks: y [9n, 294], task index t, start s, durations."""
    s = starts(n, seed); jobs = [(t, si) for si in s for t in TASKS]
    res = pool.map(teacher, jobs, chunksize=64)
    y = np.stack([r[0] for r in res]).astype(np.float32); durations = res[0][1]
    assert all(np.allclose(r[1], durations) for r in res), 'script durations differ between tasks'
    return dict(y=y, t=np.array([TASKS.index(t) for t, _ in jobs]), s=np.stack([si for _, si in jobs]), durations=durations)
