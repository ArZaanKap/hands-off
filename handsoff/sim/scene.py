"""MuJoCo scene: floating-wrist Inspire right hand (5 fingers, SPIDER asset), table, tall block, person proxy.

The wrist is six position-controlled joints (3 slide + 3 hinge), standing in for an ideal arm (no forearm).
Hand bodies are gravity-compensated, as an arm controller would do. The hand collides with the table and block through
convex hulls of its visible meshes.

World frame matches the human recordings: the person (and the hand) sit at -x and reach toward +x ("forward"); +y is
the person's left; the table top is z = 0. The finite table, floor and a static torso + head standing in for the person
give free cameras a sense of direction; the person is masked with the hand ("agent"), so masked video keeps a grey
person silhouette, as in the masked phone clips.

Grasp frame (`grasp` site, centre of the palm face): x = finger direction, z = palm normal (out of the palm),
y = z cross x (toward the little finger).
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
HAND_XML = ROOT / "external/spider/spider/assets/robots/inspire/right.xml"

HAND_CT, WORLD_CT = 2, 1  # contact bits: hand touches block+table, never itself
CONTACT_SOLREF = [0.004, 1.0]
CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2]
WRIST = ["right_pos_x", "right_pos_y", "right_pos_z", "right_rot_x", "right_rot_y", "right_rot_z"]

# Grasp frame in the hand-base frame: fingers along base -y, palm normal base -x, little finger base +z.
R_BASE_GRASP = Rotation.from_matrix(np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]))
GRASP_POS_BASE = np.array([-0.012, -0.11, -0.004])
# Top grasp in the world: fingers forward (+x), palm down.
TOP = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0]))


@dataclass
class SceneConfig:
    """Scene settings; datasets record every field. build() makes the final scene only (checked below)."""
    block_size: tuple = (0.062, 0.038, 0.086)  # full edge lengths, m: the real block standing on its end
    block_mass: float = 0.12
    block_rgba: tuple = (0.85, 0.2, 0.15, 1)
    table_rgba: tuple = (0.85, 0.85, 0.82, 1)
    friction: float = 1.0
    timestep: float = 0.002
    context: bool = True                      # finite table, floor, person proxy
    floor_rgba: tuple = (0.35, 0.3, 0.27, 1)
    light: tuple = None                       # the asset's lights
    collision_geometry: str = "full"          # visible mesh hulls collide
    forearm: bool = False                     # floating hand without a forearm
    carry_height: float = .10                 # block carry height above its start (m)
    approach_height: float = .25              # hand approach/retreat height (m)
    camera_fovy: float = 55.


def build(cfg: SceneConfig = SceneConfig()):
    if cfg.collision_geometry != "full" or cfg.forearm or not cfg.context or cfg.light is not None:
        raise ValueError("Only the final scene is supported: full collision geometry, context, no forearm, asset lights")
    spec = mujoco.MjSpec.from_file(str(HAND_XML))
    spec.option.timestep = cfg.timestep
    spec.option.iterations = 50
    spec.option.ls_iterations = 50
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    for i, g in enumerate(spec.geoms):
        if g.name.startswith("collision_hand"):   # the asset's simplified proxies: off
            g.contype, g.conaffinity, g.condim = 0, 0, 3
            g.friction = [cfg.friction, 0.005, 0.0001]
            g.solref, g.solimp = CONTACT_SOLREF, CONTACT_SOLIMP
        elif g.type == mujoco.mjtGeom.mjGEOM_MESH and g.group == 0:
            # One physical mesh per link (group 1 is the duplicate visual copy); MuJoCo collides its convex hull.
            g.name = f"collision_visual_{i}"
            g.contype, g.conaffinity, g.condim = HAND_CT, 0, 3
            g.friction = [cfg.friction, 0.005, 0.0001]
            g.solref, g.solimp = CONTACT_SOLREF, CONTACT_SOLIMP
    for b in spec.bodies:
        if b.name.startswith("right_"):
            b.gravcomp = 1.0
    for light in spec.lights:  # no shadows: a hand-shaped shadow would leak the agent past the mask
        light.castshadow = False
    spec.body("right_hand_base").add_site(name="grasp", pos=GRASP_POS_BASE,
                                          quat=R_BASE_GRASP.as_quat(scalar_first=True), size=[0.005, 0, 0])

    wb = spec.worldbody
    # table top at z = 0; near edge 25 cm toward the person, 55 cm far side, +-65 cm sideways
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.40, 0.65, 0.02], pos=[0.15, 0, -0.02],
                contype=WORLD_CT, conaffinity=WORLD_CT | HAND_CT, friction=[cfg.friction, 0.005, 0.0001],
                rgba=cfg.table_rgba)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.01], pos=[0, 0, -0.75],
                rgba=cfg.floor_rgba, contype=0, conaffinity=0)
    ghost = dict(contype=0, conaffinity=0, rgba=(0.8, 0.65, 0.55, 1))
    wb.add_geom(name="person_torso", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.16, 0.27],
                pos=[-0.50, 0.03, 0.0], **ghost)
    wb.add_geom(name="person_head", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.09, 0, 0],
                pos=[-0.50, 0.05, 0.42], **ghost)
    half = np.array(cfg.block_size) / 2
    block = wb.add_body(name="block", pos=[0, 0, half[2]])
    block.add_freejoint(name="block_free")
    block.add_geom(name="block", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=cfg.block_mass,
                   rgba=cfg.block_rgba, contype=WORLD_CT, conaffinity=WORLD_CT | HAND_CT, condim=4,
                   friction=[cfg.friction, 0.005, 0.0001], solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP)
    model = spec.compile()
    model.vis.global_.fovy = cfg.camera_fovy
    return model


class Hand:
    """Index helpers and grasp-pose -> joint-target conversion for the floating Inspire hand."""

    def __init__(self, m):
        self.m = m
        self.wrist_act = np.array([m.actuator(f"{n}_position").id for n in WRIST])
        self.wrist_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in WRIST])
        self.finger_act = np.array([i for i in range(m.nu) if i not in self.wrist_act])
        self.finger_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in self.finger_act])
        self.finger_names = [m.actuator(i).name.removeprefix("right_").removesuffix("_position")
                             for i in self.finger_act]
        self.lo, self.hi = m.actuator_ctrlrange[self.finger_act].T
        self.block_qadr = m.jnt_qposadr[m.joint("block_free").id]
        self.grasp_site = m.site("grasp").id
        self.block_geom = m.geom("block").id
        self.table_geom = m.geom("table").id
        self.clearance_geoms = np.array([i for i in range(m.ngeom)
                                         if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[i] == 0])
        self.clearance_vertices = {}
        for g in self.clearance_geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                adr, num = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                self.clearance_vertices[g] = m.mesh_vert[adr:adr + num]

    def wrist_targets(self, pos, rot: Rotation):
        """Wrist joint values placing the grasp frame at world `pos` with orientation `rot`.

        The hand-base body rests at the origin with identity orientation, and its hinge chain is
        Rx Ry Rz (intrinsic), so a rotation about world x only changes the first angle.
        """
        base_rot = rot * R_BASE_GRASP.inv()
        base_pos = np.asarray(pos) - base_rot.apply(GRASP_POS_BASE)
        return np.r_[base_pos, base_rot.as_euler("XYZ")]

    def block_pose(self, d):
        q = d.qpos[self.block_qadr:self.block_qadr + 7]
        return q[:3].copy(), Rotation.from_quat(q[3:], scalar_first=True)

    def finger_q(self, f):
        """Finger targets from 12 fractions of each actuator's range."""
        return self.lo + np.clip(f, 0, 1) * (self.hi - self.lo)

    def table_clearance(self, d, distmax=0.1):
        """Signed distance of the hand's visible hulls to the table."""
        return min(mujoco.mj_geomDistance(self.m, d, self.table_geom, int(g), distmax, None)
                   for g in self.clearance_geoms)

    def minimum_geometry_z(self, d):
        """Lowest hand mesh vertex, ignoring table edges."""
        return float(min(d.geom_xpos[g, 2] + np.min(self.clearance_vertices[g] @ d.geom_xmat[g].reshape(3, 3)[2])
                         for g in self.clearance_geoms))
