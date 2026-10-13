"""MuJoCo cameras and rendering with agent masks (every pixel of the robot hand or the person proxy).

The head camera starts at the person proxy's eyes looking at the block's start; the dataset then turns it into a
phone-like camera (see dataset.PHONE). Masks come from MuJoCo's segmentation pass.
"""
import os

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# The front of the person proxy's head sphere (centre (-0.50, 0.05, 0.42), radius 0.09).
EYE = (-0.40, 0.05, 0.42)


def eye_camera(rng, lookat):
    """Head camera at EYE looking at the block's start (jittered by 1 cm)."""
    la = np.asarray(lookat) + rng.normal(0, 0.01, 3)
    f = la - np.asarray(EYE)
    dist = float(np.linalg.norm(f))
    return dict(kind='head', lookat=la.tolist(), azimuth=float(np.degrees(np.arctan2(f[1], f[0]))),
                elevation=float(np.degrees(np.arcsin(f[2] / dist))), distance=dist)


def mjv_camera(c):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = c['lookat']
    cam.azimuth, cam.elevation, cam.distance = c['azimuth'], c['elevation'], c['distance']
    return cam


def agent_geoms(m):
    """Geom ids of the agent: every geom under right_hand_base and the person proxy."""
    root = m.body('right_hand_base').id
    under = lambda b: b == root or (b != 0 and under(m.body_parentid[b]))  # noqa: E731
    return np.array([g for g in range(m.ngeom) if under(m.geom_bodyid[g]) or m.geom(g).name.startswith('person')])


class MaskedRenderer:
    """Square RGB + segmentation renderer pair; `render(d, cam)` -> (rgb uint8 HxWx3, agent mask bool HxW)."""

    def __init__(self, m, size):
        self.rgb = mujoco.Renderer(m, size, size)
        self.seg = mujoco.Renderer(m, size, size)
        self.seg.enable_segmentation_rendering()
        self.agent = np.zeros(m.ngeom + 1, bool)
        self.agent[agent_geoms(m)] = True

    def close(self):
        self.rgb.close()
        self.seg.close()

    def render(self, d, cam):
        self.rgb.update_scene(d, camera=cam)
        self.seg.update_scene(d, camera=cam)
        img = self.rgb.render()
        s = self.seg.render()
        geom = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
        return img, self.agent[geom]   # geom -1 indexes the trailing False
