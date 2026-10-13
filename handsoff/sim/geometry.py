"""Camera and pose maths shared by the MuJoCo and the photoreal (ovrtx) renderers; numpy only."""
import numpy as np

FOVY = 45.0   # MuJoCo global fovy, used when a camera has none


def quat_mat(q):
    """Unit quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Rotation (columns = local x, y, z in world) for a camera at eye looking along -z at target."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(x, f), -f])


def free_camera(c):
    """MuJoCo free camera (lookat, azimuth, elevation, distance in degrees/m) -> (eye, R)."""
    az, el = np.radians(c['azimuth']), np.radians(c['elevation'])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = np.asarray(c['lookat']) - c['distance'] * fwd
    return eye, look_at(eye, c['lookat'])


def project(P, cam):
    """World points [..., 3] -> normalised image coords [..., 2] in [-1, 1] (x right, y down), square image."""
    eye, R = free_camera(cam)                     # R columns: camera x (right), y (up), z (backwards) in world
    pc = (np.asarray(P) - eye) @ R
    f = 1 / np.tan(np.radians(cam.get('fovy', FOVY)) / 2)
    return np.stack([f * pc[..., 0] / -pc[..., 2], -f * pc[..., 1] / -pc[..., 2]], -1)


def camera_frame(P, cam):
    """World points [..., 3] -> camera frame [..., 3] (x right, y up, z = depth in front of the camera); also the
    world -> camera rotation [3, 3]."""
    eye, R = free_camera(cam)
    M = R.T * np.array([1, 1, -1])[:, None]      # rows: right, up, forward
    return (np.asarray(P) - eye) @ M.T, M
