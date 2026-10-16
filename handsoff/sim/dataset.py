"""Build the sim training set: 36 starts x the nine tasks, floating Inspire hand, phone-like cameras (MuJoCo part).

Stages, in order (each resumes; run with MUJOCO_GL=egl):
  physics  every start x task with the verified grip and motion (config/grip_v3.json); all must pass gently
  window   the model sees 8 frames at t = 3, 4, ..., 10 s (mid-roll included); the full trajectory is kept
  cameras  per start: a head phone (the person leans over the table, 50-75 deg down) and three free phones, one per
           third of MuJoCo azimuth 30-330 (never behind the person); redrawn until the block stays in view
  render   848x478 frames -> the real clips' start-centred crop -> 448 px RGB + agent masks (flat looks, diagnostic)
  tracks   block tracks from the rendered block masks with the real clips' features()
Whole starts belong to one fold: 30 train, 6 validation. The photoreal RGB that the model trains on is made from these
episodes by sim/photoreal.py.
Usage: python -m handsoff.sim.dataset --stage STAGE [--out DIR] [--limit N]
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

from handsoff.sim.geometry import camera_frame, project, quat_mat
from handsoff.sim.primitive import Action, placement_success, run, success, task_grip
from handsoff.sim.render import EYE, eye_camera
from handsoff.sim.scene import SceneConfig, build

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'intent_tasks_train_v3'
GRIP = ROOT/'config/grip_v3.json'
# Cameras behave like the recording phones: 848x478 landscape frames, then the same square crop as the real clips.
# Unknown phone settings are randomised.
PHONE = dict(width=848, height=478, hfov=(60., 75.), height_m=(.12, .28), distance=(.30, .45),
             # where the phone puts the block's start in the raw frame (0-1 across, 0-1 down): the range measured in
             # the 32 original phone views; 0.5, 0.5 = aimed at the block
             aim_x=(.32, .70), aim_y=(.33, .53),
             # head phone: the person leans over the table, 50-75 deg down; never inside/behind the
             # proxy head (x >= EYE x)
             head_elevation=(-75., -50.), head_distance=(.45, .60), head_azimuth_jitter=10.)
FREE_SECTORS = ((30., 130.), (130., 230.), (230., 330.))
IN_FRAME = .95   # every block corner of all nine motions (3-10 s) stays inside 95% of the phone frame
FIRST_VISIBLE = .9   # in every motion's first model frame the hand hides at most 10% of the block, as in the real
                     # clips (the crop is centred on the first-frame block); steep head views can look through the hand
SAMPLE_TIMES = list(range(3, 11))


def save(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def encode(path, frames, fps, gray=False):
    height, width = frames[0].shape[:2]
    tmp = path.with_name(path.stem+'.tmp.mp4')
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'gray' if gray else 'rgb24',
                          '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '1',
                          '-preset', 'fast', '-crf', '0' if gray else '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)],
                         stdin=subprocess.PIPE)
    for frame in frames:
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    tmp.replace(path)


def vertical_fovy(hfov):
    return float(np.degrees(2*np.arctan(np.tan(np.radians(hfov)/2)*PHONE['height']/PHONE['width'])))


def in_frame(points, cam):
    if (camera_frame(points, cam)[0][:, 2] <= 0).any():
        return False
    uv = project(points, cam) * [PHONE['height']/PHONE['width'], 1]   # project() scales both axes by fovy
    return bool(np.abs(uv).max() <= IN_FRAME)


def aim(cam, rng):
    """Turn the phone about its own position so the block starts off-centre, as in the real raw frames."""
    hfov = 2*np.degrees(np.arctan(np.tan(np.radians(cam['fovy'])/2)*PHONE['width']/PHONE['height']))
    forward = lambda az, el: np.array([np.cos(np.radians(el))*np.cos(np.radians(az)),  # noqa: E731
                                       np.cos(np.radians(el))*np.sin(np.radians(az)), np.sin(np.radians(el))])
    eye = np.asarray(cam['lookat']) - cam['distance']*forward(cam['azimuth'], cam['elevation'])
    azimuth = cam['azimuth'] + (rng.uniform(*PHONE['aim_x'])-.5)*hfov        # turn left: block moves right
    elevation = cam['elevation'] + (rng.uniform(*PHONE['aim_y'])-.5)*cam['fovy']   # tilt up: block moves down
    return dict(cam, azimuth=float(azimuth), elevation=float(elevation),
                lookat=(eye + cam['distance']*forward(azimuth, elevation)).tolist())


def phone_camera(rng, draw, points, clear, tries=1000):
    """Redraw the unknown settings until the block stays in view in all nine motions and is not hidden by the hand
    at the start, as the real phones were framed. Returns the camera and the number of draws."""
    for n in range(1, tries+1):
        cam = aim(dict(draw(), fovy=vertical_fovy(rng.uniform(*PHONE['hfov']))), rng)
        if in_frame(points, cam) and clear(cam):
            return cam, n
    raise ValueError('No phone framing keeps the block in view; widen the ranges explicitly')


def initialise(out, grip):
    record = json.loads(grip.read_text())
    if not record.get('gentle') or record.get('verified_cases') != 45:
        raise ValueError('Need the verified gentle grip')
    if record.get('motion') != 'clearance_side_human' or 'left_params' not in record:
        raise ValueError('Need the verified human-like roll-left grip')
    rng = np.random.default_rng(2026101201)
    groups = []
    for i in range(36):
        xy = rng.uniform(-.03, .03, 2).tolist()
        yaw = float(rng.uniform(-.25, .25))
        cameras = [dict(eye_camera(rng, (*xy, .04)), look=0)]   # lens and free cameras come after physics
        groups.append(dict(group=i, split='train' if i < 30 else 'validation', xy=xy, yaw=yaw, cameras=cameras))
    result = dict(schema=SCHEMA, seed=2026101201, episode_base=40000,
                  complete=False, physics_complete=False, render_complete=False, fps=10, resolution=448,
                  scene=SceneConfig(block_size=(.062, .038, .086), collision_geometry='full',
                                    forearm=False, carry_height=.10, approach_height=.25, camera_fovy=55.).__dict__,
                  embodiment='floating_hand_only', motion=record['motion'], grip=record,
                  grip_file=str(grip.relative_to(ROOT)), groups=groups, rows=[],
                  looks=[dict(block=[.80, .18, .12, 1], table=[.82, .82, .78, 1], floor=[.3, .3, .3, 1]),
                         dict(block=[.20, .40, .85, 1], table=[.65, .70, .65, 1], floor=[.45, .4, .35, 1])],
                  phone=PHONE, free_sectors=FREE_SECTORS, in_frame=IN_FRAME,
                  note='270 training + 54 validation episodes; whole start groups disjoint.')
    save(out/'manifest.json', result)
    return result


def physics(out, result, limit):
    cfg = SceneConfig(**result['scene']); m = build(cfg); start = time.time()
    for g in result['groups']:
        for di, dy in enumerate((-.15, 0, .15)):
            for ri, roll in enumerate((-np.pi/2, 0, np.pi/2)):
                ep = result['episode_base']+g['group']*9+di*3+ri; directory = out/f'ep{ep:05d}'; directory.mkdir(exist_ok=True)
                if any(r['episode'] == ep for r in result['rows']):
                    continue
                act = Action(dy=dy, roll=roll); gp = task_grip(result['grip'], roll)
                o = run(act, gp, cfg, block_xy=g['xy'], block_yaw=g['yaw'], model=m, record_hz=result['fps'],
                        motion=result['motion'])
                np.savez_compressed(directory/'traj.npz', **o.log)
                row = dict(episode=ep, case=directory.name, method='gentle', group=g['split'], start_group=g['group'],
                           action=act.__dict__, xy=g['xy'], yaw_rad=g['yaw'], success=placement_success(o, cfg), task_success=success(o, cfg),
                           displacement_mm=(o.disp*1000).tolist(), measured_roll_deg=float(np.rad2deg(o.roll)),
                           pos_error_mm=float(np.linalg.norm(o.disp[:2]-act.as_array()[:2])*1000),
                           peak_off_axis_error_deg=float(np.rad2deg(o.peak_off_axis_error)), min_table_clearance_mm=o.min_table_clearance*1000,
                           release_gap_mm=o.release_gap*1000, commanded_release_gap_mm=o.commanded_release_gap*1000,
                           touchdown_speed_m_s=o.touchdown_speed, max_lift_mm=o.max_lift*1000)
                result['rows'].append(row)
                meta = dict(episode=ep, seed=result['seed'], start_group=g['group'], split=g['split'], start=dict(xy=g['xy'], yaw=g['yaw']),
                            action=act.__dict__, scene=result['scene'], grip=gp.as_array().tolist(), motion=result['motion'], roll_frame='block',
                            outcome=dict(disp=o.disp.tolist(), roll=o.roll, max_lift=o.max_lift,
                                         orientation_error=o.orientation_error, off_axis_error=o.off_axis_error, peak_off_axis_error=o.peak_off_axis_error,
                                         min_table_clearance=o.min_table_clearance, release_gap=o.release_gap, touchdown_speed=o.touchdown_speed),
                            success=row['success'], cameras=g['cameras'], fps=result['fps'], frames=len(o.log['t']), render_complete=False)
                save(directory/'meta.json', meta); save(out/'manifest.json', result)
                print('physics', ep, row['success'], flush=True)
                if limit and len(result['rows']) >= limit:
                    return
    result.update(physics_complete=True, physics_seconds=time.time()-start)
    save(out/'manifest.json', result)
    if not all(r['success'] for r in result['rows']):
        raise ValueError('Failed motions retained; refuse to render')


def window(out, result, limit):
    """Keep the checked trajectory as full_traj.npz; traj.npz = the 8 model frames at 3, 4, ..., 10 s."""
    if result.get('render_complete') or not result.get('physics_complete'):
        raise ValueError('Need complete physics before the final sampling window')
    if result.get('model_frames_only'):
        return
    for row in result['rows']:
        ep = out/row['case']
        if not (ep/'full_traj.npz').exists():
            (ep/'traj.npz').replace(ep/'full_traj.npz')
        with np.load(ep/'full_traj.npz') as full:
            idx = np.array([np.argmin(abs(full['t']-t)) for t in SAMPLE_TIMES])
            sampled = {k: full[k][idx] for k in full.files}
            full_frames = len(full['t'])
        if not np.allclose(sampled['t'], SAMPLE_TIMES, atol=.003):
            raise ValueError('Unexpected model sampling times')
        np.savez_compressed(ep/'traj.npz', **sampled)
        meta = json.loads((ep/'meta.json').read_text())
        meta.update(frames=8, fps=1, full_frames=full_frames, clip_window_s=[3., 10.], full_motion_checked=True,
                    sample_times_s=sampled['t'].tolist(), sampling='Eight held model inputs, not a dense full-motion video')
        save(ep/'meta.json', meta)
    result.update(fps=1, model_frames_only=True, model_sample_times_s=SAMPLE_TIMES, clip_window_s=[3., 10.])
    save(out/'manifest.json', result)


def block_points(out, result, group):
    """Block corners for all nine motions of one start, every recorded step from 3 to 10 s."""
    half = np.asarray(result['scene']['block_size'])/2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*half
    points = []
    for k in range(9):
        with np.load(out/f"ep{result['episode_base']+group*9+k:05d}/full_traj.npz") as full:
            keep = (full['t'] >= 3-1e-6) & (full['t'] <= 10+1e-6)
            points += [p + corners @ quat_mat(q).T for p, q in zip(full['block_pos'][keep], full['block_quat'][keep])]
    return np.concatenate(points)


def cameras(out, result, limit):
    """Phone-like lens for the head eye plus three free phones, chosen after physics, before rendering."""
    if not result.get('model_frames_only') or result.get('render_complete'):
        raise ValueError('Phone cameras are chosen after physics/window and before rendering')
    if result.get('cameras_complete'):
        return
    rng = np.random.default_rng(2026101202); renderer = PhoneRenderer(result)
    for g in result['groups']:
        points = block_points(out, result, g['group'])
        firsts = [np.load(out/f"ep{result['episode_base']+g['group']*9+k:05d}/traj.npz")['qpos'][0] for k in range(9)]
        clear = lambda cam: min(renderer.block_visible(q, cam) for q in firsts) >= FIRST_VISIBLE  # noqa: E731
        head = g.setdefault('head_eye_camera', g['cameras'][0])   # the eye camera from initialise

        def draw_head():
            while True:
                cam = dict(head, azimuth=float(head['azimuth'] + rng.uniform(-1, 1)*PHONE['head_azimuth_jitter']),
                           elevation=float(rng.uniform(*PHONE['head_elevation'])), distance=float(rng.uniform(*PHONE['head_distance'])))
                el, az = np.radians(cam['elevation']), np.radians(cam['azimuth'])
                eye_x = cam['lookat'][0] - cam['distance']*np.cos(el)*np.cos(az)
                if eye_x >= EYE[0]:
                    return dict(cam, eye_x=float(eye_x))
        chosen = [phone_camera(rng, draw_head, points, clear)]
        for sector in FREE_SECTORS:
            def draw(sector=sector):
                height, distance = rng.uniform(*PHONE['height_m']), rng.uniform(*PHONE['distance'])
                return dict(kind='free', look=0, lookat=[*g['xy'], .04], azimuth=float(rng.uniform(*sector)),
                            elevation=float(-np.degrees(np.arcsin((height-.04)/distance))),
                            distance=float(distance), height_m=float(height))
            chosen.append(phone_camera(rng, draw, points, clear))
        g['cameras'] = [dict(c, framing_draws=n) for c, n in chosen]
        g['cameras'] += [dict(c, look=1) for c in g['cameras']]
        for k in range(9):
            path = out/f"ep{result['episode_base']+g['group']*9+k:05d}/meta.json"
            meta = json.loads(path.read_text()); meta['cameras'] = g['cameras']; save(path, meta)
    renderer.close()
    result['cameras_complete'] = True
    save(out/'manifest.json', result)


class PhoneRenderer:
    """848x478 RGB + agent/block segmentation, then the real-clip crop to the 448 model frame."""

    def __init__(self, result):
        import mujoco
        from handsoff.sim.render import agent_geoms
        self.result = result; self.m = m = build(SceneConfig(**result['scene']))
        m.vis.global_.offwidth, m.vis.global_.offheight = PHONE['width'], PHONE['height']
        self.d = mujoco.MjData(m)
        self.rgb = mujoco.Renderer(m, PHONE['height'], PHONE['width'])
        self.seg = mujoco.Renderer(m, PHONE['height'], PHONE['width']); self.seg.enable_segmentation_rendering()
        self.agent = np.zeros(m.ngeom+1, bool); self.agent[agent_geoms(m)] = True
        self.block = np.zeros(m.ngeom+1, bool); self.block[m.geom('block').id] = True

    def view(self, qpos, cam):
        """-> cropped rgb (8,448,448,3), agent mask, block mask, crop record, uncropped rgb."""
        import cv2
        import mujoco
        from handsoff.real.framing import RES, crop_frames
        from handsoff.sim.render import mjv_camera
        m, d = self.m, self.d; look = self.result['looks'][cam['look']]
        for geom in ('block', 'table', 'floor'):
            m.geom(geom).rgba[:] = look[geom]
        m.vis.global_.fovy = cam['fovy']; mj = mjv_camera(cam); frames, agents, blocks = [], [], []
        for q in qpos:
            d.qpos[:] = q; mujoco.mj_forward(m, d)
            self.rgb.update_scene(d, camera=mj); self.seg.update_scene(d, camera=mj)
            s = self.seg.render(); geom = np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)
            frames.append(self.rgb.render().copy()); agents.append(self.agent[geom]); blocks.append(self.block[geom])
        full = np.stack(frames); rgb, crop = crop_frames(full, np.stack(blocks))
        matrix = np.float32(crop['affine_source_to_crop'])
        warp = lambda masks: np.stack([cv2.warpAffine(x.astype(np.uint8), matrix, (RES, RES), flags=cv2.INTER_NEAREST) > 0  # noqa: E731
                                       for x in masks])
        return rgb, warp(agents), warp(blocks), crop, full

    def block_pixels(self, q, cam):
        import mujoco
        from handsoff.sim.render import mjv_camera
        self.m.vis.global_.fovy = cam['fovy']; self.d.qpos[:] = q; mujoco.mj_forward(self.m, self.d)
        self.seg.update_scene(self.d, camera=mjv_camera(cam)); s = self.seg.render()
        return int(self.block[np.where(s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, s[..., 0], -1)].sum())

    def block_visible(self, q, cam):
        """Fraction of the block's pixels not hidden by the hand (hand moved 10 m up for the reference)."""
        away = np.array(q, copy=True); away[2] += 10.   # qpos[0:3]: right_hand_base slide joints
        full = self.block_pixels(away, cam)
        return self.block_pixels(q, cam)/full if full else 0.

    def close(self):
        self.rgb.close(); self.seg.close()


def render(out, result, limit):
    from handsoff.real.framing import features
    if not result.get('cameras_complete'):
        raise ValueError('Choose phone cameras before rendering')
    renderer = PhoneRenderer(result); done = 0
    for row in result['rows']:
        ep = out/row['case']; meta = json.loads((ep/'meta.json').read_text())
        if meta.get('render_complete'):
            continue
        qpos = np.load(ep/'traj.npz')['qpos']; crops, tracks = [], []
        for v, cam in enumerate(meta['cameras']):
            rgb, agent, block, crop, _ = renderer.view(qpos, cam)
            encode(ep/f'v{v}_rgb.mp4', list(rgb), result['fps'])
            encode(ep/f'v{v}_mask.mp4', [x.astype(np.uint8)*255 for x in agent], result['fps'], gray=True)
            crops.append(crop); tracks.append([features(b) for b in block])
        np.savez_compressed(ep/'track.npz', track=np.array(tracks, np.float32))
        meta.update(crops=crops, render_complete=True); save(ep/'meta.json', meta)
        done += 1; print('render', row['case'], flush=True)
        if limit and done >= limit:
            break
    renderer.close()
    result['render_complete'] = all(json.loads((out/r['case']/'meta.json').read_text())['render_complete'] for r in result['rows'])
    result['complete'] = False   # tracks stage completes the dataset
    save(out/'manifest.json', result)


def tracks(out, result, limit):
    """Block tracks from rendered block masks with the real clips' features(), not from 3D projection."""
    if not result.get('render_complete'):
        raise ValueError('Render before collecting tracks')
    rec = dict(ep=[], view=[], track=[])
    for row in result['rows']:
        for v, track in enumerate(np.load(out/row['case']/'track.npz')['track']):
            rec['ep'].append(row['episode']); rec['view'].append(v); rec['track'].append(track)
    np.savez(out/'block_track.npz', **{k: np.array(v) for k, v in rec.items()})
    result.update(complete=True, track_source='rendered block masks after the real-clip crop; real.framing.features')
    save(out/'manifest.json', result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['physics', 'window', 'cameras', 'render', 'tracks'], required=True)
    ap.add_argument('--out', type=Path, default=ROOT/f'data/sim/{SCHEMA}')
    ap.add_argument('--grip', type=Path, default=GRIP)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    result = json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else initialise(out, a.grip.resolve())
    if result['schema'] != SCHEMA:
        raise ValueError('Cannot overwrite another dataset')
    {'physics': physics, 'window': window, 'cameras': cameras, 'render': render, 'tracks': tracks}[a.stage](out, result, a.limit)


if __name__ == '__main__':
    main()
