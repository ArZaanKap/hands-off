"""Photoreal (ovrtx) RGB for the sim dataset, reusing its exact poses, cameras and crops.

The flat MuJoCo render (sim/dataset.py) fixed every phone camera and its crop window. Here:
  export (.venv-spider, CPU): recorded qpos -> ep*/photoreal/scene.usda.gz, 848x478 cameras with each lens.
  render (.venv-render, GPU): two random looks per start (HDRI/room, table/block/floor materials, sun), shared by all
      nine motions of that start so composition operands match; a look with a mostly dark view is redrawn on the
      start's first motion. Writes uncropped ep*/photoreal/full_v{i}_rgb.mp4 + agent masks.
  crop (.venv-spider, CPU): the same crop windows as the flat render -> v{i}_rgb.mp4 / v{i}_mask.mp4; the flat videos
      move to <data>_flat (diagnostic). Records the agent-mask IoU between photoreal and flat renders (camera
      conversion check) and manifest['photoreal'].
Usage: python -m handsoff.sim.photoreal {export|render|crop} [--data DIR] [--limit N] [--groups 0 7 ...]
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/sim/intent_tasks_train_v3'
FLAT = ROOT/'data/sim/intent_tasks_train_v3_flat'
SEED = 2026101203


def save(path, value):
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=1)); os.replace(tmp, path)


def episodes(manifest, groups=None):
    for row in manifest['rows']:
        if groups is None or row['start_group'] in groups:
            yield DATA/row['case'], row['start_group']


def views0(meta):
    """Look-0 cameras; both looks share camera poses, so v{look*4+k} uses camera k."""
    cams = [c for c in meta['cameras'] if c['look'] == 0]
    assert len(cams) * 2 == len(meta['cameras'])
    return cams


def export(a, manifest):
    from handsoff.sim import ovrtx
    from handsoff.sim.scene import SceneConfig, build
    phone = manifest['phone']; m = build(SceneConfig(**manifest['scene'])); done = 0
    for ep, _ in episodes(manifest, a.groups):
        out = ep/'photoreal'/'scene.usda.gz'
        if out.exists():
            continue
        meta = json.loads((ep/'meta.json').read_text())
        usda, info = ovrtx.scene_usda(m, np.load(ep/'traj.npz')['qpos'], views0(meta), size=(phone['width'], phone['height']))
        out.parent.mkdir(exist_ok=True); tmp = out.with_suffix('.tmp')
        with gzip.open(tmp, 'wt', compresslevel=3) as f:
            f.write(usda)
        os.replace(tmp, out); save(ep/'photoreal'/'export.json', info); done += 1
        if a.limit and done >= a.limit:
            break
    print('exported', done, flush=True)


def render(a, manifest):
    import imageio.v2 as imageio
    from handsoff.sim import ovrtx
    phone = manifest['phone']; size = (phone['width'], phone['height'])
    looks_path = DATA/'photoreal_looks.json'
    chosen = json.loads(looks_path.read_text()) if looks_path.exists() else {}
    R = ovrtx.Renderer((DATA/"ovrtx.log").resolve()); t0 = time.time(); done = 0
    for ep, group in episodes(manifest, a.groups):
        pr = ep/'photoreal'
        if (pr/'done.json').exists() or not (pr/'scene.usda.gz').exists():
            continue
        with gzip.open(pr/'scene.usda.gz', 'rt') as f:
            usda = f.read()
        V = len(views0(json.loads((ep/'meta.json').read_text()))); views = list(range(V)); te = time.time(); record = []
        for li in range(2):
            key = f'{group}/{li}'
            for attempt in range(ovrtx.MAX_TRIES if key not in chosen else 1):
                look = chosen.get(key) or dict(ovrtx.sample_look(np.random.default_rng([SEED, group, li, attempt])), attempt=attempt)
                r = R.open(ovrtx.look_stage(usda, look, views, size))
                rgb, seg = [[] for _ in views], [[] for _ in views]
                for t in range(8):
                    r.update_from_usd_time(float(t))
                    for k, (c, s) in enumerate(ovrtx.grab(r, views, 1, a.steps)):
                        rgb[k].append(c); seg[k].append(s >= 2)
                rgb = np.array(rgb, np.uint8); dark = (rgb.mean(axis=(2, 3, 4)) < ovrtx.DARK_MEAN).mean(1)
                if key in chosen or dark.max() <= .5:
                    break
                print(f'  start {group} look {li} attempt {attempt}: dark views {np.flatnonzero(dark > .5).tolist()}', flush=True)
            if key not in chosen:   # first motion of the start fixes the look for all nine
                chosen[key] = look; save(looks_path, chosen)
            for k in views:
                imageio.mimwrite(pr/f'full_v{li*V+k}_rgb.mp4', rgb[k], fps=1, quality=9, macro_block_size=2)
                np.savez_compressed(pr/f'full_v{li*V+k}_agent.npz', mask=np.packbits(np.array(seg[k])), shape=np.array(np.shape(seg[k])))
            record.append(dict(look=li, dark_fraction=dark.round(3).tolist()))
        save(pr/'done.json', dict(looks=[chosen[f'{group}/{li}'] for li in range(2)], views=record,
                                  seconds=time.time()-te, renderer='ovrtx', steps=a.steps))
        done += 1
        if done <= 3 or done % 25 == 0:
            print(f'{done} episodes, {time.time()-t0:.0f}s ({time.time()-te:.1f}s this one)', flush=True)
        if a.limit and done >= a.limit:
            break
    if R.r is not None:
        R.r.destroy()
    print('rendered', done, flush=True)


def crop(a, manifest):
    import cv2
    import imageio.v3 as iio
    from handsoff.real.framing import RES
    from handsoff.sim.dataset import encode
    ious, done = [], 0
    for ep, _ in episodes(manifest, a.groups):
        pr, meta = ep/'photoreal', json.loads((ep/'meta.json').read_text())
        if meta.get('photoreal_complete') or not (pr/'done.json').exists():
            continue
        flat = FLAT/ep.name; flat.mkdir(parents=True, exist_ok=True)
        for v, cam_crop in enumerate(meta['crops']):
            for kind in ('rgb', 'mask'):
                if not (flat/f'v{v}_{kind}.mp4').exists():
                    shutil.copy2(ep/f'v{v}_{kind}.mp4', flat/f'v{v}_{kind}.mp4')
            matrix = np.float32(cam_crop['affine_source_to_crop'])
            rgb = iio.imread(pr/f'full_v{v}_rgb.mp4')
            z = np.load(pr/f'full_v{v}_agent.npz'); agent = np.unpackbits(z['mask'])[:np.prod(z['shape'])].reshape(z['shape']).astype(bool)
            rgb = [cv2.warpAffine(f, matrix, (RES, RES), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)) for f in rgb]
            agent = np.stack([cv2.warpAffine(x.astype(np.uint8), matrix, (RES, RES), flags=cv2.INTER_NEAREST) > 0 for x in agent])
            flat_agent = iio.imread(flat/f'v{v}_mask.mp4'); flat_agent = (flat_agent[..., 0] if flat_agent.ndim == 4 else flat_agent) > 127
            both, either = (agent & flat_agent).sum(), (agent | flat_agent).sum()
            ious.append(float(both/either) if either else 1.)
            encode(ep/f'v{v}_rgb.mp4', rgb, 1)
            encode(ep/f'v{v}_mask.mp4', [x.astype(np.uint8)*255 for x in agent], 1, gray=True)
        done_info = json.loads((pr/'done.json').read_text())
        meta.update(photoreal_complete=True, looks=done_info['looks'], flat_render=str(flat.relative_to(ROOT)))
        save(ep/'meta.json', meta); done += 1
    rows = [json.loads((DATA/r['case']/'meta.json').read_text()) for r in manifest['rows']]
    complete = all(m.get('photoreal_complete') for m in rows)
    looks = json.loads((DATA/'photoreal_looks.json').read_text())
    manifest['photoreal'] = dict(renderer='ovrtx', complete=complete, looks=list(looks.values()), seed=SEED,
                                 flat_render=str(FLAT.relative_to(ROOT)),
                                 note='Model RGB/masks from ovrtx; crops/tracks from the flat MuJoCo render of the same geometry.')
    if ious:
        manifest['photoreal'].setdefault('agent_iou_vs_flat', []).append(
            dict(views=len(ious), min=min(ious), median=float(np.median(ious))))
    save(DATA/'manifest.json', manifest)
    print(f'cropped {done} episodes; agent-mask IoU photoreal vs flat: min {min(ious) if ious else float("nan"):.3f} '
          f'median {np.median(ious) if ious else float("nan"):.3f}; complete={complete}', flush=True)


def main():
    global DATA, FLAT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['export', 'render', 'crop'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--groups', type=int, nargs='*')
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--data', type=Path, default=DATA, help='dataset; flat videos go to <data>_flat')
    a = ap.parse_args()
    DATA = a.data.resolve(); FLAT = DATA.parent/(DATA.name+'_flat'); manifest = json.loads((DATA/'manifest.json').read_text())
    if not manifest.get('render_complete') or not manifest.get('cameras_complete'):
        sys.exit('Need the complete flat render (cameras and crop windows) first')
    dict(export=export, render=render, crop=crop)[a.stage](a, manifest)


if __name__ == '__main__':
    main()
