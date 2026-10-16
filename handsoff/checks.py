"""Consistency checks on a sim dataset or on prepared real takes; exits with status 1 if any check fails.

The checks read only metadata (camera poses, crop records, labels) and take seconds. The scripts run them before
embedding or training, so that data which no longer matches the setup described in the README, for example after a
change to the camera or crop code, is rejected before a long GPU run.

  python -m handsoff.checks --sim DATA                          sim dataset (scripts/build_sim.sh, scripts/train.sh)
  python -m handsoff.checks --real DATA [--labels LABELS ...]   prepared real takes (scripts/run_test.sh)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def placement(cam, target):
    """Azimuth and elevation (MuJoCo convention, degrees) of the line from the camera's position to `target`, and
    the position itself. The stored azimuth is the viewing direction, which changes when a camera is re-aimed."""
    az, el = np.radians(cam['azimuth']), np.radians(cam['elevation'])
    eye = np.asarray(cam['lookat']) - cam['distance']*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    v = np.asarray(target, float) - eye
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]/np.linalg.norm(v))), eye


def episodes(data):
    return [json.loads(p.read_text()) for p in sorted(data.glob('ep*/meta.json'))]


def summary(problems, ok_message, shown=3):
    if not problems:
        return True, ok_message
    more = f' (+{len(problems) - shown} more)' if len(problems) > shown else ''
    return False, '; '.join(problems[:shown]) + more


# Sim dataset

def free_cameras(sim, manifest):
    """Free cameras stand anywhere except behind the person (azimuth 30-330 deg), and every start has cameras on the
    person's right, the far side and the person's left."""
    from handsoff.sim.dataset import FREE_SECTORS
    problems, used = [], []
    for g in manifest['groups']:
        az = [placement(c, [*g['xy'], .04])[0] for c in g['cameras'] if c['kind'] == 'free' and c['look'] == 0]
        used += az
        if any(not 30 <= a <= 330 for a in az):
            problems.append(f"start {g['group']}: a camera stands behind the person")
        if any(not any(lo <= a <= hi for a in az) for lo, hi in FREE_SECTORS):
            problems.append(f"start {g['group']}: free cameras miss the right, far or left side")
    if len(set(np.round(used, 3))) < 2:
        problems.append('all free cameras share one azimuth')
    return summary(problems, f'{len(used)} free cameras, each start covered from the right, far side and left')


def head_camera(sim, manifest):
    """The head camera looks 50-75 deg down at the block's start and stands in front of the person's head."""
    from handsoff.sim.render import EYE
    problems = []
    for g in manifest['groups']:
        head = [c for c in g['cameras'] if c['kind'] == 'head' and c['look'] == 0]
        if len(head) != 1:
            problems.append(f"start {g['group']}: {len(head)} head cameras"); continue
        _, down, eye = placement(head[0], g.get('head_eye_camera', head[0])['lookat'])
        if not -75 - 1e-6 <= down <= -50 + 1e-6:
            problems.append(f"start {g['group']}: head camera {-down:.0f} deg down")
        if eye[0] < EYE[0] - 1e-6:
            problems.append(f"start {g['group']}: head camera behind the person's head")
    return summary(problems, 'head cameras 50-75 deg down, in front of the head')


def block_in_view(sim, manifest):
    """The whole block stays in frame for every camera through all nine motions from its start."""
    from handsoff.sim.dataset import block_points, in_frame
    bad = []
    for g in manifest['groups']:
        points = block_points(sim, manifest, g['group'])
        bad += [f"start {g['group']} {c['kind']}" for c in g['cameras'] if c['look'] == 0 and not in_frame(points, c)]
    n = sum(1 for g in manifest['groups'] for c in g['cameras'] if c['look'] == 0)
    return summary([f'{b}: block leaves the frame' for b in bad], f'block in frame for all {n} cameras and 9 motions')


def photoreal(sim, manifest):
    """Training frames are the photoreal (ovrtx) renders with varied materials, not the flat MuJoCo renders."""
    looks = manifest.get('photoreal', {})
    if looks.get('renderer') != 'ovrtx' or not looks.get('complete'):
        return False, 'photoreal render missing or incomplete'
    n = len({json.dumps(look, sort_keys=True) for look in looks.get('looks', [])})
    return n >= 2, f'photoreal render, {n} distinct looks'


def sim_crop(sim, manifest):
    """Sim frames are 848x478 phone frames, cropped by the same code as the real clips, with block tracks from the
    same feature code."""
    from handsoff.real.framing import CROP_RULE
    if manifest.get('phone', {}).get('width') != 848 or manifest['phone'].get('height') != 478:
        return False, 'frames are not 848x478'
    problems = [f"ep{m['episode']}: crop missing or made by other code" for m in episodes(sim)
                if len(m.get('crops', [])) != len(m['cameras'])
                or any(c['rule'] != CROP_RULE or c['size_source_px'] < 478 for c in m['crops'])]
    if 'features' not in manifest.get('track_source', ''):
        problems.append('block tracks not computed with the real-clip feature code')
    return summary(problems, 'same crop and track code as the real clips')


# Prepared real takes

def labels(real, label_files):
    """Every take has a confirmed label, and the prepared label matches it."""
    known = {}
    for path in (ROOT/'config/labels_train.json', *label_files):
        for r in json.loads(Path(path).read_text())['annotations']:
            if r['pair'] in known:
                return False, f"take {r['pair']} labelled twice ({path})"
            known[r['pair']] = r
    problems = []
    for m in episodes(real):
        take = int(m.get('parent_episode', m['episode'])); r = known.get(take)
        if r is None or not r.get('user_confirmed'):
            problems.append(f'take {take}: label not confirmed')
        elif (r['translation'], r['roll']) != (m['label']['translation'], m['label']['roll']):
            problems.append(f'take {take}: prepared label differs from labels file')
    return summary(problems, 'every take has a confirmed label')


def real_crop(real, label_files):
    """Each phone view is cropped by the same code as the sim frames, centred on the block's start."""
    from handsoff.real.framing import CROP_RULE
    problems = [f"ep{m['episode']}: view not cropped by the shared code" for m in episodes(real)
                for c in m['cameras'] if not c.get('generated') and c.get('crop', {}).get('rule') != CROP_RULE]
    return summary(problems, 'same crop code as the sim frames, centred on the block start')


SIM_CHECKS = [free_cameras, head_camera, block_in_view, photoreal, sim_crop]
REAL_CHECKS = [labels, real_crop]


def failures(sim):
    """Failed sim checks as 'name: detail' strings (used by the training data loader)."""
    manifest = json.loads((sim/'manifest.json').read_text())
    return [f'{fn.__name__}: {detail}' for fn in SIM_CHECKS for ok, detail in [fn(sim, manifest)] if not ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sim', type=Path, action='append', default=[], help='sim dataset folder')
    ap.add_argument('--real', type=Path, help='prepared real takes folder')
    ap.add_argument('--labels', type=Path, action='append', default=[], help='labels file for new takes')
    a = ap.parse_args()
    runs = [(s, SIM_CHECKS, (s.resolve(), json.loads((s/'manifest.json').read_text()))) for s in a.sim]
    if a.real:
        runs.append((a.real, REAL_CHECKS, (a.real.resolve(), a.labels)))
    failed = False
    for data, checks, args in runs:
        print(f'== {data}')
        for fn in checks:
            ok, detail = fn(*args)
            failed |= not ok
            print(f"{'PASS' if ok else 'FAIL'} {fn.__name__}: {detail}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
