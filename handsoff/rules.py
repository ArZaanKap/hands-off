"""Check datasets against the user rules in RULES.md; exit 1 if any rule fails.

Every `auto: <check>` in RULES.md must have a function here (registered with @check); the script refuses to run
otherwise, so a rule cannot be written down and silently go unchecked. Chains run this before GPU/training stages.

Usage: check_rules.py [--sim DIR ...] [--real DIR] [--split FILE] [--labels FILE ...]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {}


def check(name, applies):
    def register(fn):
        CHECKS[name] = (applies, fn)
        return fn
    return register


def registry():
    """{check name: rule IDs} from RULES.md; fail if an auto rule names a check that does not exist."""
    rules = {}
    for line in (ROOT/'docs/decisions.md').read_text().splitlines():
        row = re.match(r'\|\s*(R\d+)\s*\|.*\|\s*auto:\s*([a-z_]+)', line)
        if row:
            rules.setdefault(row.group(2), []).append(row.group(1))
    missing = sorted(set(rules) - set(CHECKS))
    unused = sorted(set(CHECKS) - set(rules))
    if missing or unused:
        sys.exit(f'RULES.md and check_rules.py disagree: no check for {missing}; check without a rule {unused}')
    return rules


def placement(cam, target):
    """Where the camera stands, independent of aiming: MuJoCo azimuth/elevation of the line from the camera to the
    point it was placed to look at (the block's start, or the head camera's original look-at point), and its position (aiming turns a camera about its own position, so its stored azimuth is
    the view direction, not where it stands)."""
    az, el = np.radians(cam['azimuth']), np.radians(cam['elevation'])
    eye = np.asarray(cam['lookat']) - cam['distance']*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    v = np.asarray(target, float) - eye
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]/np.linalg.norm(v))), eye


def sim_metas(sim):
    return [json.loads(p.read_text()) for p in sorted(sim.glob('ep*/meta.json'))]


@check('sim_roll_frame', 'sim')
def _(sim, **_):
    bad = [m['episode'] for m in sim_metas(sim) if m.get('roll_frame') != 'block']
    return not bad, f'{len(bad)} episodes without block-frame roll' if bad else 'all episodes use block-frame roll'


@check('sim_no_forearm', 'sim')
def _(sim, manifest, **_):
    bad = [m['episode'] for m in sim_metas(sim) if m['scene'].get('forearm', True)]
    ok = not manifest['scene'].get('forearm', True) and not bad
    return ok, 'no forearm' if ok else f'forearm present (manifest or {len(bad)} episodes)'


@check('sim_free_cameras', 'sim')
def _(manifest, **_):
    """Free cameras: MuJoCo azimuth 30-330 (none 150-210 deg behind the person) and spread over that range."""
    from handsoff.sim.dataset import FREE_SECTORS
    problems, used = [], []
    for g in manifest['groups']:
        az = [placement(c, [*g['xy'], .04])[0] for c in g['cameras'] if c['kind'] == 'free' and c['look'] == 0]
        used += az
        if any(not 30 <= a <= 330 for a in az):
            problems.append(f"start {g['group']}: camera behind the person")
        if any(not any(lo <= a <= hi for a in az) for lo, hi in FREE_SECTORS):
            problems.append(f"start {g['group']}: free cameras do not cover the person's right, far side and left")
    if len(set(np.round(used, 3))) < 2:
        problems.append('every free camera sits at the same azimuth (not spread over the range)')
    return not problems, '; '.join(problems[:4]) + (f' (+{len(problems)-4} more)' if len(problems) > 4 else '') \
        if problems else f'{len(used)} free cameras within 30-330, every start covers right/far/left'


@check('sim_head_camera', 'sim')
def _(manifest, **_):
    """Head camera stands 50-75 deg above the block's start (measured from where it stands, not where it is aimed)
    and in front of the person's head."""
    from handsoff.sim.render import EYE
    problems = []
    for g in manifest['groups']:
        head = [c for c in g['cameras'] if c['kind'] == 'head' and c['look'] == 0]
        if len(head) != 1:
            problems.append(f"start {g['group']}: {len(head)} head cameras"); continue
        _, down, eye = placement(head[0], g.get('head_eye_camera', head[0])['lookat'])
        if not -75 - 1e-6 <= down <= -50 + 1e-6:
            problems.append(f"start {g['group']}: head looks {-down:.0f} deg down at the block")
        if eye[0] < EYE[0] - 1e-6:
            problems.append(f"start {g['group']}: head camera not in front of the person's head")
    return not problems, '; '.join(problems[:3]) + (f' (+{len(problems)-3} more)' if len(problems) > 3 else '') \
        if problems else 'every head camera 50-75 deg above the block, in front of the head'


@check('sim_block_in_view', 'sim')
def _(sim, manifest, **_):
    if 'phone' not in manifest:
        return False, 'no phone-like cameras recorded (block-in-view rule not applied)'
    from handsoff.sim.dataset import block_points, in_frame
    bad = []
    for g in manifest['groups']:
        points = block_points(sim, manifest, g['group'])
        cams = [c for c in g['cameras'] if c['look'] == 0]
        bad += [(g['group'], c['kind']) for c in cams if not in_frame(points, c) or 'framing_draws' not in c]
    return not bad, f'{len(bad)} cameras lose the block: {bad[:4]}' if bad else \
        f"all {sum(len(g['cameras'])//2 for g in manifest['groups'])} cameras keep the block in view in all 9 motions"


@check('sim_photoreal', 'sim')
def _(manifest, **_):
    looks = manifest.get('photoreal', {})
    if looks.get('renderer') != 'ovrtx' or not looks.get('complete'):
        return False, 'model RGB is not photoreal ovrtx (flat MuJoCo looks are diagnostic only)'
    n = len({json.dumps(l, sort_keys=True) for l in looks.get('looks', [])})
    return n >= 2, f"ovrtx photoreal RGB, {n} distinct looks"


@check('sim_human_like_grip', 'sim')
def _(manifest, **_):
    """R20: at the grasp, the fingers point away from the person (+x) for all nine tasks."""
    from handsoff.sim.primitive import grasp_frame, grasp_variant, task_grip
    bad = []
    for roll in (-np.pi/2, 0., np.pi/2):
        gp, bank = grasp_variant(task_grip(manifest['grip'], roll), roll)
        _, rot = grasp_frame(gp, np.zeros(3), 0., bank=bank)
        if rot.apply([1, 0, 0])[0] < .5:
            bad.append({-1: 'roll right', 0: 'no roll', 1: 'roll left'}[int(np.sign(roll))])
    return not bad, f"robot fingers point toward the person (reaches like someone opposite) for: {', '.join(bad)}" if bad else \
        'robot grips from the person\'s side for every task'


@check('same_crop_code', 'sim real')
def _(sim=None, manifest=None, real=None, **_):
    from handsoff.real.framing import CROP_RULE
    problems, notes = [], []
    if sim is not None:
        if manifest.get('phone', {}).get('width') != 848 or manifest['phone'].get('height') != 478:
            problems.append('sim frames are not 848x478 phone frames')
        elif not manifest.get('render_complete'):
            notes.append('sim not rendered yet: crops not checked')
        else:
            for m in sim_metas(sim):
                crops = m.get('crops', [])
                if len(crops) != len(m['cameras']) or any(c['rule'] != CROP_RULE or c['size_source_px'] < 478 for c in crops):
                    problems.append(f"ep{m['episode']}: crop missing or not the real-clip crop"); break
            if 'features' not in manifest.get('track_source', ''):
                problems.append('sim tracks not from rendered masks with clip_framing.features')
    if real is not None:
        for p in sorted(real.glob('ep*/meta.json')):
            for c in json.loads(p.read_text())['cameras']:
                if not c.get('generated') and c.get('crop', {}).get('rule') != CROP_RULE:
                    problems.append(f'{p.parent.name}: original view without the start-centred real-clip crop')
    return not problems, '; '.join(problems[:3]) if problems else \
        ('same start-centred crop code on sim and real' + (f" ({'; '.join(notes)})" if notes else ''))


@check('real_cosmos_approved', 'real')
def _(real, **_):
    from handsoff.real.review_gate import require_dataset_approval
    try:
        require_dataset_approval(real)
    except Exception as error:   # the gate raises with the offending clip
        return False, str(error)
    return True, 'every generated view has hash-bound user approval'


def trainer_source():
    from handsoff.model import train
    return Path(train.__file__).read_text()


@check('trainer_all_views', 'code')
def _(**_):
    source = trainer_source()
    ok = "views = np.arange(D['look'].shape[1])" in source and 'held_out(' not in source and '--held-out' not in source
    return ok, 'the trainer draws from every sim camera' if ok else 'the trainer leaves some sim cameras out'


@check('trainer_real_labels', 'code')
def _(**_):
    source = trainer_source()
    ok = 'load_queries' not in source and 'real_x' not in source and 'alignment' not in source
    return ok, 'the final head trains on sim clips only (no real-video loss)' if ok else \
        'the trainer has a real-video loss (the final head is sim-only)'


def annotations(extra=()):
    """Confirmed take labels: demos3 (D01-D17) plus any test-take label files (ids from 100, never colliding)."""
    out = {}
    for path in (ROOT/'config/labels_train.json', *extra):
        for r in json.loads(Path(path).read_text())['annotations']:
            if r['pair'] in out:
                sys.exit(f"take {r['pair']} labelled twice ({path})")
            out[r['pair']] = r
    return out


@check('real_labels_confirmed', 'real')
def _(real, labels, **_):
    ann = annotations(labels); problems = []
    for p in sorted(real.glob('ep*/meta.json')):
        m = json.loads(p.read_text()); e = int(m.get('parent_episode', m['episode'])); a = ann.get(e)
        if a is None or not a.get('user_confirmed'):
            problems.append(f'D{e:02d} not user-confirmed')
        elif (a['translation'], a['roll']) != (m['label']['translation'], m['label']['roll']):
            problems.append(f'D{e:02d} prepared label differs from the confirmed label')
    return not problems, '; '.join(problems[:4]) if problems else 'every real take label is user-confirmed'


@check('real_split', 'real')
def _(real, split, labels, **_):
    """R08 (changed 12 Oct): whole takes, one fold each; every approved recorded take trains; the test set is new
    recordings that cover every task type (checked once they are added). Takes numbered 100+ in no fold are new
    recordings (e.g. someone else's): they pass, as the final head is sim-only (trainer_real_labels)."""
    s = json.loads(split.read_text())['folds']; ann = annotations(labels); problems = []
    member = {}
    for fold, takes in s.items():
        for t in takes:
            if t in member:
                problems.append(f'D{t:02d} in {member[t]} and {fold}')
            member[t] = fold
    takes = {int(json.loads(p.read_text()).get('parent_episode', int(p.parent.name[2:]))) for p in real.glob('ep*/meta.json')}
    new = sorted(t for t in takes - set(member) if t >= 100)
    problems += [f'D{t:02d} has no fold' for t in sorted(takes - set(member) - set(new))]
    problems += [f'D{t:02d} is in {member[t]}, not train' for t in sorted(takes) if t in member and member[t] != 'train' and t <= 17]
    task = lambda t: (ann[t]['translation'], ann[t]['roll'])
    if s['test']:
        every = {(m, r) for m in ('right', 'stay', 'left') for r in ('right', 'upright', 'left')}
        missing = every - {task(t) for t in s['test'] if t in ann}
        if missing:
            problems.append(f'test lacks task types {sorted(missing)}')
    note = f"{len(s['test'])} test takes" if s['test'] else 'test = 14 Oct recordings, not in this file'
    note += f"; {len(new)} new recordings outside the split (never trained on)" if new else ''
    return not problems, '; '.join(problems[:4]) if problems else \
        f"split by take: {len(s['train'])} train / {len(s['validation'])} validation; {note}"


@check('trainer_fixed_epochs', 'code')
def _(**_):
    source = trainer_source()
    ok = "if epoch+1 == a.epochs:\n            atomic_torch(a.out/'head.pt'" in source and 'best' not in source
    return ok, 'head.pt = final epoch of a fixed schedule' if ok else 'the trainer can pick an epoch by validation'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sim', type=Path, action='append', default=[])
    ap.add_argument('--real', type=Path)
    ap.add_argument('--split', type=Path, default=ROOT/'config/real_split.json')
    ap.add_argument('--labels', type=Path, action='append', default=[], help='test-take labels (e.g. RAW/labels.json)')
    a = ap.parse_args(); rules = registry(); failed = False
    targets = [('code', dict(code='training code'))]   # code rules apply to every run
    targets += [('sim', dict(sim=s.resolve(), manifest=json.loads((s/'manifest.json').read_text()))) for s in a.sim]
    if a.real:
        targets.append(('real', dict(real=a.real.resolve(), split=a.split, labels=a.labels)))
    for kind, args in targets:
        print(f"== {kind}: {args.get('sim') or args.get('real') or args.get('code')}")
        for name, (applies, fn) in CHECKS.items():
            if kind in applies.split():
                ok, detail = fn(**args)
                failed |= not ok
                print(f"{'PASS' if ok else 'FAIL'} {'/'.join(rules[name])} {name}: {detail}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
