"""Cosmos-Transfer2.5 appearance variants of the development takes: two new looks (table colour, sleeve) per clip.
Used only for the control experiment in docs/tried_and_dropped.md; the final system uses no generated video.

Every generated video needs a full-video human review before any model use. Steps:
  specs   (CPU) one generation spec per original phone clip x look, with a prompt describing the take's task:
          COSMOS/specs/*.json + COSMOS/manifest.json. Generate with scripts/run_cosmos.sh COSMOS.
  review  (CPU) for each generated video: a side-by-side MP4 (original | generated, full length) and a 16-frame
          sheet in COSMOS/review/, and COSMOS/review_pending.json listing every video's hash. A reviewer watches each
          one; the decisions go into config/cosmos_review.json (status approved/rejected + the video's sha256).
Approved variants enter a dataset through `prepare inputs --cosmos COSMOS`, which also runs the automatic
task-preservation check below (block found at start and end, same place as in the original, distinct from the table).
Usage: python -m handsoff.real.cosmos {specs|review} --data DATA --cosmos COSMOS
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N, FPS, SIZE = 93, 16, 478   # the original clips: 93 frames over the take's common window, 478 px
LOOKS = [('grey_laminate', 'plain light grey laminate table', 'navy blue'),
         ('light_blue', 'plain matte light blue table', 'dark green')]
KEEP = dict(score=.8, iou=.7, delta_e=10)   # preservation thresholds, fixed before checking any variant


def save(path, value):
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def read(path):
    cap = cv2.VideoCapture(str(path)); frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or fps <= 0:
        raise ValueError(f'Invalid video: {path}')
    return frames, fps


def encode(path, frames, fps=FPS):
    """BGR frames -> H.264 MP4 (crf 16)."""
    tmp = path.with_name(path.stem + '.tmp.mp4'); h, w = frames[0].shape[:2]
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
                          '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast',
                          '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
    for frame in frames:
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('ffmpeg failed')
    tmp.replace(path)


def generated_path(cosmos, take_name, kind, look):
    return Path(cosmos)/'out'/f'{take_name}_{kind}_{look}.mp4'


def prompt(label, kind, surface, sleeve):
    translation, roll = label['translation'], label['roll']
    motion = ('picks up the block and puts it back upright near the same spot' if translation == 'stay'
              else f"picks up the block, moves it to the person's {translation} and places it upright")
    if roll != 'upright':
        motion = (f"picks up the block and rolls it a quarter turn to the person's {roll} around the block's initial "
                  'forward axis, placing it on its side near the starting spot')
    if label.get('exclude_from_pure_sideways_control'):
        motion += ", with some extra displacement along the block's initial local +y direction"
    return (f'A realistic {"first-person" if kind == "head" else "external side-camera"} video of a {surface} in soft daylight. '
            f"One small light wooden rectangular block is on the table. A person's right hand with a {sleeve} sleeve "
            f'{motion}, then withdraws. Preserve the input camera, block dimensions and complete motion. Soft shadows.')


def specs(a):
    (a.cosmos/'specs').mkdir(parents=True, exist_ok=True); rows = []
    for meta_path in sorted(a.data.glob('ep*/meta.json')):
        meta = json.loads(meta_path.read_text())
        for camera in meta['cameras']:
            if camera['generated']:
                continue
            for look, surface, sleeve in LOOKS:
                name = generated_path(a.cosmos, meta['take_name'], camera['kind'], look).stem
                spec = dict(name=name, video_path=str(ROOT/camera['source']), guidance=3, num_steps=4, seed=1,
                            prompt=prompt(meta['label'], camera['kind'], surface, sleeve), edge=dict(control_weight=1.0))
                save(a.cosmos/'specs'/f'{name}.json', spec)
                rows.append(dict(name=name, pair=f"{meta['pair']}:{camera['kind']}", look=look, original=camera['source'],
                                 original_sha256=camera['source_sha256'], expected_task=meta['label']['label']))
    save(a.cosmos/'manifest.json', dict(rows=rows))
    print(f"{len(rows)} specs in {a.cosmos/'specs'}; generate with scripts/run_cosmos.sh {a.cosmos}", flush=True)


def review(a):
    rows = json.loads((a.cosmos/'manifest.json').read_text())['rows']; (a.cosmos/'review').mkdir(exist_ok=True)
    pending, lines = {}, ['# Cosmos videos to review', '', 'Left: your original clip. Right: the generated look. Watch the whole '
                          'video: same task (pickup, turn, release), same block size, no extra objects.', '']
    for i, row in enumerate(rows, 1):
        path = Path(a.cosmos)/'out'/f"{row['name']}.mp4"
        if not path.exists():
            continue
        generated, gf = read(path); original, fps = read(ROOT/row['original'])
        frames = []
        for k, src in enumerate(original):
            frame = np.zeros((SIZE+64, SIZE*2, 3), np.uint8)
            frame[64:, :SIZE] = src
            frame[64:, SIZE:] = cv2.resize(generated[min(round(k/fps*gf), len(generated)-1)], (SIZE, SIZE))
            cv2.putText(frame, f"N{i:02d} {row['name']} | expected: {row['expected_task']}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(frame, 'ORIGINAL                                   COSMOS generated', (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            frames.append(frame)
        movie = a.cosmos/'review'/f"N{i:02d}_{row['name']}.mp4"; encode(movie, frames, fps)
        selected = [cv2.resize(frames[k], (478, 271)) for k in np.linspace(0, len(frames)-1, 16).round().astype(int)]
        cv2.imwrite(str(movie.with_suffix('.jpg')), np.vstack([np.hstack(selected[k:k+4]) for k in range(0, 16, 4)]))
        pending.setdefault(row['pair'], {})[row['look']] = dict(status='pending', source=str(path.relative_to(ROOT)),
                                                                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                                media_role='generated_rgb', review_id=f'N{i:02d}')
        lines.append(f"- N{i:02d} {row['expected_task']} ({row['name']}): {movie.resolve()}")
    save(a.cosmos/'review_pending.json', dict(schema='cosmos_human_review_v1', variants=pending))
    (a.cosmos/'review'/'README.md').write_text('\n'.join(lines) + '\n')
    print(f"{sum(len(v) for v in pending.values())} videos to review: {a.cosmos/'review'/'README.md'}", flush=True)


def lab(rgb):
    c = np.where(rgb > .04045, ((rgb+.055)/1.055)**2.4, rgb/12.92)
    xyz = c @ np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]]).T
    xyz /= np.array([.95047, 1., 1.08883])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787*xyz+16/116)
    return np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]), 200*(f[..., 1]-f[..., 2])], -1)


def delta_e(rgb, mask):
    """Colour difference between the block and a ring of table around it (CIE Lab)."""
    def dilate(n):
        return cv2.dilate(mask.astype(np.uint8), np.ones((2*n+1, 2*n+1), np.uint8)) > 0
    ring = dilate(20) & ~dilate(6)
    if not mask.any() or not ring.any():
        return 0.
    colour = lab(rgb/255.)
    return float(np.linalg.norm(colour[mask].mean(0)-colour[ring].mean(0)))


def preservation(rgb, baseline, model, proc):
    """Automatic task check of a generated clip against its original's first/last-frame block masks (baseline)."""
    from handsoff.real.sam3 import top_masks
    mask, score = top_masks(rgb[[0, -1]], 'wooden block', model, proc, .3, max_fraction=.15)
    bmask, bscore = baseline
    metrics = dict(score=score.tolist(), original_score=bscore.tolist(),
                   iou=[float((x & y).sum()/max((x | y).sum(), 1)) for x, y in zip(mask, bmask)],
                   dE=[delta_e(f, m) for f, m in zip(rgb[[0, -1]], bmask)])
    ok = (min(bscore) >= .8 and min(metrics['score']) >= KEEP['score'] and min(metrics['iou']) >= KEEP['iou']
          and min(metrics['dE']) >= KEEP['delta_e'])
    return metrics, bool(ok)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['specs', 'review'])
    ap.add_argument('--data', type=Path, help='prepared dataset with the original clips (specs)')
    ap.add_argument('--cosmos', type=Path, default=ROOT/'results/cosmos')
    a = ap.parse_args()
    {'specs': specs, 'review': review}[a.stage](a)
