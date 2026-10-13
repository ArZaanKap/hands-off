"""Phone takes -> model inputs (training and test takes alike): two synced phones, agent masked, block tracked.

  sync    (CPU, .venv-spider) audio offset between the two phones, their common window, a full side-by-side MP4 and a
          24-frame motion sheet per take for the user's review, and a labels file to fill in (never overwritten).
  inputs  (GPU, .venv-mask; refuses takes without user-confirmed labels, R09) per phone: 8 samples over the window
          -> SAM3 block -> square crop centred on the block's start (R10) -> 93 cropped frames at 478 px (the clip
          Cosmos sees) -> 8 frames at 448 -> person/hand/arm mask + block track. With --cosmos, the take's
          user-approved Cosmos variants (R01) that pass the automatic preservation check are added as extra views.
Raw layout: RAW/arz{N}.mp4 (head phone) + RAW/del{N}.mp4 (second phone). Take id = --id-offset + N: training takes
D01-D17 use offset 0, test takes offset 100 (T101...).
Usage: python -m handsoff.real.prepare sync --raw RAW --review DIR [--id-offset 100]
       python -m handsoff.real.prepare inputs --raw RAW --review DIR --out DATA [--labels FILE] [--tag NAME] [--cosmos DIR]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from handsoff.real.cosmos import LOOKS, N, SIZE, encode, generated_path
from handsoff.real.framing import crop_frames, features

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = dict(schema='approved_demos3_v3', frames=8, resolution=448, prompts=['person', 'hand', 'arm'], threshold=.4,
                dilation_px=7, block_prompt='wooden block', max_block_fraction=.15, max_agent_overlap=.5,
                window='common take from the audio-envelope offset, 0.2 s in from each end',
                crop='SAM3 wooden block on 8 window samples, threshold .4 -> framing.crop_frames',
                block_centroid_envelope='endpoint centroid bounds plus twice endpoint bbox width/height',
                reject_boundary_block_masks=True, sampling='8 uniform frame indices from exactly 93 same-window frames')
MODEL_FPS = 16*8/93


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take_name(take):
    return f'D{take:02d}' if take < 100 else f'T{take}'


def numbers(raw):
    found = lambda prefix: {int(re.fullmatch(prefix + r'(\d+)', p.stem).group(1)) for p in raw.glob(prefix + '*.mp4')  # noqa: E731
                            if re.fullmatch(prefix + r'(\d+)', p.stem)}
    head, second = found('arz'), found('del')
    if head != second:
        raise ValueError(f'Unpaired files: head only {sorted(head-second)}, second only {sorted(second-head)}')
    return sorted(head)


def paired_mp4(head, second, offset, path):
    """Both phones side by side, aligned by the audio offset (second time = head time + offset), full length."""
    trim = lambda p, t: (['-ss', f'{t:.3f}'] if t > 0 else []) + ['-i', str(p)]  # noqa: E731
    tmp = path.with_name(path.stem + '.tmp.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', *trim(head, -offset), *trim(second, offset),
                    '-filter_complex', '[0:v]scale=-2:478,setsar=1[a];[1:v]scale=-2:478,setsar=1[b];[a][b]hstack',
                    '-an', '-shortest', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
                    '-threads', '2', str(tmp)], check=True)
    tmp.replace(path)


def sync(a):
    from handsoff.real.sync import audio, compare, frame, probe, tile
    a.review.mkdir(parents=True, exist_ok=True); rows = []
    for n in numbers(a.raw):
        head, second = a.raw/f'arz{n}.mp4', a.raw/f'del{n}.mp4'
        ph, ps = probe(head), probe(second)
        offset, score = compare(audio(head)[1], audio(second)[1], 100)
        lo = max(0, -offset)+.2; hi = min(ph['duration_s'], ps['duration_s']-offset)-.2
        take = a.id_offset+n; name = take_name(take)
        rows.append(dict(take=take, number=n, head=ph, second=ps, envelope_offset_s=offset, envelope_score=score,
                         window_head_s=[lo, hi]))
        paired_mp4(head, second, offset, a.review/f'{name}_both_phones.mp4')
        ch, cs = cv2.VideoCapture(str(head)), cv2.VideoCapture(str(second)); tiles = []
        for t in np.linspace(lo, hi, 24):
            tiles.append(np.hstack([tile(frame(ch, t), f'{name} head {t:.2f}s'), tile(frame(cs, t+offset), f'second {t+offset:.2f}s')]))
        ch.release(); cs.release()
        cv2.imwrite(str(a.review/f'{name}_motion.jpg'), np.vstack([np.hstack(tiles[j:j+3]) for j in range(0, 24, 3)]))
        print(f'{name}: offset {offset:+.2f} s, envelope score {score:.2f}, window {hi-lo:.1f} s', flush=True)
    atomic_json(a.review/'pairing.json', dict(takes=rows, convention='second time = head time + envelope_offset_s',
                                              id_offset=a.id_offset, raw=str(a.raw.relative_to(ROOT))))
    if not a.labels.exists():
        atomic_json(a.labels, dict(schema='test_take_labels', annotations=[
            dict(pair=r['take'], head_source=r['head']['source'], second_source=r['second']['source'],
                 translation=None, roll=None, label=None, user_confirmed=False, notes='') for r in rows]))
    print(f'{len(rows)} takes; review {a.review}/*_both_phones.mp4 and *_motion.jpg; labels {a.labels}', flush=True)


def dense(path, lo, hi, crop, video):
    """The 93-frame original clip: uniform times over the window, the model crop at 478 px, H.264."""
    cap = cv2.VideoCapture(str(path)); frames = []; actual = []
    matrix = np.array(crop['affine_source_to_crop'], np.float32)*SIZE/448
    for t in np.linspace(lo, hi, N):
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000); ok, f = cap.read()
        if not ok:
            raise ValueError(f'Cannot read {path} at {t}')
        actual.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        frames.append(cv2.warpAffine(f, matrix, (SIZE, SIZE), flags=cv2.INTER_AREA,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)))
    cap.release(); encode(video, frames)
    return actual


def sample(path, times):
    """RGB frames of a video at the given times (s)."""
    cap = cv2.VideoCapture(str(path)); rgb = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode {path} at {t}')
        rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(rgb)


def model_frames(path):
    """The 8 model frames (of 93) of a clip, at 448 px; and their indices."""
    v = iio.imread(path)
    idx = np.linspace(0, len(v) - 1, 8).round().astype(int)
    if idx[-1] != N-1:
        raise ValueError(f'Need an exact {N}-frame clip: {path}')
    return np.stack([cv2.resize(f, (448, 448), interpolation=cv2.INTER_AREA) for f in v[idx]]), idx


def endpoint_envelope(endpoint_masks, n):
    """Loose 2D search region from the block at the start and end; never a task/3D label."""
    bounds = []
    for mask in endpoint_masks:
        yy, xx = np.nonzero(mask)
        if not len(xx):
            return None
        bounds.append((xx.mean(), yy.mean(), xx.max()-xx.min()+1, yy.max()-yy.min()+1))
    b = np.array(bounds); lo = b[:, :2].min(0)-2*b[:, 2:].max(0); hi = b[:, :2].max(0)+2*b[:, 2:].max(0)
    return np.broadcast_to(np.r_[lo, hi], (n, 4)).copy()


def tile(rgb, text, width=180):
    canvas = np.zeros((width+28, width, 3), np.uint8)
    canvas[28:] = cv2.resize(rgb, (width, width), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, text, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    return canvas


def prepared_view(ep, view, camera, rgb, model, proc, baseline, review):
    """Agent mask + block track for one view; writes v{view}_rgb/mask.mp4 and a review sheet + sampled MP4."""
    import imageio
    from handsoff.real.sam3 import FILL, box, sam3_masks, top_masks, write_view
    actor = sam3_masks(rgb, SETTINGS['prompts'], model, proc, .4, batch=8, dilate=7)
    mask = actor['person'] | actor['hand'] | actor['arm']
    envelope = endpoint_envelope(baseline[0], len(rgb))
    block, score = top_masks(rgb, 'wooden block', model, proc, .4, mask, .5, max_fraction=.15, allowed_centres=envelope,
                             reject_boundary=True)
    track = np.stack([features(m) for m in block])
    write_view(ep, view, rgb, mask, MODEL_FPS)
    camera.update(block_centroid_envelope_px=envelope[0].tolist() if envelope is not None else None,
                  agent_fraction=mask.mean((1, 2)).tolist(), block_detected=(score > 0).tolist(), block_score=score.tolist(),
                  block_agent_overlap=[float((b & m).sum()/max(b.sum(), 1)) for b, m in zip(block, mask)])
    shown = rgb.copy(); shown[mask] = FILL
    for f, t in zip(shown, track):
        box(f, t, (255, 0, 255))
    name = f"{ep.name}_{camera['kind']}_{camera['look_name']}"
    iio.imwrite(review/f'{name}.jpg', np.concatenate([np.concatenate([tile(f, name) for f in rgb], 1),
                                                       np.concatenate([tile(f, 'masked + SAM3 box') for f in shown], 1)], 0))
    imageio.mimwrite(review/f'{name}.mp4', np.concatenate([rgb, shown], 2), fps=MODEL_FPS, quality=8, macro_block_size=16,
                     ffmpeg_params=['-threads', '1'])
    return track, score


def inputs(a):
    from handsoff.real.cosmos import preservation
    from handsoff.real.review_gate import approved, read_reviews, require_dataset_approval
    from handsoff.real.sam3 import load, top_masks
    pairing = json.loads((a.review/'pairing.json').read_text())
    labels = {r['pair']: r for r in json.loads(a.labels.read_text())['annotations']}
    rows = [r for r in pairing['takes'] if not a.takes or r['number'] in a.takes]
    for r in rows:
        if not labels.get(r['take'], {}).get('user_confirmed'):
            raise ValueError(f"{take_name(r['take'])}: label not confirmed by the user (R09)")
        for kind in ('head', 'second'):
            if digest(ROOT/r[kind]['source']) != r[kind]['sha256']:
                raise ValueError(f"{r[kind]['source']} changed since sync; rerun sync")
    reviews = read_reviews() if a.cosmos else None
    a.out.mkdir(parents=True, exist_ok=True); (a.review/'inputs').mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(dict(settings=SETTINGS, pairing=pairing, cosmos=str(a.cosmos)), sort_keys=True).encode()).hexdigest()
    if (a.out/'config.json').exists() and json.loads((a.out/'config.json').read_text())['sha256'] != signature:
        raise ValueError('Inputs/settings changed; use a fresh output directory')
    atomic_json(a.out/'config.json', dict(sha256=signature, settings=SETTINGS, pairing=str(a.review/'pairing.json'),
                                          cosmos=str(a.cosmos) if a.cosmos else None))
    model, proc = load()
    start = time.time(); rec = dict(ep=[], view=[], track=[], score=[]); quality = []
    for r in rows:
        take = r['take']; name = take_name(take); label = labels[take]; ep = a.out/f'ep{take:05d}'; ep.mkdir(exist_ok=True)
        if (ep/'meta.json').exists():
            meta = json.loads((ep/'meta.json').read_text())
            if meta['config_sha256'] != signature or meta['label'] != label:
                raise ValueError(f'{ep}: prepared with other settings or label; use a fresh output directory')
            cached = np.load(ep/'track.npz'); tracks, scores = list(cached['track']), list(cached['score'])
            if any(iio.imread(ep/f'v{v}_{k}.mp4').shape[0] != 8 for v in range(len(meta['cameras'])) for k in ('rgb', 'mask')):
                raise ValueError(f'Corrupt prepared clip {ep}')
        else:
            lo, hi = r['window_head_s']; cameras, tracks, scores, baselines = [], [], [], {}
            for kind, delta in (('head', 0.), ('second', r['envelope_offset_s'])):
                raw = ROOT/r[kind]['source']
                full = sample(raw, np.linspace(lo, hi, 8)+delta)
                _, crop = crop_frames(full, top_masks(full, 'wooden block', model, proc, .4, max_fraction=.15)[0])
                video = a.review/'inputs'/f'{name}_{kind}.mp4'
                actual = dense(raw, lo+delta, hi+delta, crop, video)
                rgb, idx = model_frames(video)
                camera = dict(kind=kind, look_name='original', generated=False, media_role='original_rgb',
                              source=str(video.relative_to(ROOT)), source_sha256=digest(video),
                              raw_source=r[kind]['source'], raw_source_sha256=r[kind]['sha256'],
                              window_source_s=[lo+delta, hi+delta], actual_sample_times_s=actual, crop=crop, frame_idx=idx.tolist())
                baselines[kind] = top_masks(rgb[[0, -1]], 'wooden block', model, proc, .3, max_fraction=.15)
                tr, sc = prepared_view(ep, len(cameras), camera, rgb, model, proc, baselines[kind], a.review)
                cameras.append(camera); tracks.append(tr); scores.append(sc)
            for kind in ('head', 'second') if a.cosmos else ():
                for look, _, _ in LOOKS:
                    pair = f'{a.tag}:{name}:{kind}'
                    if not approved(reviews, pair, look):
                        continue
                    record = reviews['variants'][pair][look]
                    path = generated_path(a.cosmos, name, kind, look)
                    if (ROOT/record['source']).resolve() != path.resolve():
                        raise ValueError(f"Approved video {record['source']} is not {path}")
                    rgb, idx = model_frames(path)
                    metrics, ok = preservation(rgb, baselines[kind], model, proc)
                    quality.append(dict(source=record['source'], source_sha256=record['source_sha256'], review_id=record['review_id'],
                                        human_approved=True, metrics=metrics, task_preservation_pass=ok))
                    if not ok:
                        continue
                    camera = dict(source=record['source'], source_sha256=record['source_sha256'], kind=kind, look_name=look,
                                  review_pair=pair, review_id=record['review_id'], generated=True, media_role='generated_rgb',
                                  frame_idx=idx.tolist(), task_preservation_pass=True,
                                  task_preservation_source_sha256=record['source_sha256'], endpoint_metrics=metrics)
                    tr, sc = prepared_view(ep, len(cameras), camera, rgb, model, proc, baselines[kind], a.review)
                    cameras.append(camera); tracks.append(tr); scores.append(sc)
            np.savez(ep/'track.npz', track=np.array(tracks), score=np.array(scores))
            atomic_json(ep/'meta.json', dict(episode=take, parent_episode=take, take_name=name, pair=f'{a.tag}:{name}',
                                             label=label, roll_frame='block', frames=8, fps=MODEL_FPS, cameras=cameras,
                                             config_sha256=signature, preparation=SETTINGS,
                                             source=dict(sync_offset_second_s=r['envelope_offset_s'], window_head_s=r['window_head_s'])))
        for v, (tr, sc) in enumerate(zip(tracks, scores)):
            rec['ep'].append(take); rec['view'].append(v); rec['track'].append(tr); rec['score'].append(sc)
        tmp = a.out/'block_track_sam3.tmp.npz'; np.savez(tmp, **{k: np.array(v) for k, v in rec.items()})
        tmp.replace(a.out/'block_track_sam3.npz')
        print(f"{name} {label['label']}: {len(tracks)} views, block found in {[int((s > 0).sum()) for s in scores]}/8 frames; "
              f'{time.time()-start:.0f}s', flush=True)
    require_dataset_approval(a.out)
    atomic_json(a.out/'preparation.json', dict(complete=True, config_sha256=signature, takes=[r['take'] for r in rows],
                                               views=len(rec['ep']), cosmos_checked=quality, seconds=time.time()-start))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['sync', 'inputs'])
    ap.add_argument('--raw', type=Path, required=True); ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--out', type=Path); ap.add_argument('--labels', type=Path, help='default RAW/labels.json')
    ap.add_argument('--id-offset', type=int, default=100)
    ap.add_argument('--tag', default='test', help='identity prefix of the takes (demos3 for the training takes)')
    ap.add_argument('--cosmos', type=Path, help='add the approved Cosmos variants from this directory')
    ap.add_argument('--takes', type=int, nargs='+', help='file numbers N (smoke test into a separate --out)')
    a = ap.parse_args()
    here = lambda p: Path(os.path.abspath(ROOT/p)) if p else None   # noqa: E731  (keeps symlinked data/ inside the repo)
    a.raw, a.review, a.out, a.cosmos = here(a.raw), here(a.review), here(a.out), here(a.cosmos)
    a.labels = here(a.labels) or a.raw/'labels.json'
    if a.stage == 'inputs' and not a.out:
        ap.error('inputs needs --out')
    {'sync': sync, 'inputs': inputs}[a.stage](a)
