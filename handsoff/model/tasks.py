"""The nine tasks and task composition ("intent arithmetic").

A task is a category (movement, roll), each -1/0/+1: move right / stay / move left, and roll right / none / roll left
(+ = toward the person's left). Roll is about the block's initial forward axis; movement is in the table frame.

Composition queries z(a) - z(b) + z(c), three families (and their left/right mirrors):
  extract_roll         right+roll - right + stay          -> roll in place
  extract_translation  right+roll - roll in place + stay  -> right, upright
  transfer_roll        right+roll - right + left          -> left with the same roll
Translation and roll are scored separately; exact distances are never scored.
"""
import numpy as np

FAMILIES = ('extract_roll', 'extract_translation', 'transfer_roll')
CONFIG = dict(deadband_m=0.03, forward_limit_m=0.03, motion_match_m=0.03, start_xy_m=0.02, start_yaw_deg=10,
              block_size_m=0.005)
MOVE = {-1: 'move right', 0: 'stay', 1: 'move left'}
ROLL = {-1: 'roll right', 0: 'no roll', 1: 'roll left'}


def name(c):
    return f'{MOVE[int(c[0])]} + {ROLL[int(c[1])]}'


def category(o, deadband=0.03):
    """Measured outcome (dx, dy, roll) -> (movement, roll) category."""
    o = np.asarray(o)
    return np.stack([np.where(np.abs(o[..., 1]) < deadband, 0, np.sign(o[..., 1])),
                     np.rint(o[..., 2] / (np.pi / 2))], -1).astype(int)


def eligible(o):
    """A clean example of one task: little forward drift, a whole quarter turn at most, and a 'stay' that stays."""
    o = np.asarray(o)
    c = category(o)
    return ((np.abs(o[..., 0]) <= CONFIG['forward_limit_m'])
            & (np.abs(c[..., 1]) <= 1)
            & ((c[..., 0] != 0) | (np.linalg.norm(o[..., :2], axis=-1) <= CONFIG['deadband_m'])))


def matched_start(a, b):
    delta = a['start']['yaw'] - b['start']['yaw']
    yaw = abs(np.arctan2(np.sin(delta), np.cos(delta)))
    return (np.linalg.norm(np.asarray(a['start']['xy']) - b['start']['xy']) <= CONFIG['start_xy_m']
            and yaw <= np.radians(CONFIG['start_yaw_deg'])
            and np.max(np.abs(np.asarray(a['size']) - b['size'])) <= CONFIG['block_size_m'])


def make_queries(rows, ids):
    """All composition queries among clips `ids` (metadata only, independent of any model)."""
    out = []
    for ai in ids:
        a = rows[ai]
        d, r = a['cat']
        if d == 0 or r == 0:
            continue
        for family in FAMILIES:
            bc = (d, 0) if family != 'extract_translation' else (0, r)
            cc = (0, 0) if family != 'transfer_roll' else (-d, 0)
            target = (0, r) if family == 'extract_roll' else (d, 0) if family == 'extract_translation' else (-d, r)
            for bi in ids:
                b = rows[bi]
                if tuple(b['cat']) != bc or a['clip_dir'] == b['clip_dir']:
                    continue
                if family != 'extract_translation' and \
                        np.linalg.norm(np.asarray(a['outcome'][:2]) - b['outcome'][:2]) > CONFIG['motion_match_m']:
                    continue
                for ci in ids:
                    c = rows[ci]
                    if tuple(c['cat']) != cc or c['clip_dir'] in (a['clip_dir'], b['clip_dir']):
                        continue
                    if family == 'transfer_roll' and \
                            abs(abs(c['outcome'][1]) - abs(a['outcome'][1])) > CONFIG['motion_match_m']:
                        continue
                    out.append(dict(a=ai, b=bi, c=ci, family=family, target=list(target),
                                    start_matched=matched_start(a, b) and matched_start(a, c)))
    return out


def composition_queries(rows, indices):
    """The 12 queries of each start group (all nine tasks from one start, so operands share start and block)."""
    queries = []
    for group in sorted({rows[i]['start_group'] for i in indices}):
        ids = [int(i) for i in indices if rows[i]['start_group'] == group]
        cats = [tuple(rows[i]['cat']) for i in ids]
        if len(ids) != 9 or set(cats) != {(d, r) for d in (-1, 0, 1) for r in (-1, 0, 1)}:
            raise ValueError(f'Start group {group} needs exactly nine valid task categories')
        first = rows[ids[0]]
        if any(rows[i]['start'] != first['start'] or rows[i]['size'] != first['size'] for i in ids):
            raise ValueError('Composition operands must share exact start and block dimensions')
        actual = make_queries(rows, ids)
        if len(actual) != 12 or not all(q['start_matched'] for q in actual):
            raise ValueError('Need all 12 motion-matched, start-matched mirrored compositions per group')
        queries += actual
    return queries


def composition_accuracy(queries, rows, z, gallery, gallery_ids):
    """Fraction of queries whose nearest gallery clip (excluding the operand clips) has the target task."""
    cats = np.array([rows[i]['cat'] for i in gallery_ids])
    clips = [rows[i]['clip_dir'] for i in gallery_ids]
    hits = []
    for q in queries:
        operands = {rows[q[k]]['clip_dir'] for k in ('a', 'b', 'c')}
        allowed = np.array([c not in operands for c in clips])
        good = (cats == q['target']).all(1) & allowed
        if not good.any():
            continue
        v = z[q['a']] - z[q['b']] + z[q['c']]
        if np.linalg.norm(v) < 1e-8:
            hits.append(0.); continue
        s = gallery @ v / np.linalg.norm(v)
        s[~allowed] = -np.inf
        hits.append(float(good[int(s.argmax())]))
    return dict(n=len(hits), joint=float(np.mean(hits)) if hits else float('nan'))
