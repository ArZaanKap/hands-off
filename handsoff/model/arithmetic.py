"""Task arithmetic on held-out sim starts with a trained head: hit rates per family + one worked example sheet each.

Same protocol as train.validation(): sources (a, b, c) from the first half of the validation starts (head view,
look 0); answers from the other half (head view, the other look), so the answer is never an operand and is always in
a different room/material look. Baseline: c alone. The head is trained with arithmetic on the training starts, so
this tests whether it generalises to new starts and looks, not whether it emerges on its own.
Usage: python -m handsoff.model.arithmetic RUN [--sim DATA] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from handsoff.model.data import FEAT, TrackNoise, load_sim, read_inputs
from handsoff.model.head import Head, embed_all
from handsoff.model.tasks import FAMILIES, composition_queries, name

ROOT = Path(__file__).resolve().parents[2]
TITLE = dict(extract_roll='remove the move, keep the roll', extract_translation='remove the roll, keep the move',
             transfer_roll='swap the move direction, keep the roll')
T = 200


def strip(clip, view, label, colour, font):
    _, masked = read_inputs(Path(clip), view)
    s = Image.new('RGB', (T*5 + 330, T), (255, 255, 255)); d = ImageDraw.Draw(s)
    d.multiline_text((8, 60), label, fill=colour, font=font, spacing=6)
    for j, f in enumerate(masked[[0, 2, 4, 6, 7]]):
        s.paste(Image.fromarray(f).resize((T, T)), (330 + j*T, 0))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path, help='training output with head.pt and results.json')
    ap.add_argument('--sim', type=Path, default=ROOT/'data/sim/intent_tasks_train_v3')
    ap.add_argument('--out', type=Path, default=ROOT/'results/arithmetic')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    D, metas, rows, folds = load_sim(a.sim)
    seed = json.loads((a.run/'results.json').read_text())['provenance']['seed']
    head = Head(FEAT); head.load_state_dict(torch.load(a.run/'head.pt', map_location='cpu')); head.eval()
    val = folds['validation']; x = {k: v[val].clone() for k, v in D['x'].items()}
    x['track'] = TrackNoise()(x['track'], torch.Generator().manual_seed(1000+seed))   # same as training validation
    zv = embed_all(x, head, 'cpu')
    z = np.zeros((len(rows), *zv.shape[1:]), zv.dtype); z[val] = zv
    groups = sorted({rows[i]['start_group'] for i in val})
    source = [int(i) for i in val if rows[i]['start_group'] in groups[:len(groups)//2]]
    gallery = [int(i) for i in val if i not in source]
    gview = [int(np.flatnonzero(D['head'][i] & (D['look'][i] != D['look'][i, 0]))[0]) for i in gallery]
    G = z[gallery, gview]; gcat = np.array([rows[i]['cat'] for i in gallery])
    results = []
    for q in composition_queries(rows, source):
        v = z[q['a'], 0] - z[q['b'], 0] + z[q['c'], 0]; v /= np.linalg.norm(v)
        k = int((G @ v).argmax())
        c_only = bool((gcat[int((G @ z[q['c'], 0]).argmax())] == q['target']).all())
        results.append(dict(q, answer=gallery[k], answer_view=gview[k], hit=bool((gcat[k] == q['target']).all()),
                            c_only_hit=c_only))
    summary = {f: dict(n=len(r), hit=float(np.mean([x['hit'] for x in r])), c_only=float(np.mean([x['c_only_hit'] for x in r])))
               for f in FAMILIES for r in [[x for x in results if x['family'] == f]]}
    print(a.run.name, json.dumps(summary), flush=True)
    font = ImageFont.load_default(size=17)
    for fam in FAMILIES:
        r = [x for x in results if x['family'] == fam]; q = next((x for x in r if x['hit']), r[0])
        clip = lambda i: rows[i]['clip_dir']  # noqa: E731
        parts = [strip(clip(q['a']), 0, f"A  {name(rows[q['a']]['cat'])}\n(start {rows[q['a']]['start_group']})", (0, 0, 0), font),
                 strip(clip(q['b']), 0, f"MINUS  B  {name(rows[q['b']]['cat'])}", (180, 0, 0), font),
                 strip(clip(q['c']), 0, f"PLUS  C  {name(rows[q['c']]['cat'])}", (0, 120, 0), font),
                 strip(clip(q['answer']), q['answer_view'],
                       f"MODEL'S ANSWER:\n{name(rows[q['answer']]['cat'])}\n(start {rows[q['answer']]['start_group']}, other look)\n"
                       f"expected: {name(q['target'])}\n{'CORRECT' if q['hit'] else 'WRONG'}", (0, 0, 200), font)]
        sheet = Image.new('RGB', (parts[0].width, T*4 + 40), (255, 255, 255))
        ImageDraw.Draw(sheet).text((8, 8), f"{TITLE[fam]}  |  {a.run.name}  |  this family: {summary[fam]['hit']:.0%} correct "
                                   f"(C alone: {summary[fam]['c_only']:.0%})", fill=(0, 0, 0), font=font)
        for i, p in enumerate(parts):
            sheet.paste(p, (0, 40 + i*T))
        sheet.save(a.out/f'{fam}_example.jpg', quality=88)
    (a.out/f'{a.run.name}_summary.json').write_text(json.dumps(dict(run=str(a.run), families=summary, queries=len(results)), indent=1))


if __name__ == '__main__':
    main()
