"""Embed every view of a dataset (sim or real) with frozen Cosmos-Embed1-448p, agent masked out (GPU, bf16).

Reads ep*/v{i}_rgb.mp4 and v{i}_mask.mp4, greys out the agent (person / robot hand), samples 8 frames uniformly and
writes <dataset>/cosmos_agent.npz: ep (N,), view (N,), proj (N, 768), cls (N, 768), frame_cls (N, 8, 1408).
Reruns only embed episodes missing from an existing file; saves every --save-every episodes (atomic rename).
Needs transformers==4.51.3 (the model's remote code breaks on 5.x).
Usage: python -m handsoff.model.embed DATA [--batch 4] [--save-every 100] [--limit N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_MODULES_CACHE', os.path.expanduser('~/.cache/hf_modules'))
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from handsoff.real.review_gate import require_dataset_approval  # noqa: E402

MODEL_ID = 'nvidia/Cosmos-Embed1-448p'
REVISION = 'f60ec73636eb7c9cc25267367713b7b1b0cffaf3'
N_FRAMES = 8
FILL = np.array([128, 128, 128], np.uint8)


def _sdpa_forward(self, x, rel_pos_bias=None):
    """Same maths as the remote ViT Attention.forward, with PyTorch's fused attention."""
    B, N, _ = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    bias = None
    if self.relative_position_bias_table is not None:
        n = self.window_size[0] * self.window_size[1] + 1
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1)
        bias = bias.permute(2, 0, 1).unsqueeze(0).to(q.dtype)
    if rel_pos_bias is not None:
        bias = rel_pos_bias if bias is None else bias + rel_pos_bias
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, -1)))


def load(device='cuda'):
    from transformers import AutoModel, AutoProcessor
    dtype = torch.bfloat16 if device != 'cpu' else torch.float32
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION).to(device, dtype).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, revision=REVISION)
    for mod in model.modules():
        if type(mod).__name__ == 'Attention' and hasattr(mod, 'qkv'):
            mod.forward = _sdpa_forward.__get__(mod)
    return model, proc


@torch.no_grad()
def embed(model, proc, clips, batch=4):
    """clips: list of (8, H, W, 3) uint8 -> dict: proj (N, 768), cls (N, 768), frame_cls (N, 8, 1408)."""
    dev, dt = next(model.parameters()).device, next(model.parameters()).dtype
    out = dict(proj=[], cls=[], frame_cls=[])
    for i in range(0, len(clips), batch):
        x = np.stack(clips[i:i + batch]).transpose(0, 1, 4, 2, 3)  # B T C H W
        o = model.get_video_embeddings(**proc(videos=x).to(dev, dt))
        out['proj'].append(o.visual_proj.float().cpu().numpy())
        out['cls'].append(o.visual_cls_tokens.float().cpu().numpy())
        out['frame_cls'].append(o.frame_cls_tokens[:, :, 0].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def read_view(ep_dir, v):
    """8 uniformly sampled frames of one view, agent greyed out."""
    rgb = iio.imread(ep_dir/f'v{v}_rgb.mp4')
    idx = np.linspace(0, len(rgb) - 1, N_FRAMES).round().astype(int)
    rgb = rgb[idx].copy()
    mask = iio.imread(ep_dir/f'v{v}_mask.mp4')[idx]
    rgb[(mask[..., 0] if mask.ndim == 4 else mask) > 127] = FILL
    return rgb


def episode_ok(p):
    """meta.json parses and every view's rgb and mask video is non-empty."""
    try:
        n = len(json.loads((p/'meta.json').read_text())['cameras'])
    except (ValueError, KeyError):
        return False
    return all((p/f'v{v}_{k}.mp4').exists() and (p/f'v{v}_{k}.mp4').stat().st_size > 0
               for v in range(n) for k in ('rgb', 'mask'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', type=Path)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--save-every', type=int, default=100, help='episodes between checkpoints')
    ap.add_argument('--limit', type=int, default=0, help='only the first N episodes (smoke tests)')
    a = ap.parse_args()
    root = a.dataset
    require_dataset_approval(root)   # also before accepting an existing cache
    out = root/'cosmos_agent.npz'
    have = dict(np.load(out)) if out.exists() else None
    done = set(have['ep'].tolist()) if have else set()
    eps = sorted(p for p in root.glob('ep*') if (p/'meta.json').exists())
    bad = [p.name for p in eps if not episode_ok(p)]
    if bad:
        sys.exit(f"{len(bad)} incomplete episodes (empty/corrupt meta.json or videos): {' '.join(bad)}")
    todo = [p for p in eps if int(p.name[2:]) not in done][:a.limit or None]
    print(f'{len(eps)} episodes, {len(todo)} to embed', flush=True)
    if not todo:
        return
    model, proc = load('cuda')
    t0 = time.time(); feats = ('proj', 'cls', 'frame_cls')
    rec = dict(ep=[], view=[], **{k: [] for k in feats}); clips, keys = [], []

    def flush():
        e = embed(model, proc, clips, batch=a.batch)
        for i, (ep, v) in enumerate(keys):
            rec['ep'].append(ep); rec['view'].append(v)
            for k in feats:
                rec[k].append(e[k][i])
        clips.clear(); keys.clear()

    def save():
        new = {k: np.array(v) for k, v in rec.items()}
        if have:
            new = {k: np.concatenate([have[k], new[k]]) for k in new}
        tmp = out.with_suffix('.tmp.npz'); np.savez(tmp, **new); os.replace(tmp, out)
        return len(new['ep'])

    for n, p in enumerate(todo):
        for v in range(len(json.loads((p/'meta.json').read_text())['cameras'])):
            clips.append(read_view(p, v)); keys.append((int(p.name[2:]), v))
        if len(clips) >= 4 * a.batch:
            flush()
        if (n + 1) % a.save_every == 0:   # rec holds whole episodes only; pending clips are redone after a crash
            if clips:
                flush()
            print(f'{n + 1}/{len(todo)} {time.time() - t0:.0f}s, saved {save()} clips', flush=True)
    if clips:
        flush()
    print(f'wrote {out}: {save()} clips in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
