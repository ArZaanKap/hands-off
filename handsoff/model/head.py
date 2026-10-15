"""The intent head: a small network on frozen per-frame features -> a unit 64-d intent vector z."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Clip features {frames [B,8,1408] (Cosmos-Embed1 frame CLS), track [B,8,5] (2D block box)} -> z [B, 64] (unit
    norm). Each branch maps every frame to d; its first frame, last frame and frame mean go into the mix (0.84 M
    parameters)."""

    def __init__(self, d_frame=1408, d=256, d_out=64, d_track=5):
        super().__init__()
        self.frame = nn.Sequential(nn.LayerNorm(d_frame), nn.Linear(d_frame, d), nn.GELU())
        self.track = nn.Sequential(nn.Linear(d_track, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(6 * d, d), nn.GELU(), nn.Linear(d, d_out))

    def forward(self, x):
        summary = lambda h: [h[:, 0], h[:, -1], h.mean(1)]  # noqa: E731
        parts = summary(self.frame(x['frames'])) + summary(self.track(x['track']))
        return F.normalize(self.mix(torch.cat(parts, -1)), dim=-1)


@torch.no_grad()
def embed_all(x, head, dev, chunk=64):
    """Features {k: [E, V, ...]} -> z [E, V, 64] (numpy)."""
    E, V = x['frames'].shape[:2]
    z = [head({k: x[k][e:e + chunk].flatten(0, 1).to(dev) for k in ('frames', 'track')}).cpu() for e in range(0, E, chunk)]
    return torch.cat(z).reshape(E, V, -1).numpy()


def view_pairs(rng, episodes, look):
    """[len(episodes), 2] view indices: two views of each episode from two different looks."""
    out = []
    for i in episodes:
        l1, l2 = rng.choice(np.unique(look[i]), 2, replace=False)
        out.append([rng.choice(np.flatnonzero(look[i] == l1)), rng.choice(np.flatnonzero(look[i] == l2))])
    return np.array(out)
