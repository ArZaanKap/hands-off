"""SAM3 text-prompted segmentation of phone frames (.venv-mask, GPU): the agent mask and the block mask/track.

Agent = person + hand + arm (each prompt's instances united, dilated 7 px), because the sim masks its whole person
proxy too (R15). The block is the single best "wooden block" instance, after rejecting candidates that are too large
(the table), mostly inside the agent mask, touching the image border, or far from where the block starts and ends.
"""
import cv2
import imageio
import numpy as np
import torch

RES = 448                                   # model input size
FILL = np.array([128, 128, 128], np.uint8)  # the grey the model sees in place of the agent


def load():
    from transformers import Sam3Model, Sam3Processor
    torch.set_num_threads(8)
    proc = Sam3Processor.from_pretrained('facebook/sam3', local_files_only=True)
    model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
    return model, proc


@torch.no_grad()
def sam3_masks(frames, prompts, model, proc, thr, batch=8, dilate=7):
    """{prompt: (N, RES, RES) bool}, each the union of that prompt's instances, dilated."""
    out = {p: np.zeros(frames.shape[:3], bool) for p in prompts}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
    for i in range(0, len(frames), batch):
        imgs = list(frames[i:i + batch])
        for p in prompts:
            inp = proc(images=imgs, text=[p] * len(imgs), return_tensors='pt').to(model.device)
            inp = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
            res = proc.post_process_instance_segmentation(model(**inp), threshold=thr, mask_threshold=0.5,
                                                          target_sizes=[(RES, RES)] * len(imgs))
            for j, r in enumerate(res):
                if len(r['masks']):
                    out[p][i + j] |= r['masks'].bool().any(0).cpu().numpy()
    return {p: np.stack([cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in o]) for p, o in out.items()}


@torch.no_grad()
def top_masks(frames, prompt, model, proc, thr, agent=None, max_overlap=0.5, max_fraction=1.0, allowed_centres=None,
              reject_boundary=False):
    """(T, H, W, 3) uint8 -> (T, H, W) bool mask of the best instance per frame, (T,) score (0 if none).
    Candidates are dropped if larger than max_fraction of the image, touching the border (reject_boundary), more than
    max_overlap inside `agent` (T, H, W), or centred outside allowed_centres[t] = (x0, y0, x1, y1)."""
    inp = proc(images=list(frames), text=[prompt] * len(frames), return_tensors='pt').to(model.device)
    inp = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
    H, W = frames.shape[1:3]
    res = proc.post_process_instance_segmentation(model(**inp), threshold=thr, mask_threshold=0.5,
                                                  target_sizes=[(H, W)] * len(frames))
    masks, scores = np.zeros(frames.shape[:3], bool), np.zeros(len(frames), np.float32)
    for j, r in enumerate(res):
        ms, sc = r['masks'].bool().cpu().numpy(), r['scores'].float().cpu().numpy()
        if len(ms):
            keep = ms.mean((1, 2)) <= max_fraction
            ms, sc = ms[keep], sc[keep]
        if reject_boundary and len(ms):
            keep = ~(ms[:, 0].any(1) | ms[:, -1].any(1) | ms[:, :, 0].any(1) | ms[:, :, -1].any(1))
            ms, sc = ms[keep], sc[keep]
        if agent is not None and len(ms):
            keep = (ms & agent[j]).sum((1, 2)) <= max_overlap * np.maximum(ms.sum((1, 2)), 1)
            ms, sc = ms[keep], sc[keep]
        if allowed_centres is not None and len(ms):
            x0, y0, x1, y1 = allowed_centres[j]
            centres = []
            for candidate in ms:
                yy, xx = np.nonzero(candidate)
                centres.append((xx.mean(), yy.mean()) if len(xx) else (np.inf, np.inf))
            centres = np.asarray(centres)
            keep = (centres[:, 0] >= x0) & (centres[:, 0] <= x1) & (centres[:, 1] >= y0) & (centres[:, 1] <= y1)
            ms, sc = ms[keep], sc[keep]
        if len(ms):
            k = int(sc.argmax())
            masks[j], scores[j] = ms[k], sc[k]
    return masks, scores


def box(im, f, colour):
    """Draw a block track box [u, v, w, h, vis] onto a square image in place."""
    u, v, w, h, vis = f
    if not vis:
        return
    H = im.shape[0]
    x0, x1 = np.clip([int((u - w + 1) / 2 * H), int((u + w + 1) / 2 * H)], 0, H - 1)
    y0, y1 = np.clip([int((v - h + 1) / 2 * H), int((v + h + 1) / 2 * H)], 0, H - 1)
    im[[y0, y1], x0:x1] = colour
    im[y0:y1, [x0, x1]] = colour


def write_view(d, v, rgb, mask, fps):
    imageio.mimwrite(d / f'v{v}_rgb.mp4', rgb, fps=fps, quality=8, macro_block_size=16)
    imageio.mimwrite(d / f'v{v}_mask.mp4', mask.astype(np.uint8) * 255, fps=fps, quality=8, macro_block_size=16)
