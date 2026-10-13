"""Framing steps shared by real phone clips and simulated phone cameras.

crop_frames: the square model crop (moved unchanged from prepare_demos3.py).
features: block centroid + bbox from a mask (moved unchanged from sam3_block_track.py).
Real clips get block masks from SAM3; the simulator gets them from segmentation rendering.
"""
import cv2
import numpy as np

RES = 448   # model input size; same as real_clips.RES
CROP_RULE = "centre initial block; include detected path with 32px margin; black padding outside source"


def crop_frames(rgb, block_masks):
    ys, xs = np.nonzero(block_masks[0])
    if not len(xs):
        raise RuntimeError("Initial block not detected; inspect before proceeding")
    cx, cy = xs.mean(), ys.mean()
    extents = []
    for mask in block_masks:
        yy, xx = np.nonzero(mask)
        if len(xx):
            extents.extend([abs(xx.min()-cx), abs(xx.max()-cx), abs(yy.min()-cy), abs(yy.max()-cy)])
    size = int(np.ceil(max(rgb.shape[1], 2 * (max(extents) + 32)) / 2) * 2)
    scale = RES / size
    matrix = np.float32([[scale, 0, RES/2 - (cx+.5)*scale],
                         [0, scale, RES/2 - (cy+.5)*scale]])
    frames = np.stack([cv2.warpAffine(f, matrix, (RES, RES), flags=cv2.INTER_AREA,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)) for f in rgb])
    valid = cv2.warpAffine(np.ones(rgb.shape[1:3], np.uint8), matrix, (RES, RES), flags=cv2.INTER_NEAREST)
    return frames, dict(centre_source_px=[float(cx), float(cy)], size_source_px=size,
                        affine_source_to_crop=matrix.tolist(), padding_fraction=float(1-valid.mean()),
                        rule=CROP_RULE)


def features(mask):
    """(H, W) bool -> [u, v, w, h, vis] in block_track.py's convention."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(5, np.float32)
    H, W = mask.shape
    return np.array([2 * (xs.mean() + 0.5) / W - 1, 2 * (ys.mean() + 0.5) / H - 1,
                     (xs.max() - xs.min() + 1) / W, (ys.max() - ys.min() + 1) / H, 1], np.float32)
