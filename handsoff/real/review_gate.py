"""Every generated Cosmos video needs the user's explicit full-video approval before any model use (R01).

config/cosmos_review.json holds the decisions: variants[pair][look] = {status: approved/rejected/pending, source,
source_sha256}. An approval only counts while the file still has that hash; Cosmos control maps never count.
"""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW_PATH = ROOT / "config/cosmos_review.json"


def is_control_video(source):
    """Cosmos saves conditioning maps separately from the generated RGB output."""
    return bool(source) and '_control_' in Path(str(source)).name.lower()


def require_rgb_source(source):
    """Control maps are forbidden even if mistakenly marked human-approved."""
    if is_control_video(source):
        raise ValueError(f'Auxiliary Cosmos control video is not an RGB model input: {source}')


def read_reviews(path=REVIEW_PATH):
    path = Path(path)
    review = json.loads(path.read_text()) if path.is_file() else {"variants": {}}
    for looks in review.get("variants", {}).values():
        for record in looks.values():
            if is_control_video(record.get('source')) or record.get('media_role') == 'control':
                record.update(status='excluded', exclusion_reason='Auxiliary control maps are never model inputs')
                continue
            if record.get("status") != "approved" or "source" not in record:
                continue
            source = Path(record["source"])
            source = source if source.is_absolute() else ROOT / source
            digest = record.get("source_sha256")
            if not digest or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                record.update(status="pending", integrity_reason="Approved source missing, changed or lacks a file hash; fresh full-video review required")
    return review


def approved(review, pair, look):
    """Original recordings bypass this generated-video gate; pending/rejected never pass."""
    if look in (None, "original"):
        return True
    record = review.get("variants", {}).get(pair, {}).get(look, {})
    return (record.get("status") == "approved" and not is_control_video(record.get('source'))
            and record.get('media_role') != 'control')


def view_approved(review, meta, view):
    """Check the current registry at every model entry point, including old caches."""
    cameras = meta.get("cameras", [])
    if not 0 <= view < len(cameras):
        return False
    camera = cameras[view]
    if (is_control_video(camera.get('source')) or is_control_video(camera.get('video_path'))
            or camera.get('media_role') == 'control'):
        return False
    look = camera.get("look_name")
    # A prepared take may contain both phone views and their appearance variants.
    pair = camera.get("review_pair", meta.get("pair"))
    if camera.get("generated") and look in (None, "original"):
        return False  # missing generated-video identity must never bypass review
    if look not in (None, "original") and camera.get("source_sha256"):
        record = review.get("variants", {}).get(pair, {}).get(look, {})
        if camera["source_sha256"] != record.get("source_sha256"):
            return False  # old preparation cannot inherit approval of a regenerated source
    return approved(review, pair, look)


def require_dataset_approval(root, review_path=REVIEW_PATH):
    """Refuse bulk preparation/training on a dataset containing an unapproved view.

    Historical caches are preserved for diagnosis. Query loaders may filter rows,
    but bulk writers must stop rather than silently reindexing views or caches.
    """
    root = Path(root)
    review = read_reviews(review_path)
    blocked = []
    controls = []
    for path in sorted(root.glob("ep*/meta.json")):
        meta = json.loads(path.read_text())
        for view in range(len(meta.get("cameras", []))):
            camera = meta['cameras'][view]
            if (is_control_video(camera.get('source')) or is_control_video(camera.get('video_path'))
                    or camera.get('media_role') == 'control'):
                controls.append(f'{path.parent.name}/v{view}: {camera.get("source")}')
                continue
            if not view_approved(review, meta, view):
                blocked.append(f"{path.parent.name}/v{view}: {meta.get('pair')} "
                               f"{meta['cameras'][view].get('look_name')}")
    if controls:
        raise ValueError('Auxiliary Cosmos control videos are forbidden in training/evaluation, including cached features: '
                         + '; '.join(controls))
    if blocked:
        raise ValueError("Human full-video approval required; rejected/pending generated inputs blocked: "
                         + "; ".join(blocked))
