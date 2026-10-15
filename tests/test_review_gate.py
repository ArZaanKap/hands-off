"""Cosmos videos reach a model only with the user's approval of exactly those bytes (R01); control maps never do."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handsoff.model.data import load_queries
from handsoff.real.review_gate import approved, read_reviews, require_dataset_approval, require_rgb_source, view_approved


class ReviewGate(unittest.TestCase):
    def test_pending_and_rejected_never_pass_originals_always_do(self):
        r = {'variants': {'take': {'light_blue': {'status': 'rejected'}, 'marble': {'status': 'pending'}}}}
        self.assertFalse(approved(r, 'take', 'light_blue'))
        self.assertFalse(approved(r, 'take', 'marble'))
        self.assertFalse(approved(r, 'other', 'light_blue'))
        self.assertTrue(approved(r, 'take', 'original'))

    def test_each_phone_has_its_own_review_and_stale_bytes_fail(self):
        review = {'variants': {'demos3:D01:head': {'blue': dict(status='approved', source_sha256='h')},
                               'demos3:D01:second': {'blue': dict(status='rejected', source_sha256='s')}}}
        meta = dict(pair='demos3:D01', cameras=[
            dict(generated=True, look_name='blue', source_sha256='h', review_pair='demos3:D01:head'),
            dict(generated=True, look_name='blue', source_sha256='s', review_pair='demos3:D01:second')])
        self.assertTrue(view_approved(review, meta, 0))
        self.assertFalse(view_approved(review, meta, 1))
        meta['cameras'][0]['source_sha256'] = 'stale'
        self.assertFalse(view_approved(review, meta, 0))

    def test_overwritten_approved_video_needs_fresh_review(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); source = d/'generated.mp4'; source.write_bytes(b'reviewed video'); path = d/'reviews.json'
            path.write_text(json.dumps({'variants': {'p': {'look': {'status': 'approved', 'source': str(source),
                                                                     'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}}}))
            self.assertTrue(approved(read_reviews(path), 'p', 'look'))
            source.write_bytes(b'regenerated video')
            self.assertFalse(approved(read_reviews(path), 'p', 'look'))
            source.unlink()
            self.assertFalse(approved(read_reviews(path), 'p', 'look'))

    def test_control_maps_are_never_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); source = d/'generated_control_edge.mp4'; source.write_bytes(b'edge control')
            record = dict(status='approved', source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            review = {'variants': {'p': {'look': record}}}; path = d/'review.json'; path.write_text(json.dumps(review))
            self.assertFalse(approved(read_reviews(path), 'p', 'look'))
            with self.assertRaisesRegex(ValueError, 'not an RGB model input'):
                require_rgb_source(source)
            ep = d/'ep00000'; ep.mkdir()
            (ep/'meta.json').write_text(json.dumps({'pair': 'p', 'cameras': [dict(source=str(source), look_name='look', generated=True)]}))
            with self.assertRaisesRegex(ValueError, 'control videos are forbidden'):
                require_dataset_approval(d, path)

    def test_bulk_writers_block_unapproved_and_unidentified_generated_views(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); ep = d/'ep00000'; ep.mkdir(); reviews = d/'review.json'
            reviews.write_text(json.dumps({'variants': {'take': {'marble': {'status': 'approved'}}}}))
            for camera in ({'look_name': 'light_blue'}, {'generated': True}, {'generated': True, 'look_name': 'original'}):
                (ep/'meta.json').write_text(json.dumps({'pair': 'take', 'cameras': [camera]}))
                with self.assertRaisesRegex(ValueError, 'approval required'):
                    require_dataset_approval(d, reviews)
            (ep/'meta.json').write_text(json.dumps({'pair': 'take', 'cameras': [{'look_name': 'marble'}]}))
            require_dataset_approval(d, reviews)

    def test_cached_features_of_unapproved_views_are_dropped_and_tracks_stay_aligned(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); ep = d/'ep00000'; ep.mkdir()
            (ep/'meta.json').write_text(json.dumps({'pair': 'take', 'cameras': [{'look_name': k} for k in ['original', 'light_blue', 'marble']]}))
            reviews = d/'review.json'
            reviews.write_text(json.dumps({'variants': {'take': {'light_blue': {'status': 'rejected'}, 'marble': {'status': 'approved'}}}}))
            np.savez(d/'cosmos_agent.npz', ep=[0, 0, 0], view=[0, 1, 2], frame_cls=np.array([np.full((8, 4), n) for n in [1, 2, 3]]),
                     proj=np.array([np.full(4, n) for n in [1, 2, 3]]))
            np.savez(d/'block_track_sam3.npz', ep=[0, 0, 0], view=[0, 1, 2], track=np.array([np.full((8, 5), n) for n in [1, 2, 3]]))
            keys, x, _ = load_queries(d, review_path=reviews)
            self.assertEqual(keys, [(0, 0), (0, 2)])
            self.assertEqual(x['frames'][:, 0, 0].tolist(), [1, 3])
            self.assertEqual(x['track'][:, 0, 0].tolist(), [1, 3])


if __name__ == '__main__':
    unittest.main()
