"""A high-confidence table corner must not beat a block inside its observed region."""
import unittest

import numpy as np
import torch

from handsoff.real.sam3 import top_masks


class Inputs(dict):
    def to(self,device):return self


class Processor:
    def __init__(self,result):self.result=result
    def __call__(self,**kwargs):return Inputs()
    def post_process_instance_segmentation(self,*args,**kwargs):return self.result


class Model:
    device='cpu'
    dtype=torch.float32
    def __call__(self,**kwargs):return None


class BlockCandidateTests(unittest.TestCase):
    def test_boundary_table_is_rejected_even_inside_centroid_region(self):
        candidates=torch.zeros((2,32,32),dtype=torch.bool)
        candidates[0,15:22,28:32]=True
        candidates[1,14:20,15:19]=True
        result=[dict(masks=candidates,scores=torch.tensor([.98,.90]))]
        masks,_=top_masks(np.zeros((1,32,32,3),np.uint8),'block',Model(),Processor(result),.4,
                          allowed_centres=np.array([[0,0,32,32]]),reject_boundary=True)
        self.assertTrue(np.array_equal(masks[0],candidates[1].numpy()))

    def test_filter_before_ranking_and_missing_when_only_table_remains(self):
        candidates=torch.zeros((2,32,32),dtype=torch.bool)
        candidates[0,1:5,1:5]=True
        candidates[1,14:20,15:19]=True
        result=[dict(masks=candidates,scores=torch.tensor([.98,.90])),
                dict(masks=candidates[:1],scores=torch.tensor([.98]))]
        masks,scores=top_masks(np.zeros((2,32,32,3),np.uint8),'block',Model(),Processor(result),.4,
                               allowed_centres=np.array([[10,10,25,25]]*2))
        self.assertTrue(np.array_equal(masks[0],candidates[1].numpy()))
        self.assertAlmostEqual(float(scores[0]),.90,places=6)
        self.assertFalse(masks[1].any())
        self.assertEqual(scores[1],0)
        historical,_=top_masks(np.zeros((2,32,32,3),np.uint8),'block',Model(),Processor(result),.4)
        self.assertTrue(np.array_equal(historical[0],candidates[0].numpy()))


if __name__=='__main__':unittest.main()
