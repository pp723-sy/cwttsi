import os
from pathlib import Path
import unittest
import torch
from model_convnext_stban import ConvNeXtSTBAN


@unittest.skipUnless(Path(os.environ.get('CWT_TSI_SSTBAN_SOURCE','external/SSTBAN/model/sstban_model.py')).is_file(), 'Acquire the separately supplied SSTBAN dependency first')
class ExternalBranchTests(unittest.TestCase):
    def test_temporal_branch_forecasts_and_backpropagates(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        model=ConvNeXtSTBAN(num_nodes=3,predict_length=12,time_slots_per_day=108,ablation_mode='no_img')
        calendar=torch.zeros(2,24,2,dtype=torch.long)
        calendar[...,0]=8
        calendar[...,1]=109
        pred=model(torch.zeros(2,1,128,128),torch.randn(2,3,12),calendar,torch.tensor(0.),torch.tensor(1.))
        self.assertEqual(tuple(pred.shape),(2,3,12))
        self.assertTrue(torch.isfinite(pred).all())
        pred.square().mean().backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.sta.parameters()))


if __name__=='__main__':unittest.main()
