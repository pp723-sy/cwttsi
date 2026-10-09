import unittest, importlib.util, tempfile
from pathlib import Path
import torch
from torch import nn
from utils.multi_epoch_dataloader import MultiEpochsDataLoader
from torch.utils.data import TensorDataset

class CheckpointTests(unittest.TestCase):
    def test_resume_reproduces_next_epoch_with_repeat_sampler(self):
        spec=importlib.util.find_spec('training_checkpoint')
        self.assertIsNotNone(spec,'Missing durable training state support')
        from training_checkpoint import save_training_state,restore_training_state
        torch.set_num_threads(2)
        def setup():
            loader=MultiEpochsDataLoader(TensorDataset(torch.arange(24.).reshape(8,3)/24),batch_size=2,shuffle=True,num_workers=0)
            model=nn.Sequential(nn.Linear(3,4),nn.Dropout(.2),nn.Linear(4,1))
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            return loader,model,opt,sched,scaler
        def epoch(loader,model,opt,sched):
            batches=[]
            for (x,) in loader:
                batches.append(x.clone());opt.zero_grad();model(x).square().mean().backward();opt.step()
            sched.step();return batches
        torch.manual_seed(1)
        a=setup();epoch(*a[:4])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
            p=Path(td)/'state.pt'
            save_training_state(p,*a[1:],metadata={'epoch':1,'best_epoch':1})
            order=epoch(*a[:4]);expected={k:v.clone() for k,v in a[1].state_dict().items()}
            b=setup();meta=restore_training_state(p,*b[1:])
            self.assertEqual(meta['epoch'],1)
            actual=epoch(*b[:4])
            for x,y in zip(order,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)
            for key,v in expected.items():torch.testing.assert_close(v,b[1].state_dict()[key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
