"""Atomic epoch-boundary checkpoints for trusted local experiment files."""
from pathlib import Path
import os,random
import numpy as np
import torch

def save_training_state(path,model,optimizer,scheduler,scaler,metadata):
    path=Path(path);temp=path.with_suffix('.tmp')
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),metadata=metadata,
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(state,temp);os.replace(temp,path)

def restore_training_state(path,model,optimizer,scheduler,scaler):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['metadata']
