"""Load the separately obtained, pinned SSTBAN dependency without bundling it.

Only bottleneck building blocks are loaded; the author's self-supervised model
and training program are not imported. Compatibility adaptations reproduce the
archived supervised branch's default mask and device-aware embedding behavior.
"""
from pathlib import Path
import ast
import hashlib
import os

COMMIT = '30bafd9d27c99cf7efe311c76ac145b1b9f7fc0c'
SHA256 = 'cc81572f3f1dcb01f1c59fcfd45c9f1b4df653a3d2f27b1c7b4cc5599e76424a'
NAMES = {'conv2d_', 'FC', 'STEmbedding', 'MAB', 'spatialAttention',
         'temporalAttention', 'STAttBlock', 'transformAttention'}
_components = None


def load_components():
    global _components
    if _components is not None:
        return _components
    path = Path(os.environ.get('CWT_TSI_SSTBAN_SOURCE', 'external/SSTBAN/model/sstban_model.py'))
    if not path.is_file():
        raise FileNotFoundError('Obtain SSTBAN separately at commit '+COMMIT+
            '; see EXTERNAL_DEPENDENCIES.md. Missing: '+str(path))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('SSTBAN source hash does not match the audited dependency')
    tree = ast.parse(data.decode('utf-8'), filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in NAMES]
    if {n.name for n in classes} != NAMES:
        raise ValueError('Missing audited SSTBAN building blocks')
    for cls in classes:
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == 'forward':
                if fn.args.args[-1].arg == 'mask' and not fn.args.defaults:
                    fn.args.defaults = [ast.Constant(None)]
    import torch
    import math
    from torch import nn
    import torch.nn.functional as F
    scope = dict(__name__=__name__, torch=torch, nn=nn, F=F, math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(path),'exec'),scope)
    # Same one-hot representation as the archive, on the spatial embedding device.
    def embedding_forward(self, spatial, calendar, daily_slots):
        calendar=calendar.to(device=spatial.device,dtype=torch.long)
        week=F.one_hot(calendar[...,0].remainder(7),7)
        slot=F.one_hot(calendar[...,1].remainder(daily_slots),daily_slots)
        features=torch.cat((week,slot),dim=-1).float().unsqueeze(2)
        return spatial[None,None]+self.FC_te(features)
    scope['STEmbedding'].forward = embedding_forward
    _components = {name:scope[name] for name in NAMES}
    globals().update(_components)
    return _components


def __getattr__(name):
    if name in NAMES:
        return load_components()[name]
    raise AttributeError(name)
