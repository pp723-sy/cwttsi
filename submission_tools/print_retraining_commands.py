"""Print the 12 own-model historical commands, without executing training."""
from pathlib import Path
import json
import shlex

package=Path(__file__).resolve().parents[1]
runs=json.loads((package/'run_index.json').read_text(encoding='utf-8'))
print('# Historical window_ratio protocol; early CLI settings were partly reconstructed.')
print('# Use fresh run names. Authorized data and the external dependency are required.')
for r in runs:
    mode=next((m for m in ['no_img','no_ts','no_fuse','full'] if r['run'].endswith('_'+m)),None)
    if mode is None:continue
    ds=r['dataset']
    cmd=['python','train.py','--dataset_name',ds,'--run_name','replay_20261009_'+r['run'],'--history_step','12','--target_step','12','--batch_size','16','--num_epochs','100','--learning_rate','0.001','--early_stop_patience','20','--loss','Huber','--seed','1','--split_protocol','window_ratio','--ablation_mode',mode,'--time_slots_per_day','108' if ds=='BJMetro' else '216','--x_mode','cwt','--img_size','128','--loss_scale','raw','--amp','--num_workers','2']
    print(' '.join(shlex.quote(c) for c in cmd))
