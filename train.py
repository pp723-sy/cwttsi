import os
import sys
import gc
import warnings
import time
import math
import argparse
import torch
import json
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
import pynvml
from utils.metrics import evaluate_performance
from utils.random_seed import set_seed
from utils.logger import Logger
from utils.multi_epoch_dataloader import MultiEpochsDataLoader
from load_data import load_dataset
from model_convnext_stban import ConvNeXtSTBAN
from experiment_protocol import temporal_split_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name: XMBRT, HZMetro, or BJMetro')
    parser.add_argument('--history_step', type=int, default=12)
    parser.add_argument('--target_step', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--loss', type=str, default='Huber', choices=['L1', 'L2', 'Huber'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--time_slots_per_day', type=int, default=216, help='Time slots per day (XMBRT/HZMetro: 216, BJMetro: 108)')
    parser.add_argument('--x_mode', type=str, default='cwt',
                        choices=['cwt', 'cwt_multiband', 'interp', 'multiscale'])
    parser.add_argument('--x_variant', type=str, default='')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--ablation_mode', type=str, default='full',
                        choices=['full', 'no_img', 'no_ts', 'no_fuse'],
                        help='Ablation variant: full CWT-TSI, no_img, no_ts, or no_fuse.')
    parser.add_argument('--fusion_init_conv', type=float, default=0.0)
    parser.add_argument('--fusion_init_sta', type=float, default=0.0)
    parser.add_argument('--local_focus_enabled', action='store_true')
    parser.add_argument('--local_focus_kernel', type=int, default=3)
    parser.add_argument('--local_focus_temperature', type=float, default=1.0)
    parser.add_argument('--local_focus_topk', type=int, default=0)
    parser.add_argument('--halo_local_enabled', action='store_true')
    parser.add_argument('--halo_window_size', type=int, default=7)
    parser.add_argument('--halo_num_heads', type=int, default=4)
    parser.add_argument('--route_set_dim', type=int, default=3,
                        help='Bottleneck/routing token count in STBAN attention.')
    parser.add_argument('--train_subset_ratio', type=float, default=1.0)
    parser.add_argument('--val_subset_ratio', type=float, default=1.0)
    parser.add_argument('--subset_seed', type=int, default=1)
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (FP16)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers for parallel data loading')
    parser.add_argument('--resume_from', type=str, default='',
                        help='Path to .pt model checkpoint to resume training from')
    parser.add_argument('--split_protocol', choices=['window_ratio', 'target_time'], default='window_ratio')
    parser.add_argument('--loss_scale', choices=['raw', 'standardized'], default='raw')
    return parser.parse_args()


def sample_tensor_subset(*tensors, ratio=1.0, seed=1):
    if ratio >= 1.0:
        return tensors
    if ratio <= 0.0:
        raise ValueError('subset ratio must be in (0, 1].')
    total = tensors[0].shape[0]
    keep = max(1, int(total * ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:keep]
    return tuple(tensor[indices] for tensor in tensors)


args = parse_args()
set_seed(args.seed)
warnings.filterwarnings('ignore')
save_file_name = args.run_name if args.run_name else time.strftime('%Y%m%d-%H%M%S', time.localtime())
os.makedirs('./log', exist_ok=True)
os.makedirs('./save', exist_ok=True)
sys.stdout = Logger('./log/' + save_file_name + '.log')
os.makedirs('save/' + save_file_name, exist_ok=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
is_virtual_env = sys.prefix != sys.base_prefix
print('Python executable:', sys.executable)
print('Python version:', sys.version.replace('\n', ' '))
print('Python prefix:', sys.prefix)
print('Python base prefix:', sys.base_prefix)
print('Virtual environment active:', is_virtual_env)
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Torch CUDA version:', torch.version.cuda)
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
dataset_name = args.dataset_name
history_step = args.history_step
target_step = args.target_step
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
early_stop_patience = args.early_stop_patience
loss = args.loss
run_dir = f'save/{save_file_name}'

start_time = time.time()
x_mode = args.x_mode if not args.x_variant else args.x_variant
x, y, ts, te = load_dataset(dataset_name, history_step, target_step, x_mode=x_mode, img_size=args.img_size)
if x is None:
    raise FileNotFoundError(
        f'Dataset files are missing. Please run preprocess script first for {dataset_name}, x_mode={x_mode}.'
    )

matrix_width = x.shape[2]
matrix_height = x.shape[3]
train_idx, val_idx, test_idx = temporal_split_indices(
    len(x), history_step, target_step, args.split_protocol
)
train_idx = torch.as_tensor(train_idx, dtype=torch.long)
val_idx = torch.as_tensor(val_idx, dtype=torch.long)
test_idx = torch.as_tensor(test_idx, dtype=torch.long)
x_train, y_train, ts_train, te_train = (v[train_idx].float() for v in (x, y, ts, te))
x_val, y_val, ts_val, te_val = (v[val_idx].float() for v in (x, y, ts, te))
x_test, y_test, ts_test, te_test = (v[test_idx].float() for v in (x, y, ts, te))

x_train, y_train, ts_train, te_train = sample_tensor_subset(
    x_train, y_train, ts_train, te_train,
    ratio=args.train_subset_ratio,
    seed=args.subset_seed,
)
x_val, y_val, ts_val, te_val = sample_tensor_subset(
    x_val, y_val, ts_val, te_val,
    ratio=args.val_subset_ratio,
    seed=args.subset_seed + 997,
)

print('Train subset ratio:', args.train_subset_ratio)
print('Val subset ratio:', args.val_subset_ratio)
print('Subset seed:', args.subset_seed)
print('Split protocol:', args.split_protocol)
print('Loss scale:', args.loss_scale)
mean, std = torch.mean(torch.flatten(ts_train)), torch.std(torch.flatten(ts_train))
print('Mean:', mean, 'Std:', std)
ts_train = (ts_train - mean) / std
ts_val = (ts_val - mean) / std
ts_test = (ts_test - mean) / std
train_dataset = TensorDataset(x_train, y_train, ts_train, te_train)
val_dataset = TensorDataset(x_val, y_val, ts_val, te_val)
test_dataset = TensorDataset(x_test, y_test, ts_test, te_test)
print('Train dataset:', len(train_dataset))
print('Validate dataset:', len(val_dataset))
print('Test dataset:', len(test_dataset))
print('Batch size:', batch_size)
num_workers = args.num_workers
train_loader = MultiEpochsDataLoader(
    torch.utils.data.TensorDataset(x_train, y_train, ts_train, te_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True if num_workers > 0 else False,
)
val_loader = MultiEpochsDataLoader(
    torch.utils.data.TensorDataset(x_val, y_val, ts_val, te_val),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if num_workers > 0 else False,
)
test_loader = MultiEpochsDataLoader(
    torch.utils.data.TensorDataset(x_test, y_test, ts_test, te_test),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if num_workers > 0 else False,
)
load_end_time = time.time()
load_time = load_end_time - start_time
print('Load data time: {:.4f}s'.format(load_time))

print('Model: ConvNeXtSTBAN')
print('Ablation mode:', args.ablation_mode)
num_nodes = y.shape[1]
image_channels = x.shape[1]
model = ConvNeXtSTBAN(
    num_nodes=num_nodes,
    predict_length=target_step,
    history_step=history_step,
    image_channels=image_channels,
    time_slots_per_day=args.time_slots_per_day,
    fusion_init_conv=args.fusion_init_conv,
    fusion_init_sta=args.fusion_init_sta,
    local_focus_enabled=args.local_focus_enabled,
    local_focus_kernel=args.local_focus_kernel,
    local_focus_temperature=args.local_focus_temperature,
    local_focus_topk=args.local_focus_topk,
    halo_local_enabled=args.halo_local_enabled,
    halo_window_size=args.halo_window_size,
    halo_num_heads=args.halo_num_heads,
    route_set_dim=args.route_set_dim,
    ablation_mode=args.ablation_mode,
).float()
model = model.to(device)
if args.resume_from:
    print(f'Resuming from checkpoint: {args.resume_from}')
    saved_model = torch.load(args.resume_from, map_location=device, weights_only=False)
    if hasattr(saved_model, 'state_dict'):
        model.load_state_dict(saved_model.state_dict())
    else:
        model.load_state_dict(saved_model)
    del saved_model
    gc.collect()
    torch.cuda.empty_cache()
    print('Checkpoint loaded successfully')
optimizer = optim.AdamW(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
print('Loss:', loss)
print('X mode:', x_mode)
print('Ablation mode:', args.ablation_mode)
print('Fusion init (conv, sta):', args.fusion_init_conv, args.fusion_init_sta)
print('Local focus enabled:', args.local_focus_enabled)
if args.local_focus_enabled:
    print('Local focus config (kernel, temperature, topk):',
          args.local_focus_kernel, args.local_focus_temperature, args.local_focus_topk)
print('Halo local enabled:', args.halo_local_enabled)
if args.halo_local_enabled:
    print('Halo local config (window, heads):', args.halo_window_size, args.halo_num_heads)
print('Route set dim:', args.route_set_dim)
if loss == 'L1':
    loss_func = nn.L1Loss()
elif loss == 'L2':
    loss_func = nn.MSELoss()
elif loss == 'Huber':
    loss_func = nn.HuberLoss()
else:
    loss_func = None

def compute_loss(prediction, target):
    if args.loss_scale == 'standardized':
        local_mean = mean.to(prediction.device)
        local_std = std.to(prediction.device)
        return loss_func((prediction - local_mean) / local_std, (target - local_mean) / local_std)
    return loss_func(prediction, target)

set_seed(args.seed)

print('>>> Start training...')
use_amp = args.amp and torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
print('AMP enabled:', use_amp)
print('Num workers:', num_workers)
handle = None
if torch.cuda.is_available():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mean, std = mean.to(device), std.to(device)
best_val_loss = math.inf
best_epoch = 0
early_stop_counter = 0
train_loss_history = []
val_loss_history = []
for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)
    train_target_list = torch.tensor([])
    val_target_list = torch.tensor([])
    train_predict_list = torch.tensor([])
    val_predict_list = torch.tensor([])

    model.train()
    train_start_time = time.time()
    for train_data, train_target, train_ts, train_te in tqdm(train_loader, leave=False, disable=True):
        train_target_list = torch.cat([train_target_list, train_target])
        train_data = train_data.to(device)
        train_target = train_target.to(device)
        train_ts = train_ts.to(device)
        train_te = train_te.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            train_predict = model(train_data, train_ts, train_te, mean, std)
            train_loss_iter = compute_loss(train_predict, train_target)
        train_predict_list = torch.cat([train_predict_list, train_predict.detach().cpu()])
        scaler.scale(train_loss_iter).backward()
        scaler.step(optimizer)
        scaler.update()
    train_end_time = time.time()
    print('Train time: {:.4f}s'.format(train_end_time - train_start_time))
    train_loss = compute_loss(train_predict_list, train_target_list)
    train_loss = torch.nan_to_num(train_loss, nan=torch.tensor(float('inf')))
    train_loss_history.append(float(train_loss))
    print('Train loss: {:.4f}'.format(train_loss))
    if handle is not None:
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('Memory used: {:.4f}G'.format(meminfo.used / 1024 ** 3))

    model.eval()
    with torch.no_grad():
        val_start_time = time.time()
        for val_data, val_target, val_ts, val_te in val_loader:
            val_target_list = torch.cat([val_target_list, val_target])
            val_data = val_data.to(device)
            val_ts = val_ts.to(device)
            val_te = val_te.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                val_predict = model(val_data, val_ts, val_te, mean, std)
            val_predict_list = torch.cat([val_predict_list, val_predict.detach().cpu()])
        val_end_time = time.time()
        print('Validate time: {:.4f}s'.format(val_end_time - val_start_time))
        val_loss = compute_loss(val_predict_list, val_target_list)
        if torch.isnan(val_loss):
            print('Validate loss is NaN, replace with inf for early-stop logic')
            val_loss = torch.tensor(float('inf'))
        val_loss_history.append(float(val_loss))
        print('Validate loss: {:.4f}'.format(val_loss))

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print('Current LR: {:.6f}'.format(current_lr))

    if val_loss < best_val_loss:
        early_stop_counter = 0
        print('Epoch {} validate loss decreases from {:.4f} to {:.4f}'.format(epoch + 1, best_val_loss, val_loss))
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model, f'save/{save_file_name}/{best_epoch}.pt')
        print('Train | Validate:')
        mae_train, rmse_train, mmape_train = evaluate_performance(train_target_list.flatten(0, 1),
                                                                  train_predict_list.flatten(0, 1))
        mae_val, rmse_val, mmape_val = evaluate_performance(val_target_list.flatten(0, 1),
                                                            val_predict_list.flatten(0, 1))
        print('MAE: {:.4f} | {:.4f}    RMSE: {:.4f} | {:.4f}    Masked_MAPE: {:.4f}% | {:.4f}%'
              .format(mae_train, mae_val, rmse_train, rmse_val, mmape_train * 100, mmape_val * 100))
    else:
        early_stop_counter += 1
        print('*** Early stop: {} / {}'.format(early_stop_counter, early_stop_patience))
        if early_stop_counter >= early_stop_patience:
            print('*** Early stopped ***')
            break

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if best_epoch == 0:
    best_epoch = num_epochs
    torch.save(model, f'save/{save_file_name}/{best_epoch}.pt')
    print('No valid best epoch found, saved fallback checkpoint at epoch {}'.format(best_epoch))

saved_net = torch.load(f'save/{save_file_name}/{best_epoch}.pt', map_location=device, weights_only=False)
saved_net.eval()
test_target_list = torch.tensor([])
test_predict_list = torch.tensor([])
infer_time_list = []
with torch.no_grad():
    for test_data, test_target, test_ts, test_te in test_loader:
        test_target_list = torch.cat([test_target_list, test_target])
        test_data = test_data.to(device)
        test_ts = test_ts.to(device)
        test_te = test_te.to(device)
        if torch.cuda.is_available():
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.amp.autocast('cuda', enabled=use_amp):
                test_predict = saved_net(test_data, test_ts, test_te, mean, std)
            end_event.record()
            torch.cuda.synchronize()
            infer_time_list.append(start_event.elapsed_time(end_event))
        else:
            infer_start = time.time()
            test_predict = saved_net(test_data, test_ts, test_te, mean, std)
            infer_time_list.append((time.time() - infer_start) * 1000)
        test_predict_list = torch.cat([test_predict_list, test_predict.detach().cpu()])
    print('Infer time: {:.4f}ms'.format(sum(infer_time_list)))
    print('Infer time per sample: {:.4f}ms'.format(sum(infer_time_list) / len(test_loader)))

print('### Training finished, best epoch: {}'.format(best_epoch))
test_loss = compute_loss(test_predict_list, test_target_list)
print('Test loss: {:.4f}'.format(test_loss))
print('Test:')
test_target_list = test_target_list.flatten(0, 1)
test_predict_list = test_predict_list.flatten(0, 1)
per_step_metrics = []
for i in range(target_step):
    mae_test_step, rmse_test_step, mmape_test_step = evaluate_performance(test_target_list[:, i],
                                                                          test_predict_list[:, i])
    print('Step: {}    MAE: {:.4f}    RMSE: {:.4f}    Masked_MAPE: {:.4f}%'.format(
        i + 1, mae_test_step, rmse_test_step, mmape_test_step * 100))
    per_step_metrics.append({
        'step': i + 1,
        'mae': float(mae_test_step),
        'rmse': float(rmse_test_step),
        'mape': float(mmape_test_step * 100),
    })
mae_test, rmse_test, mmape_test = evaluate_performance(test_target_list, test_predict_list)
print('Average:    MAE: {:.4f}    RMSE: {:.4f}    Masked_MAPE: {:.4f}%'.format(mae_test, rmse_test,
                                                                               mmape_test * 100))

np.save(f'{run_dir}/test_target.npy', test_target_list.cpu().numpy())
np.save(f'{run_dir}/test_predict.npy', test_predict_list.cpu().numpy())
np.savez(f'{run_dir}/history.npz', train_loss=np.array(train_loss_history), val_loss=np.array(val_loss_history))
metrics = {
    'dataset': dataset_name,
    'x_mode': x_mode,
    'ablation_mode': args.ablation_mode,
    'environment': {
        'python_executable': sys.executable,
        'python_version': sys.version,
        'python_prefix': sys.prefix,
        'python_base_prefix': sys.base_prefix,
        'is_virtual_env': is_virtual_env,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'torch_cuda_version': torch.version.cuda,
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',
    },
    'train_subset_ratio': args.train_subset_ratio,
    'val_subset_ratio': args.val_subset_ratio,
    'subset_seed': args.subset_seed,
    'local_focus_enabled': args.local_focus_enabled,
    'local_focus_kernel': args.local_focus_kernel,
    'local_focus_temperature': args.local_focus_temperature,
    'local_focus_topk': args.local_focus_topk,
    'halo_local_enabled': args.halo_local_enabled,
    'halo_window_size': args.halo_window_size,
    'halo_num_heads': args.halo_num_heads,
    'route_set_dim': args.route_set_dim,
    'history_step': history_step,
    'target_step': target_step,
    'num_nodes': int(num_nodes),
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'seed': args.seed,
    'split_protocol': args.split_protocol,
    'split_sizes': {
        'train': len(train_dataset),
        'validation': len(val_dataset),
        'test': len(test_dataset),
    },
    'loss_scale': args.loss_scale,
    'amp': use_amp,
    'best_epoch': int(best_epoch),
    'mae': float(mae_test),
    'rmse': float(rmse_test),
    'mape': float(mmape_test * 100),
    'per_step_metrics': per_step_metrics,
    'train_loss_history': train_loss_history,
    'val_loss_history': val_loss_history,
}
with open(f'{run_dir}/metrics.json', 'w', encoding='utf-8') as file:
    json.dump(metrics, file, ensure_ascii=False, indent=2)
print('Saved metrics:', f'{run_dir}/metrics.json')

end_time = time.time()
print('Run time: {:.4f}s'.format(end_time - start_time))
print('Save file name:', save_file_name)
print('### Completed ###')
