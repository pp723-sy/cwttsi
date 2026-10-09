import os
import glob
import math
import numpy as np
import torch


def _load_new_layout(dataset, history_step, target_step, img_size=128, x_mode='cwt'):
    suffix = f'hs{history_step}_ts{target_step}'
    base_path = f'./datasets/{dataset}'
    x_path = f'{base_path}/x/x_{suffix}_i{img_size}_{x_mode}.npy'
    if not os.path.exists(x_path):
        # Try to find generated variant files (e.g. cwt_flatten_s1-128, cwtmb_flatten_b1-32_33-64_65-128).
        if x_mode == 'cwt':
            pattern = f'{base_path}/x/x_{suffix}_i{img_size}_cwt_*.npy'
            candidates = sorted(glob.glob(pattern))
            if candidates:
                x_path = candidates[0]
                print(f'Auto-discovered CWT variant: {x_path}')
        elif x_mode == 'cwt_multiband':
            pattern = f'{base_path}/x/x_{suffix}_i{img_size}_cwtmb_*.npy'
            candidates = sorted(glob.glob(pattern))
            if candidates:
                x_path = candidates[0]
                print(f'Auto-discovered CWT multiband variant: {x_path}')
        # Also try legacy filename without x_mode suffix
        if not os.path.exists(x_path):
            legacy_x_path = f'{base_path}/x/x_{suffix}_i{img_size}.npy'
            if os.path.exists(legacy_x_path):
                x_path = legacy_x_path
    y_path = f'{base_path}/y/y_{suffix}.npy'
    ts_path = f'{base_path}/ts/ts_{suffix}.npy'
    te_path = f'{base_path}/te/te_{suffix}.npy'
    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(ts_path) and os.path.exists(te_path)):
        return None, None, None, None
    print('Reading data from new layout')
    print('X directory:', x_path)
    print('Y directory:', y_path)
    print('TS directory:', ts_path)
    print('TE directory:', te_path)
    x = torch.from_numpy(np.load(x_path))
    y = torch.from_numpy(np.load(y_path))
    ts = torch.from_numpy(np.load(ts_path))
    te = torch.from_numpy(np.load(te_path))
    print('X:', x.shape)
    print('Y:', y.shape)
    print('TS:', ts.shape)
    print('TE:', te.shape)
    return x, y, ts, te


def load_x(dataset, target_step):
    print('Reading data')
    if dataset == 'bjmetro':
        data_path = f'./datasets/img_{dataset}_8_102060120_hs12_128.npy'
    else:
        data_path = f'./datasets/img_{dataset}_8_5103060_hs12_128.npy'
    print('Data directory:', data_path)
    x = np.load(data_path)
    x = x[: len(x) - (target_step - 1), :, :]
    img_resolution = int(math.sqrt(x.shape[2]))
    x = x.reshape((len(x), x.shape[1], img_resolution, img_resolution))
    print('X:', x.shape)
    x = torch.from_numpy(x)
    return x


def load_y(dataset, history_step, target_step):
    print('Reading target')
    target_path = f'./datasets/y_{dataset}_hs{history_step}_ts{target_step}.npy'
    print('Target directory:', target_path)
    y = np.load(target_path)
    print('Y:', y.shape)
    y = torch.from_numpy(y)
    return y


def load_ts(dataset, history_step, target_step):
    print('Reading time-series')
    ts_path = f'./datasets/ts_{dataset}_hs{history_step}_ts{target_step}.npy'
    print('Time-series directory:', ts_path)
    ts = np.load(ts_path)
    print('Time-series:', ts.shape)
    ts = torch.from_numpy(ts)
    return ts


def load_te(dataset, history_step, target_step):
    print('Reading TE')
    te_path = f'./datasets/te_{dataset}.npy'
    print('TE directory:', te_path)
    te_origin = np.load(te_path)
    te = []
    if history_step > 12:
        padding = np.zeros((history_step - 12, 2))
        te_origin = np.concatenate([padding, te_origin], axis=0)
    for i in range(history_step, te_origin.shape[0] - target_step + 1):
        ste_temp = [te_origin[i - history_step: i, :], te_origin[i: i + target_step, :]]
        ste_temp = np.concatenate(ste_temp)
        te.append(ste_temp)
    te = np.array(te)
    print('TE:', te.shape)
    te = torch.from_numpy(te)
    return te


def load_dataset(dataset, history_step, target_step, x_mode='cwt', img_size=128):
    print('Dataset:', dataset)
    print('History step:', history_step)
    print('Predict step:', target_step)
    # Try new layout first for all supported datasets
    if dataset in ['METR-LA', 'PEMS-BAY', 'XMBRT', 'HZMetro', 'BJMetro']:
        x, y, ts, te = _load_new_layout(dataset, history_step, target_step, img_size=img_size, x_mode=x_mode)
        if x is not None:
            return x, y, ts, te
        print(f'New layout not found for {dataset}, trying legacy loading...')
    # Fallback to legacy loading for metro/BRT datasets
    if dataset == 'XMBRT':
        x = load_x('xmbrt', target_step)
        y = load_y('xmbrt', history_step, target_step)
        ts = load_ts('xmbrt', history_step, target_step)
        te = load_te('xmbrt', history_step, target_step)
    elif dataset == 'HZMetro':
        x = load_x('hzmetro', target_step)
        y = load_y('hzmetro', history_step, target_step)
        ts = load_ts('hzmetro', history_step, target_step)
        te = load_te('hzmetro', history_step, target_step)
    elif dataset == 'BJMetro':
        x = load_x('bjmetro', target_step)
        y = load_y('bjmetro', history_step, target_step)
        ts = load_ts('bjmetro', history_step, target_step)
        te = load_te('bjmetro', history_step, target_step)
    else:
        x, y, ts, te = None, None, None, None
    return x, y, ts, te


# load_dataset('XMBRT', 12, 12)
# load_dataset('HZMetro', 12, 12)
# load_dataset('BJMetro', 12, 12)
