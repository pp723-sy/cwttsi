"""
适配 MM-TSI 原版三个数据集的 CWT 时频成像预处理脚本
将原版多尺度散点图成像替换为连续小波变换(CWT)时频成像
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pywt
from tqdm import tqdm

# ============================================================
# 数据集配置（从MM-TSI原版代码精确提取）
# ============================================================
DATASET_CONFIG = {
    "XMBRT": {
        "csv_file": "datasets_raw/XMBRT/XMBRT_4320x44.csv",
        "num_nodes": 44,
        "num_times_day": 216,   # 每天5分钟粒度，运营18小时
        "num_days": 20,
        "time_slots_per_day": 216,
        "time_granularity_min": 5,
        "description": "厦门快速公交系统，44个站点，5分钟粒度，共20天"
    },
    "HZMetro": {
        "csv_file": "datasets_raw/HZMetro/HZMetro_5616x80.csv",
        "num_nodes": 80,
        "num_times_day": 216,   # 每天5分钟粒度，运营18小时
        "num_days": 26,
        "time_slots_per_day": 216,
        "time_granularity_min": 5,
        "description": "杭州地铁，80个站点，5分钟粒度，共26天"
    },
    "BJMetro": {
        "csv_file": "datasets_raw/BJMetro/BJMetro_2700x276.csv",
        "num_nodes": 276,
        "num_times_day": 108,   # 每天10分钟粒度，运营18小时
        "num_days": 25,
        "time_slots_per_day": 108,
        "time_granularity_min": 10,
        "description": "北京地铁，276个站点，10分钟粒度，共25天"
    },
}

def build_te_index(num_times_day: int, num_days: int) -> np.ndarray:
    """
    构建时间编码数组 te_origin，shape=(总时间步, 2)
    第0列：星期几(0-6)
    第1列：当天第几个时间槽(0-num_times_day-1)
    对应MM-TSI原版 te_{dataset}.npy 的格式
    """
    total_steps = num_times_day * num_days
    time_slot = np.tile(np.arange(num_times_day, dtype=np.int64), num_days)
    # 假设从周一(0)开始
    day_of_week = np.repeat(np.arange(num_days, dtype=np.int64) % 7, num_times_day)
    te = np.stack([day_of_week, time_slot], axis=-1)
    return te

def build_te_windows(te_origin: np.ndarray,
                     history_step: int,
                     target_step: int) -> np.ndarray:
    """
    构建滑动窗口的te，对应MM-TSI原版 load_te() 函数逻辑
    输出shape: (M, history_step+target_step, 2)
    """
    te_list = []
    for i in range(history_step, te_origin.shape[0] - target_step + 1):
        ste_temp = np.concatenate([
            te_origin[i - history_step: i, :],
            te_origin[i: i + target_step, :]
        ], axis=0)
        te_list.append(ste_temp)
    return np.array(te_list, dtype=np.int64)

def build_ts_windows(values: np.ndarray,
                     history_step: int,
                     target_step: int) -> tuple:
    """
    构建 ts（历史序列）和 y（预测目标）的滑动窗口
    ts shape: (M, N, history_step)
    y  shape: (M, N, target_step)
    """
    window_len = history_step + target_step
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=window_len, axis=0
    )
    # windows shape: (M, N, window_len)
    ts = windows[:, :, :history_step].astype(np.float32)
    y = windows[:, :, history_step:].astype(np.float32)
    return ts, y

def build_x_cwt(ts: np.ndarray, img_size: int,
                wavelet: str = "morl",
                signal_mode: str = "flatten",
                scale_min: int = 1,
                scale_max: int = 128,
                chunk_size: int = 64) -> np.ndarray:
    """
    对每个样本的多节点历史时序做CWT时频成像
    ts shape: (M, N, history_step)
    输出 shape: (M, 1, img_size, img_size)，float16
    """
    M = ts.shape[0]
    x_out = np.empty((M, 1, img_size, img_size), dtype=np.float16)
    scales = np.arange(scale_min, scale_max + 1, dtype=np.float32)

    for idx in tqdm(range(M), desc="CWT时频成像", leave=True):
        sample = ts[idx]  # shape: (N, history_step)

        if signal_mode == "mean":
            signal = sample.mean(axis=0).astype(np.float32)
        else:  # flatten（默认）：所有节点时序首尾拼接
            signal = sample.reshape(-1).astype(np.float32)

        # z-score标准化
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        # CWT连续小波变换
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
        # coeffs shape: (num_scales, len(signal))

        # 对数幅值谱（类声谱图处理）
        tf = np.log1p(np.abs(coeffs).astype(np.float32))

        # 双线性插值到固定尺寸
        tf_tensor = torch.from_numpy(tf).unsqueeze(0).unsqueeze(0)
        tf_tensor = F.interpolate(
            tf_tensor, size=(img_size, img_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # min-max归一化到[0,1]
        min_val = tf_tensor.amin()
        max_val = tf_tensor.amax()
        tf_tensor = (tf_tensor - min_val) / (max_val - min_val + 1e-8)

        x_out[idx] = tf_tensor.cpu().numpy().astype(np.float16)

    return x_out

def ensure_dirs(base_dir: str) -> dict:
    dirs = {
        "x": os.path.join(base_dir, "x"),
        "y": os.path.join(base_dir, "y"),
        "ts": os.path.join(base_dir, "ts"),
        "te": os.path.join(base_dir, "te"),
        "splits": os.path.join(base_dir, "splits"),
        "meta": os.path.join(base_dir, "meta"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def preprocess_one(dataset_name: str, history_step: int, target_step: int,
                   img_size: int, max_samples: int | None,
                   cwt_wavelet: str, cwt_signal_mode: str,
                   cwt_scale_min: int, cwt_scale_max: int):

    cfg = DATASET_CONFIG[dataset_name]
    csv_path = cfg["csv_file"]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"原始CSV未找到: {csv_path}")

    print(f"\n=== 预处理 {dataset_name} ===")
    print(f"配置: {cfg['description']}")
    print(f"读取: {csv_path}")

    # 读取原始CSV（无表头，列=节点，行=时间步）
    df = pd.read_csv(csv_path, header=None)
    values = df.values.astype(np.float32)
    # values shape: (total_time_steps, num_nodes)
    num_time, num_nodes = values.shape
    print(f"原始数据形状: {values.shape}")

    # 验证节点数
    assert num_nodes == cfg["num_nodes"], \
        f"节点数不匹配: CSV={num_nodes}, 配置={cfg['num_nodes']}"

    # 构建时间编码
    te_origin = build_te_index(cfg["num_times_day"], cfg["num_days"])
    assert te_origin.shape[0] == num_time, \
        f"时间步不匹配: CSV={num_time}, te={te_origin.shape[0]}"

    # 构建滑动窗口
    print("构建滑动窗口...")
    ts, y = build_ts_windows(values, history_step, target_step)
    te = build_te_windows(te_origin, history_step, target_step)
    # ts: (M, N, 12), y: (M, N, 12), te: (M, 24, 2)
    num_samples = ts.shape[0]
    print(f"样本数: {num_samples}")
    print(f"ts shape: {ts.shape}, y shape: {y.shape}, te shape: {te.shape}")

    if max_samples is not None:
        ts = ts[:max_samples]
        y = y[:max_samples]
        te = te[:max_samples]
        num_samples = ts.shape[0]
        print(f"截取后样本数: {num_samples}")

    # CWT时频成像
    print(f"CWT参数: wavelet={cwt_wavelet}, "
          f"signal_mode={cwt_signal_mode}, "
          f"scales=[{cwt_scale_min},{cwt_scale_max}]")
    x = build_x_cwt(
        ts, img_size=img_size,
        wavelet=cwt_wavelet,
        signal_mode=cwt_signal_mode,
        scale_min=cwt_scale_min,
        scale_max=cwt_scale_max,
    )
    print(f"x shape: {x.shape}")

    # 数据集划分（6:2:2，与MM-TSI原版一致）
    train_end = int(num_samples * 0.6)
    val_end = int(num_samples * 0.8)
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, num_samples, dtype=np.int64)

    # 保存文件
    base_dir = os.path.join("datasets", dataset_name)
    dirs = ensure_dirs(base_dir)
    suffix = f"hs{history_step}_ts{target_step}"
    x_tag = f"cwt_{cwt_signal_mode}_s{cwt_scale_min}-{cwt_scale_max}"

    x_path = os.path.join(dirs["x"], f"x_{suffix}_i{img_size}_{x_tag}.npy")
    y_path = os.path.join(dirs["y"], f"y_{suffix}.npy")
    ts_path = os.path.join(dirs["ts"], f"ts_{suffix}.npy")
    te_path = os.path.join(dirs["te"], f"te_{suffix}.npy")
    split_path = os.path.join(dirs["splits"], f"split_6_2_2_{suffix}.npz")
    meta_path = os.path.join(dirs["meta"], f"meta_{suffix}_{x_tag}.json")

    np.save(x_path, x)
    np.save(y_path, y)
    np.save(ts_path, ts)
    np.save(te_path, te)
    np.savez(split_path, train_idx=train_idx,
             val_idx=val_idx, test_idx=test_idx)

    meta = {
        "dataset": dataset_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": cfg["description"],
        "history_step": history_step,
        "target_step": target_step,
        "image_size": img_size,
        "num_nodes": int(num_nodes),
        "num_time_steps": int(num_time),
        "num_samples": int(num_samples),
        "time_granularity_min": cfg["time_granularity_min"],
        "time_slots_per_day": cfg["time_slots_per_day"],
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx))
        },
        "cwt": {
            "wavelet": cwt_wavelet,
            "signal_mode": cwt_signal_mode,
            "scale_min": cwt_scale_min,
            "scale_max": cwt_scale_max,
        },
        "files": {
            "x": x_path, "y": y_path,
            "ts": ts_path, "te": te_path,
            "splits": split_path,
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 保存完成:")
    print(f"   x: {x.shape} → {x_path}")
    print(f"   y: {y.shape} → {y_path}")
    print(f"   ts: {ts.shape} → {ts_path}")
    print(f"   te: {te.shape} → {te_path}")
    print(f"   meta → {meta_path}")
    print(f"   划分: train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "XMBRT", "HZMetro", "BJMetro"])
    parser.add_argument("--history_step", type=int, default=12)
    parser.add_argument("--target_step", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cwt_wavelet", type=str, default="morl")
    parser.add_argument("--cwt_signal_mode", type=str, default="flatten",
                        choices=["flatten", "mean"])
    parser.add_argument("--cwt_scale_min", type=int, default=1)
    parser.add_argument("--cwt_scale_max", type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    datasets = (["XMBRT", "HZMetro", "BJMetro"]
                if args.dataset == "all" else [args.dataset])
    for ds in datasets:
        preprocess_one(
            dataset_name=ds,
            history_step=args.history_step,
            target_step=args.target_step,
            img_size=args.img_size,
            max_samples=args.max_samples,
            cwt_wavelet=args.cwt_wavelet,
            cwt_signal_mode=args.cwt_signal_mode,
            cwt_scale_min=args.cwt_scale_min,
            cwt_scale_max=args.cwt_scale_max,
        )

if __name__ == "__main__":
    main()