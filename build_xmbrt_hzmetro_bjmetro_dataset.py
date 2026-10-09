"""
Preprocess XMBRT / HZMetro / BJMetro datasets from raw CSV files.
Generates CWT (or interp) image features in the new layout format,
compatible with load_data._load_new_layout().

CSV files are located in ts2img/origin_data/.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import pywt
from tqdm import tqdm


DATASET_CONFIG = {
    "XMBRT": {
        "csv_file": "ts2img/origin_data/XMBRT_4320x44.csv",
        "num_nodes": 44,
        "num_times_day": 216,
        "num_days": 20,
        "slots_per_day": 216,     # 5-min intervals, 24*60/5 = 288? Actually 216 from to_matrix.py
        "time_scales": "5103060",
    },
    "HZMetro": {
        "csv_file": "ts2img/origin_data/HZMetro_5616x80.csv",
        "num_nodes": 80,
        "num_times_day": 216,
        "num_days": 26,
        "slots_per_day": 216,
        "time_scales": "5103060",
    },
    "BJMetro": {
        "csv_file": "ts2img/origin_data/BJMetro_2700x276.csv",
        "num_nodes": 276,
        "num_times_day": 108,
        "num_days": 25,
        "slots_per_day": 108,
        "time_scales": "102060120",
    },
}


def ensure_dirs(base_dir: str):
    module_dirs = {
        "x": os.path.join(base_dir, "x"),
        "y": os.path.join(base_dir, "y"),
        "ts": os.path.join(base_dir, "ts"),
        "te": os.path.join(base_dir, "te"),
        "splits": os.path.join(base_dir, "splits"),
        "meta": os.path.join(base_dir, "meta"),
    }
    for path in module_dirs.values():
        os.makedirs(path, exist_ok=True)
    return module_dirs


def build_x_from_ts_interp(ts: np.ndarray, img_size: int, chunk_size: int = 256) -> np.ndarray:
    """Build image features via bilinear interpolation (same as METR-LA/PEMS-BAY interp mode)."""
    x_out = np.empty((ts.shape[0], 1, img_size, img_size), dtype=np.float16)
    for start in tqdm(range(0, ts.shape[0], chunk_size), desc="Building x(interp)", leave=False):
        end = min(start + chunk_size, ts.shape[0])
        ts_chunk = ts[start:end]
        img_chunk = torch.from_numpy(ts_chunk).float().unsqueeze(1)
        min_val = img_chunk.amin(dim=(2, 3), keepdim=True)
        max_val = img_chunk.amax(dim=(2, 3), keepdim=True)
        img_chunk = (img_chunk - min_val) / (max_val - min_val + 1e-8)
        img_chunk = F.interpolate(img_chunk, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x_out[start:end] = img_chunk.cpu().numpy().astype(np.float16)
    return x_out


def parse_cwt_bands(bands: str) -> list[tuple[int, int]]:
    parsed = []
    for item in bands.split(","):
        start, end = item.strip().split("-")
        start, end = int(start), int(end)
        if start <= 0 or end < start:
            raise ValueError(f"Invalid CWT band: {item}")
        parsed.append((start, end))
    return parsed


def make_signal(sample: np.ndarray, signal_mode: str) -> np.ndarray:
    """Convert one (num_nodes, history_step) sample into a 1D signal for imaging."""
    if signal_mode == "mean":
        signal = sample.mean(axis=0)
    else:
        signal = sample.reshape(-1)
    signal = signal.astype(np.float32)
    return (signal - signal.mean()) / (signal.std() + 1e-8)


def scalogram_to_image(coeffs: np.ndarray, img_size: int) -> torch.Tensor:
    tf = np.log1p(np.abs(coeffs).astype(np.float32))
    tf_tensor = torch.from_numpy(tf).unsqueeze(0).unsqueeze(0)
    tf_tensor = F.interpolate(tf_tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tf_tensor = tf_tensor.squeeze(0)
    min_val = tf_tensor.amin()
    max_val = tf_tensor.amax()
    return (tf_tensor - min_val) / (max_val - min_val + 1e-8)


def build_x_from_ts_cwt(ts: np.ndarray, img_size: int, wavelet: str = "morl",
                        signal_mode: str = "flatten", scale_min: int = 1, scale_max: int = 128) -> np.ndarray:
    """Build a single-channel CWT scalogram image."""
    x_out = np.empty((ts.shape[0], 1, img_size, img_size), dtype=np.float16)
    scales = np.arange(scale_min, scale_max + 1, dtype=np.float32)
    for idx in tqdm(range(ts.shape[0]), desc="Building x(cwt)", leave=False):
        signal = make_signal(ts[idx], signal_mode)
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
        x_out[idx] = scalogram_to_image(coeffs, img_size).cpu().numpy().astype(np.float16)
    return x_out


def build_x_from_ts_cwt_multiband(ts: np.ndarray, img_size: int, wavelet: str = "morl",
                                  signal_mode: str = "flatten",
                                  bands: str = "1-32,33-64,65-128") -> np.ndarray:
    """Build a multi-channel CWT image; each channel corresponds to one scale band."""
    parsed_bands = parse_cwt_bands(bands)
    x_out = np.empty((ts.shape[0], len(parsed_bands), img_size, img_size), dtype=np.float16)
    for idx in tqdm(range(ts.shape[0]), desc="Building x(cwt_multiband)", leave=False):
        signal = make_signal(ts[idx], signal_mode)
        for band_idx, (scale_min, scale_max) in enumerate(parsed_bands):
            scales = np.arange(scale_min, scale_max + 1, dtype=np.float32)
            coeffs, _ = pywt.cwt(signal, scales, wavelet)
            x_out[idx, band_idx] = scalogram_to_image(coeffs, img_size).cpu().numpy().astype(np.float16)
    return x_out


def build_te_from_slots(num_total_steps: int, slots_per_day: int, num_samples: int,
                        history_step: int, target_step: int) -> np.ndarray:
    """
    Build time encoding (day_of_week, time_slot) for each sample.
    Since CSV data doesn't have timestamps, we synthesize TE based on slot indices.
    """
    # Generate synthetic time encoding: (day_of_week, time_slot_in_day)
    te_origin = np.zeros((num_total_steps, 2), dtype=np.int64)
    for i in range(num_total_steps):
        day_idx = i // slots_per_day
        slot_idx = i % slots_per_day
        te_origin[i, 0] = day_idx % 7       # day_of_week
        te_origin[i, 1] = slot_idx           # time_slot
    # Build sliding windows for TE
    te = []
    for i in range(history_step, num_total_steps - target_step + 1):
        ste_temp = np.concatenate([
            te_origin[i - history_step: i, :],
            te_origin[i: i + target_step, :]
        ])
        te.append(ste_temp)
    te = np.array(te, dtype=np.int64)
    return te


def preprocess_one(dataset_name: str, history_step: int, target_step: int, img_size: int,
                   max_samples: int | None, x_mode: str,
                   cwt_wavelet: str, cwt_signal_mode: str, cwt_scale_min: int, cwt_scale_max: int,
                   cwt_bands: str):
    config = DATASET_CONFIG[dataset_name]
    csv_path = config["csv_file"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"\n=== Preprocessing {dataset_name} ===")
    print(f"CSV file: {csv_path}")
    # Read CSV: shape (num_total_steps, num_nodes), no header
    values = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    num_total_steps, num_nodes = values.shape
    print(f"Raw shape: {values.shape} (expected nodes: {config['num_nodes']})")
    assert num_nodes == config["num_nodes"], f"Node count mismatch: {num_nodes} != {config['num_nodes']}"

    window_len = history_step + target_step
    if num_total_steps < window_len:
        raise ValueError(f"Not enough timesteps: {num_total_steps} < {window_len}")

    # Build sliding windows: ts shape (N, num_nodes, history_step), y shape (N, num_nodes, target_step)
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_len, axis=0)
    # windows shape: (N, num_nodes, window_len)
    ts = windows[:, :, :history_step].astype(np.float32)  # (N, num_nodes, history_step)
    y = windows[:, :, history_step:].astype(np.float32)    # (N, num_nodes, target_step)

    # Build TE
    slots_per_day = config["slots_per_day"]
    te = build_te_from_slots(num_total_steps, slots_per_day, ts.shape[0], history_step, target_step)

    if max_samples is not None:
        ts = ts[:max_samples]
        y = y[:max_samples]
        te = te[:max_samples]

    num_samples = ts.shape[0]
    print(f"Samples: {num_samples}, Nodes: {num_nodes}")
    print(f"TS shape: {ts.shape}, Y shape: {y.shape}, TE shape: {te.shape}")

    # Build image features
    if x_mode == "cwt":
        x = build_x_from_ts_cwt(
            ts,
            img_size=img_size,
            wavelet=cwt_wavelet,
            signal_mode=cwt_signal_mode,
            scale_min=cwt_scale_min,
            scale_max=cwt_scale_max,
        )
    elif x_mode == "cwt_multiband":
        x = build_x_from_ts_cwt_multiband(
            ts,
            img_size=img_size,
            wavelet=cwt_wavelet,
            signal_mode=cwt_signal_mode,
            bands=cwt_bands,
        )
    else:
        x = build_x_from_ts_interp(ts, img_size=img_size)

    # Compute splits: 60/20/20
    train_end = int(num_samples * 0.6)
    val_end = int(num_samples * 0.8)
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, num_samples, dtype=np.int64)

    # Save in new layout
    base_dir = os.path.join("datasets", dataset_name)
    module_dirs = ensure_dirs(base_dir)

    suffix = f"hs{history_step}_ts{target_step}"
    x_tag = x_mode
    if x_mode == "cwt":
        x_tag = f"{x_mode}_{cwt_signal_mode}_s{cwt_scale_min}-{cwt_scale_max}"
    elif x_mode == "cwt_multiband":
        band_tag = cwt_bands.replace(",", "_")
        x_tag = f"cwtmb_{cwt_signal_mode}_b{band_tag}"
    x_path = os.path.join(module_dirs["x"], f"x_{suffix}_i{img_size}_{x_tag}.npy")
    y_path = os.path.join(module_dirs["y"], f"y_{suffix}.npy")
    ts_path = os.path.join(module_dirs["ts"], f"ts_{suffix}.npy")
    te_path = os.path.join(module_dirs["te"], f"te_{suffix}.npy")
    split_path = os.path.join(module_dirs["splits"], f"split_6_2_2_{suffix}.npz")
    meta_path = os.path.join(module_dirs["meta"], f"meta_{suffix}.json")

    np.save(x_path, x)
    np.save(y_path, y)
    np.save(ts_path, ts)
    np.save(te_path, te)
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    meta = {
        "dataset": dataset_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history_step": history_step,
        "target_step": target_step,
        "image_size": img_size,
        "slots_per_day": slots_per_day,
        "num_total_steps": int(num_total_steps),
        "num_nodes": int(num_nodes),
        "num_samples": int(num_samples),
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "csv_file": csv_path,
        "files": {
            "x": x_path,
            "y": y_path,
            "ts": ts_path,
            "te": te_path,
            "splits": split_path,
        },
        "x_mode": x_mode,
        "cwt": {
            "wavelet": cwt_wavelet,
            "signal_mode": cwt_signal_mode,
            "scale_min": cwt_scale_min,
            "scale_max": cwt_scale_max,
            "bands": cwt_bands,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print(f"Saved x: {x.shape} -> {x_path}")
    print(f"Saved y: {y.shape} -> {y_path}")
    print(f"Saved ts: {ts.shape} -> {ts_path}")
    print(f"Saved te: {te.shape} -> {te_path}")
    print(f"Saved splits: {split_path}")
    print(f"Saved meta: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess XMBRT/HZMetro/BJMetro from CSV for CWT-TSI.")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "XMBRT", "HZMetro", "BJMetro"])
    parser.add_argument("--history_step", type=int, default=12)
    parser.add_argument("--target_step", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--x_mode", type=str, default="cwt", choices=["cwt", "cwt_multiband", "interp"])
    parser.add_argument("--cwt_wavelet", type=str, default="morl")
    parser.add_argument("--cwt_signal_mode", type=str, default="flatten", choices=["flatten", "mean"])
    parser.add_argument("--cwt_scale_min", type=int, default=1)
    parser.add_argument("--cwt_scale_max", type=int, default=128)
    parser.add_argument("--cwt_bands", type=str, default="1-32,33-64,65-128",
                        help="Scale bands for cwt_multiband, e.g. 1-32,33-64,65-128")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    for dataset_name in datasets:
        preprocess_one(
            dataset_name=dataset_name,
            history_step=args.history_step,
            target_step=args.target_step,
            img_size=args.img_size,
            max_samples=args.max_samples,
            x_mode=args.x_mode,
            cwt_wavelet=args.cwt_wavelet,
            cwt_signal_mode=args.cwt_signal_mode,
            cwt_scale_min=args.cwt_scale_min,
            cwt_scale_max=args.cwt_scale_max,
            cwt_bands=args.cwt_bands,
        )


if __name__ == "__main__":
    main()
