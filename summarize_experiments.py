import argparse
import csv
import html
import json
import re
from pathlib import Path


AVERAGE_RE = re.compile(
    r"Average:\s+MAE:\s+([0-9.]+)\s+RMSE:\s+([0-9.]+)\s+Masked_MAPE:\s+([0-9.]+)%"
)
STEP_RE = re.compile(
    r"Step:\s+(\d+)\s+MAE:\s+([0-9.]+)\s+RMSE:\s+([0-9.]+)\s+Masked_MAPE:\s+([0-9.]+)%"
)
DATASET_RE = re.compile(r"Dataset:\s+([A-Za-z0-9_-]+)")
SAVE_RE = re.compile(r"Save file name:\s+(.+)")

DATASET_ORDER = {"XMBRT": 0, "HZMetro": 1, "BJMetro": 2}
ABLATION_ORDER = {"no_img": 0, "no_ts": 1, "no_fuse": 2, "full": 3}
ABLATION_LABEL = {
    "no_img": "No-Img",
    "no_ts": "No-TS",
    "no_fuse": "No-Fuse",
    "full": "CWT-TSI",
}


def infer_ablation_mode(name: str) -> str:
    lowered = name.lower().replace("-", "_")
    if "no_img" in lowered or "noimg" in lowered:
        return "no_img"
    if "no_ts" in lowered or "nots" in lowered:
        return "no_ts"
    if "no_fuse" in lowered or "nofuse" in lowered:
        return "no_fuse"
    if "full" in lowered or "cwt_tsi" in lowered or "cwt_v4" in lowered:
        return "full"
    return ""


def read_metrics_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    run_name = path.parent.name
    ablation_mode = data.get("ablation_mode") or infer_ablation_mode(run_name)
    return {
        "run": run_name,
        "dataset": data.get("dataset", "unknown"),
        "x_mode": data.get("x_mode", ""),
        "ablation_mode": ablation_mode,
        "train_subset_ratio": float(data.get("train_subset_ratio", 1.0)),
        "val_subset_ratio": float(data.get("val_subset_ratio", 1.0)),
        "mae": float(data.get("mae")),
        "rmse": float(data.get("rmse")),
        "mape": float(data.get("mape")),
        "source": str(path),
        "per_step_metrics": data.get("per_step_metrics", []),
    }


def infer_dataset(name: str) -> str:
    lowered = name.lower()
    if "xmbrt" in lowered:
        return "XMBRT"
    if "hzmetro" in lowered:
        return "HZMetro"
    if "bjmetro" in lowered:
        return "BJMetro"
    return "unknown"


def read_log(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    average = AVERAGE_RE.search(text)
    if not average:
        return None
    dataset_match = DATASET_RE.search(text)
    save_match = SAVE_RE.search(text)
    per_step = []
    for match in STEP_RE.finditer(text):
        per_step.append({
            "step": int(match.group(1)),
            "mae": float(match.group(2)),
            "rmse": float(match.group(3)),
            "mape": float(match.group(4)),
        })
    return {
        "run": save_match.group(1).strip() if save_match else path.stem,
        "dataset": dataset_match.group(1) if dataset_match else infer_dataset(path.stem),
        "x_mode": "",
        "ablation_mode": infer_ablation_mode(save_match.group(1).strip() if save_match else path.stem),
        "train_subset_ratio": 1.0,
        "val_subset_ratio": 1.0,
        "mae": float(average.group(1)),
        "rmse": float(average.group(2)),
        "mape": float(average.group(3)),
        "source": str(path),
        "per_step_metrics": per_step,
    }


def collect_runs(root: Path) -> list[dict]:
    runs = []
    seen = set()
    for path in sorted(root.glob("save/*/metrics.json")):
        item = read_metrics_json(path)
        runs.append(item)
        seen.add(item["run"])
    for path in sorted(root.glob("new-log/*.log")):
        item = read_log(path)
        if item and item["run"] not in seen:
            runs.append(item)
            seen.add(item["run"])
    return sorted(runs, key=lambda item: (item["dataset"], item["run"]))


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def write_csv(runs: list[dict], out_dir: Path) -> None:
    fields = ["dataset", "run", "x_mode", "ablation_mode", "mae", "rmse", "mape", "source"]
    with (out_dir / "experiment_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.get(field, "") for field in fields})


def table7_runs(runs: list[dict]) -> list[dict]:
    selected = {}
    for run in runs:
        mode = run.get("ablation_mode", "")
        is_full_data = (
            float(run.get("train_subset_ratio", 1.0)) == 1.0
            and float(run.get("val_subset_ratio", 1.0)) == 1.0
        )
        if mode in ABLATION_ORDER and is_full_data:
            key = (run["dataset"], mode)
            current = selected.get(key)
            if current is None or table7_preference_key(run) > table7_preference_key(current):
                selected[key] = run
    return sorted(
        selected.values(),
        key=lambda item: (
            DATASET_ORDER.get(item["dataset"], 99),
            item["dataset"],
            ABLATION_ORDER[item["ablation_mode"]],
            item["run"],
        ),
    )


def table7_preference_key(run: dict) -> tuple[int, int, str]:
    canonical_run = f'{run["dataset"].lower()}_{run["ablation_mode"]}'
    is_canonical = int(run["run"] == canonical_run)
    is_metrics_json = int("metrics.json" in run.get("source", ""))
    return is_canonical, is_metrics_json, run["run"]


def write_ablation_table_csv(runs: list[dict], out_dir: Path) -> None:
    fields = ["model", "dataset", "mae", "rmse", "mape", "run", "source"]
    with (out_dir / "ablation_table7.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in table7_runs(runs):
            writer.writerow({
                "model": ABLATION_LABEL[run["ablation_mode"]],
                "dataset": run["dataset"],
                "mae": f'{run["mae"]:.4f}',
                "rmse": f'{run["rmse"]:.4f}',
                "mape": f'{run["mape"]:.4f}',
                "run": run["run"],
                "source": run["source"],
            })


def write_ablation_table_markdown(runs: list[dict], out_dir: Path) -> None:
    lines = [
        "# 表 7：CWT-TSI 消融实验结果",
        "",
        "| 模型 | 数据集 | MAE | RMSE | MAPE (%) | run |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines[2] = "| 模型 | 数据集 | MAE | RMSE | MAPE (%) | run |"
    for run in table7_runs(runs):
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                ABLATION_LABEL[run["ablation_mode"]],
                run["dataset"],
                run["mae"],
                run["rmse"],
                run["mape"],
                esc(run["run"]),
            )
        )
    (out_dir / "ablation_table7.md").write_text("\n".join(lines), encoding="utf-8")


def axis_ticks(max_value: float, count: int = 5) -> list[float]:
    if max_value <= 0:
        return [0.0]
    return [max_value * i / count for i in range(count + 1)]


def write_bar_svg(runs: list[dict], metric: str, out_path: Path) -> None:
    width, height = 1180, 620
    left, right, top, bottom = 90, 30, 70, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max(run[metric] for run in runs) * 1.15
    bar_gap = 14
    bar_w = max(12, (plot_w - bar_gap * (len(runs) - 1)) / max(len(runs), 1))
    colors = {"XMBRT": "#3568a9", "HZMetro": "#d98b2b", "BJMetro": "#4b9363", "unknown": "#777777"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="38" font-family="Arial" font-size="26" font-weight="700" fill="#1f2933">实验对比：{metric.upper()}</text>',
    ]
    for tick in axis_ticks(max_value):
        y = top + plot_h - tick / max_value * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8d4cc" stroke-dasharray="4 6"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#58606b">{tick:.1f}</text>')
    for idx, run in enumerate(runs):
        x = left + idx * (bar_w + bar_gap)
        value = run[metric]
        bar_h = value / max_value * plot_h
        y = top + plot_h - bar_h
        color = colors.get(run["dataset"], colors["unknown"])
        label = f'{run["dataset"]} {run["run"]}'
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#1f2933">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height-112}" text-anchor="end" transform="rotate(-35 {x + bar_w/2:.1f} {height-112})" font-family="Arial" font-size="11" fill="#384250">{esc(label)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#4b5563"/>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_step_svg(runs: list[dict], dataset: str, out_path: Path) -> None:
    subset = [run for run in runs if run["dataset"] == dataset and run["per_step_metrics"]]
    if not subset:
        return
    width, height = 980, 560
    left, right, top, bottom = 85, 190, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [item["mae"] for run in subset for item in run["per_step_metrics"]]
    min_value, max_value = min(all_values) * 0.95, max(all_values) * 1.05
    span = max(max_value - min_value, 1e-6)
    palette = ["#3568a9", "#d98b2b", "#4b9363", "#8b5fbf", "#c44e52", "#4c78a8"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="38" font-family="Arial" font-size="26" font-weight="700" fill="#1f2933">{dataset}: MAE by prediction step</text>',
    ]
    for tick in axis_ticks(max_value - min_value):
        value = min_value + tick
        y = top + plot_h - (value - min_value) / span * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8d4cc" stroke-dasharray="4 6"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#58606b">{value:.1f}</text>')
    for step in range(1, 13):
        x = left + (step - 1) / 11 * plot_w
        parts.append(f'<text x="{x:.1f}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="12" fill="#58606b">{step}</text>')
    for run_idx, run in enumerate(subset):
        points = []
        for item in run["per_step_metrics"]:
            x = left + (item["step"] - 1) / 11 * plot_w
            y = top + plot_h - (item["mae"] - min_value) / span * plot_h
            points.append((x, y))
        color = palette[run_idx % len(palette)]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"/>')
        legend_y = top + run_idx * 24
        parts.append(f'<line x1="{width-right+25}" y1="{legend_y}" x2="{width-right+55}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-right+62}" y="{legend_y+4}" font-family="Arial" font-size="12" fill="#384250">{esc(run["run"])}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#4b5563"/>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_markdown(runs: list[dict], out_dir: Path) -> None:
    fields = ["dataset", "run", "x_mode", "ablation_mode", "mae", "rmse", "mape", "source"]
    lines = [
        "# CWT-TSI 实验汇总",
        "",
        "## 平均指标",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for run in runs:
        lines.append("| " + " | ".join(esc(run.get(field, "")) for field in fields) + " |")
    lines.extend([
        "",
        "## 组会解读建议",
        "",
        "- Local Focus：参考 DRFormer 的动态 token / 关键步选择思想，用于验证关键历史步是否有助于 STBAN 分支。",
        "- Halo Local Attention：参考时频图像局部纹理建模思想，用于验证局部 CWT 纹理是否有助于 ConvNeXt 分支。",
        "- CWT Multiband：参考多频段分解思想，将低/中/高尺度拆分为不同图像通道。",
        "- Route Set Dim：调整 STBAN bottleneck token 数量，观察精度与效率的权衡。",
    ])
    (out_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize CWT-TSI experiment metrics for group meeting visuals.")
    parser.add_argument("--root", type=str, default=".", help="Project root directory.")
    parser.add_argument("--out_dir", type=str, default="outputs/group_meeting")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(root)
    if not runs:
        raise RuntimeError("No metrics found. Run training first or keep logs under new-log/.")
    write_csv(runs, out_dir)
    write_ablation_table_csv(runs, out_dir)
    write_ablation_table_markdown(runs, out_dir)
    write_markdown(runs, out_dir)
    for metric in ["mae", "rmse", "mape"]:
        write_bar_svg(runs, metric, out_dir / f"avg_{metric}.svg")
    for dataset in sorted({run["dataset"] for run in runs}):
        write_step_svg(runs, dataset, out_dir / f"step_mae_{dataset}.svg")
    print(f"Saved summary to: {out_dir}")


if __name__ == "__main__":
    main()
