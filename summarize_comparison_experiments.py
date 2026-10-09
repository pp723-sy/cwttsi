import argparse
import csv
import html
import json
import time
from pathlib import Path


DATASETS = ["XMBRT", "HZMetro", "BJMetro"]
BASELINES = ["LSTM", "ASTGCN-r", "GraphWaveNet", "GMAN", "STSGCN", "StemGNN", "STFGNN", "ASTGNN"]
MODEL_ORDER = BASELINES + ["CWT-TSI"]
CWT_FULL_RUNS = {
    "XMBRT": "xmbrt_full",
    "HZMetro": "hzmetro_full",
    "BJMetro": "bjmetro_full",
}


def safe_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "_")


def read_metrics(path: Path, model_name: str, dataset: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "model": model_name,
        "run": path.parent.name,
        "mae": float(data["mae"]),
        "rmse": float(data["rmse"]),
        "mape": float(data["mape"]),
        "best_epoch": int(data.get("best_epoch", 0)),
        "source": str(path),
        "per_step_metrics": data.get("per_step_metrics", []),
    }


def collect_results(root: Path) -> list[dict]:
    results = []
    for dataset in DATASETS:
        for model_name in BASELINES:
            path = root / "save" / f"comparison_{dataset.lower()}_{safe_model_name(model_name)}" / "metrics.json"
            if path.exists():
                results.append(read_metrics(path, model_name, dataset))
        cwt_path = root / "save" / CWT_FULL_RUNS[dataset] / "metrics.json"
        if cwt_path.exists():
            results.append(read_metrics(cwt_path, "CWT-TSI", dataset))
    return sorted(results, key=lambda item: (DATASETS.index(item["dataset"]), MODEL_ORDER.index(item["model"])))


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def write_csv(results: list[dict], out_dir: Path) -> None:
    fields = ["dataset", "model", "mae", "rmse", "mape", "best_epoch", "run", "source"]
    with (out_dir / "comparison_table.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({
                "dataset": item["dataset"],
                "model": item["model"],
                "mae": f'{item["mae"]:.4f}',
                "rmse": f'{item["rmse"]:.4f}',
                "mape": f'{item["mape"]:.4f}',
                "best_epoch": item["best_epoch"],
                "run": item["run"],
                "source": item["source"],
            })


def write_markdown(results: list[dict], out_dir: Path) -> None:
    lines = [
        "# 表 4.3：CWT-TSI 与基线模型整体性能对比",
        "",
        "| 数据集 | 模型 | MAE | RMSE | MAPE (%) | 最优轮次 | run |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f'| {item["dataset"]} | {item["model"]} | {item["mae"]:.4f} | {item["rmse"]:.4f} | '
            f'{item["mape"]:.4f} | {item["best_epoch"]} | {esc(item["run"])} |'
        )
    (out_dir / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")


def axis_ticks(max_value: float, count: int = 5) -> list[float]:
    return [max_value * i / count for i in range(count + 1)]


def write_metric_svg(results: list[dict], metric: str, out_path: Path) -> None:
    width, height = 1320, 680
    left, right, top, bottom = 85, 30, 70, 175
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max(item[metric] for item in results) * 1.15
    bar_gap = 6
    bar_w = max(7, (plot_w - bar_gap * (len(results) - 1)) / len(results))
    colors = {
        "XMBRT": "#2f6f9f",
        "HZMetro": "#b8792f",
        "BJMetro": "#4f8f61",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="40" font-family="Arial" font-size="25" font-weight="700" fill="#1f2933">Baseline comparison: {metric.upper()}</text>',
    ]
    for tick in axis_ticks(max_value):
        y = top + plot_h - tick / max_value * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8d4cc" stroke-dasharray="4 6"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#58606b">{tick:.1f}</text>')
    for idx, item in enumerate(results):
        x = left + idx * (bar_w + bar_gap)
        value = item[metric]
        bar_h = value / max_value * plot_h
        y = top + plot_h - bar_h
        color = "#9b3f3f" if item["model"] == "CWT-TSI" else colors[item["dataset"]]
        label = f'{item["dataset"]} {item["model"]}'
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#1f2933">{value:.1f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height-125}" text-anchor="end" transform="rotate(-48 {x + bar_w/2:.1f} {height-125})" font-family="Arial" font-size="10" fill="#384250">{esc(label)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#4b5563"/>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_step_svg(results: list[dict], dataset: str, out_path: Path) -> None:
    subset = [item for item in results if item["dataset"] == dataset and item["per_step_metrics"]]
    if not subset:
        return
    width, height = 1080, 620
    left, right, top, bottom = 80, 200, 70, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [step["mae"] for item in subset for step in item["per_step_metrics"]]
    min_value, max_value = min(values) * 0.95, max(values) * 1.05
    span = max(max_value - min_value, 1e-6)
    palette = ["#2f6f9f", "#b8792f", "#4f8f61", "#7c5fb0", "#c44e52", "#4c78a8", "#a35d2f", "#5b7f95", "#9b3f3f"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="40" font-family="Arial" font-size="25" font-weight="700" fill="#1f2933">{dataset}: step-wise MAE</text>',
    ]
    for tick in axis_ticks(max_value - min_value):
        value = min_value + tick
        y = top + plot_h - (value - min_value) / span * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8d4cc" stroke-dasharray="4 6"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#58606b">{value:.1f}</text>')
    for idx, item in enumerate(subset):
        points = []
        color = palette[idx % len(palette)]
        for step in item["per_step_metrics"]:
            x = left + (step["step"] - 1) / 11 * plot_w
            y = top + plot_h - (step["mae"] - min_value) / span * plot_h
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(points)}"/>')
        legend_y = top + idx * 22
        parts.append(f'<line x1="{width-right+20}" y1="{legend_y}" x2="{width-right+45}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<text x="{width-right+52}" y="{legend_y+4}" font-family="Arial" font-size="12" fill="#384250">{esc(item["model"])}</text>')
    for step in range(1, 13):
        x = left + (step - 1) / 11 * plot_w
        parts.append(f'<text x="{x:.1f}" y="{height-30}" text-anchor="middle" font-family="Arial" font-size="12" fill="#58606b">{step}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_audit(results: list[dict], out_dir: Path) -> None:
    expected = {(dataset, model) for dataset in DATASETS for model in MODEL_ORDER}
    actual = {(item["dataset"], item["model"]) for item in results}
    missing = sorted(expected - actual)
    lines = [
        "# 对比实验审查摘要",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 已收集结果数量：{len(results)} / {len(expected)}",
        f"- 缺失结果数量：{len(missing)}",
    ]
    if missing:
        lines.append("")
        lines.append("## 缺失结果")
        for dataset, model in missing:
            lines.append(f"- {dataset} / {model}")
    lines.append("")
    lines.append("## 已收集结果")
    for item in results:
        lines.append(f'- {item["dataset"]} / {item["model"]}: MAE={item["mae"]:.4f}, RMSE={item["rmse"]:.4f}, MAPE={item["mape"]:.4f}, source={item["source"]}')
    (out_dir / "audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize comparison experiments.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="outputs/comparison_table4_5")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results(root)
    write_csv(results, out_dir)
    write_markdown(results, out_dir)
    write_audit(results, out_dir)
    if results:
        for metric in ["mae", "rmse", "mape"]:
            write_metric_svg(results, metric, out_dir / f"comparison_{metric}.svg")
        for dataset in DATASETS:
            write_step_svg(results, dataset, out_dir / f"step_mae_{dataset}.svg")
    print(f"Collected {len(results)} results")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
