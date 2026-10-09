import argparse
import glob
import subprocess
import sys
from pathlib import Path


DATASET_TIME_SLOTS = {
    "XMBRT": 216,
    "HZMetro": 216,
    "BJMetro": 108,
}
ABLATION_MODES = ["no_img", "no_ts", "no_fuse", "full"]


def dataset_is_ready(root, dataset, history_step, target_step, img_size, x_mode):
    root = Path(root)
    suffix = f"hs{history_step}_ts{target_step}"
    base = root / "datasets" / dataset
    required = [
        base / "y" / f"y_{suffix}.npy",
        base / "ts" / f"ts_{suffix}.npy",
        base / "te" / f"te_{suffix}.npy",
    ]
    if x_mode == "cwt":
        x_pattern = str(base / "x" / f"x_{suffix}_i{img_size}_cwt*.npy")
    else:
        x_pattern = str(base / "x" / f"x_{suffix}_i{img_size}_{x_mode}*.npy")
    return all(path.exists() for path in required) and bool(glob.glob(x_pattern))


def build_train_command(
    python_executable,
    dataset,
    ablation_mode,
    epochs,
    batch_size,
    learning_rate,
    train_subset_ratio,
    val_subset_ratio,
    seed,
    history_step=12,
    target_step=12,
    img_size=128,
    x_mode="cwt",
    extra_args=None,
):
    run_name = f"{dataset.lower()}_{ablation_mode}"
    command = [
        python_executable,
        "train.py",
        "--dataset_name", dataset,
        "--time_slots_per_day", str(DATASET_TIME_SLOTS[dataset]),
        "--history_step", str(history_step),
        "--target_step", str(target_step),
        "--img_size", str(img_size),
        "--x_mode", x_mode,
        "--num_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--train_subset_ratio", str(train_subset_ratio),
        "--val_subset_ratio", str(val_subset_ratio),
        "--seed", str(seed),
        "--subset_seed", str(seed),
        "--ablation_mode", ablation_mode,
        "--run_name", run_name,
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def build_preprocess_command(python_executable, dataset, history_step, target_step, img_size, x_mode):
    return [
        python_executable,
        "build_xmbrt_hzmetro_bjmetro_dataset.py",
        "--dataset", dataset,
        "--history_step", str(history_step),
        "--target_step", str(target_step),
        "--img_size", str(img_size),
        "--x_mode", x_mode,
    ]


def run_command(command, dry_run=False):
    printable = subprocess.list2cmdline(command)
    print(printable)
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CWT-TSI Table 7 ablation experiments.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_TIME_SLOTS.keys()),
                        choices=list(DATASET_TIME_SLOTS.keys()))
    parser.add_argument("--modes", nargs="+", default=ABLATION_MODES, choices=ABLATION_MODES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--history_step", type=int, default=12)
    parser.add_argument("--target_step", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--x_mode", type=str, default="cwt", choices=["cwt", "cwt_multiband", "interp"])
    parser.add_argument("--train_subset_ratio", type=float, default=1.0)
    parser.add_argument("--val_subset_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prepare_missing", action="store_true",
                        help="Preprocess missing dataset files before training.")
    parser.add_argument("--rerun", action="store_true",
                        help="Run even if save/<dataset>_<mode>/metrics.json already exists.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--out_dir", type=str, default="outputs/ablation_table7")
    parser.add_argument("extra_train_args", nargs=argparse.REMAINDER,
                        help="Extra train.py arguments after --, e.g. -- --amp --num_workers 2")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(".")
    python_executable = sys.executable
    extra_train_args = args.extra_train_args
    if extra_train_args and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]

    missing = [
        dataset for dataset in args.datasets
        if not dataset_is_ready(root, dataset, args.history_step, args.target_step, args.img_size, args.x_mode)
    ]
    if missing and not args.prepare_missing:
        print("Missing preprocessed dataset files:", ", ".join(missing))
        print("Run again with --prepare_missing, or run build_xmbrt_hzmetro_bjmetro_dataset.py first.")
        return 2

    for dataset in missing:
        command = build_preprocess_command(
            python_executable,
            dataset,
            args.history_step,
            args.target_step,
            args.img_size,
            args.x_mode,
        )
        run_command(command, dry_run=args.dry_run)

    for dataset in args.datasets:
        for mode in args.modes:
            metrics_path = root / "save" / f"{dataset.lower()}_{mode}" / "metrics.json"
            if metrics_path.exists() and not args.rerun:
                print(f"Skip existing run: {metrics_path}")
                continue
            command = build_train_command(
                python_executable=python_executable,
                dataset=dataset,
                ablation_mode=mode,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                train_subset_ratio=args.train_subset_ratio,
                val_subset_ratio=args.val_subset_ratio,
                seed=args.seed,
                history_step=args.history_step,
                target_step=args.target_step,
                img_size=args.img_size,
                x_mode=args.x_mode,
                extra_args=extra_train_args,
            )
            run_command(command, dry_run=args.dry_run)

    summary_command = [
        python_executable,
        "summarize_experiments.py",
        "--root", ".",
        "--out_dir", args.out_dir,
    ]
    run_command(summary_command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
