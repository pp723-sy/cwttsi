# cwttsi

CWT-TSI passenger-flow forecasting with multi-node continuous wavelet images and a node-time branch. This repository contains the model, CSV preprocessing, training/evaluation and the three branch/fusion ablation variants.

## Dataset source

The XMBRT, HZMetro, and BJMetro datasets follow the processed passenger-flow datasets used by Lv et al. in the MM-TSI study:

Lv, Q., Liu, L., Yang, R. & Wang, Y. **Multimodal urban traffic flow prediction based on multi-scale time series imaging.** *Pattern Recognition* **164**, 111499 (2025). [https://doi.org/10.1016/j.patcog.2025.111499](https://doi.org/10.1016/j.patcog.2025.111499).

Dataset access provided by the MM-TSI authors:

- [MM-TSI author repository](https://github.com/lvqinzhi/MM-TSI) — see its "Getting started / Download Data" section.
- [MM-TSI dataset download on Google Drive](https://drive.google.com/drive/folders/1DtLKnZl2Td6jXK7vOsTxn0y-3o1q5DwT?usp=drive_link) — the link published in that repository.
- [MM-TSI paper](https://doi.org/10.1016/j.patcog.2025.111499).

Please cite Lv et al. when using the datasets obtained from their distribution. The download link is maintained by the MM-TSI authors; CWT-TSI links to that source without rehosting the data. This citation identifies the dataset source followed in our study. Please consult the paper and original data providers for applicable use terms. This repository does not redistribute raw passenger-flow observations or claim that the authors of CWT-TSI collected these datasets.

## Environment and dependency setup

Use Python 3.10 and run commands from the repository root. The author environment used Python 3.10.20 and PyTorch 2.11.0; `requirements.txt` retains the audited torch/torchvision, timm and wavelet versions. Install a suitable CPU or CUDA PyTorch build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -r requirements.txt
```

CUDA is optional: `train.py` selects an available CUDA device or CPU, and enables AMP only when requested and CUDA is available. No particular computer, GPU model, Anaconda installation or NVIDIA monitoring library is required. A fresh installation on another machine has not been verified.

The temporal branch requires separately acquired SSTBAN building blocks at the pinned revision documented in [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md). They are not redistributed or downloaded automatically. The public adapter preserves the archived supervised mask and calendar embedding behavior. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Data preprocessing

Obtain the data from the linked MM-TSI distribution and prepare time-by-node CSVs named `XMBRT_4320x44.csv`, `HZMetro_5616x80.csv` and `BJMetro_2700x276.csv` under `ts2img/origin_data/`. Check the downloaded package format before preparing these inputs. This release does not certify that every file in the external download is byte-identical to the author-local input files.

```sh
python build_xmbrt_hzmetro_bjmetro_dataset.py --dataset all --history_step 12 --target_step 12 --img_size 128 --x_mode cwt --cwt_wavelet morl --cwt_signal_mode flatten --cwt_scale_min 1 --cwt_scale_max 128
```

The builder creates `datasets/<dataset>/{x,y,ts,te,splits,meta}/`. The retained loader also understands earlier dataset layouts. Do not mix different image variants in the same dataset directory when reproducing a run.

## Training and ablation settings

Print the 12 own-model/ablation commands without executing training:

```sh
python submission_tools/print_retraining_commands.py
```

`run_index.json` contains only dataset and run identifiers. Use fresh run names and execute the printed commands from the repository root. Modes are `full`, `no_img` (temporal branch), `no_ts` (image branch) and `no_fuse` (fixed 0.5/0.5 fusion).

For a dry run of the batch launcher:

```sh
python run_ablation_experiments.py --prepare_missing --dry_run -- --amp --num_workers 2
```

Remove `--dry_run` to execute. The launcher checks for existing metrics and skips completed runs unless `--rerun` is supplied. `--prepare_missing` requires the CSV inputs above. Historical settings: 12 input/output steps, batch 16, learning rate 0.001, maximum 100 epochs, patience 20, seed 1, raw-unit Huber loss, AMP on CUDA, AdamW default weight decay 0.01 and cosine scheduler `T_max=10`. XMBRT/HZMetro use 216 daily slots and BJMetro 108. The image branch uses ConvNeXt depths [3,3,9,3], widths [96,192,384,768] and GroupNorm; the temporal branch uses L3, K16, d8 and routing size3. Local-focus and halo options were disabled in the reported runs. Early complete CLI records are missing; some settings were reconstructed from code defaults and archive notes.

Training writes `log/<run>.log` and `save/<run>/`, including checkpoints, test predictions/targets, histories and metrics. These newly generated outputs can be used to evaluate a run; archived author outputs are not bundled. `--resume_from` loads model weights only, not optimizer/scheduler/RNG state, so it does not provide an exact interruption-resume protocol. Load only trusted checkpoint files.

The historical `window_ratio` split shares boundary target times. `target_time` is available for new experiments but does not reproduce the original split. HZMetro reported full/ablation comparisons used first-test-window trimming for common-test alignment. This packaging cleanup does not change those historical results or claim new repeated-seed evidence.

## Code checks

After obtaining the separate temporal dependency:

```sh
python -m unittest discover -s tests -v
```

Checks cover temporal-branch forecasts/gradients and safe import with spawned data-loader workers. They do not substitute for retraining on the authorized datasets. Model structure, preprocessing, split calculations, optimizer, loss, metrics and seed behavior are preserved. Manuscript drawing, Excel export, private archive verification and workstation metadata have been removed. No project-wide reuse license has been assigned.
