# cwttsi

CWT-TSI uses multi-node continuous wavelet images and the original node-time representation for passenger-flow forecasting. This repository provides the CWT-TSI model, dataset preprocessing, training, branch/fusion ablations, experimental settings, saved aggregate results, and figure reproduction tools.

## Dataset source

The XMBRT, HZMetro, and BJMetro datasets follow the processed passenger-flow datasets used by Lv et al. in the MM-TSI study:

Lv, Q., Liu, L., Yang, R. & Wang, Y. **Multimodal urban traffic flow prediction based on multi-scale time series imaging.** *Pattern Recognition* **164**, 111499 (2025). [https://doi.org/10.1016/j.patcog.2025.111499](https://doi.org/10.1016/j.patcog.2025.111499).

This citation identifies the dataset source followed in our study. Please consult that paper and the original data providers for access and applicable use terms. This repository does not redistribute raw passenger-flow observations or claim that the authors of CWT-TSI collected these datasets.

This package contains the CWT-TSI model and its three ablation variants, aggregate experiment results, and figure replay. It does not contain other researchers' model implementations or their locally modified baseline versions. Raw data, target/prediction arrays, checkpoints and font files are excluded. Baseline comparison CSVs contain reported numerical results only.

## Environment and external dependency

Run commands from this directory. Archive records: Python 3.10.20, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 5060 Laptop GPU. Use Python 3.10 and install a compatible PyTorch/torchvision pair from [official instructions](https://pytorch.org/get-started/locally/), then install `requirements.txt`. Pins describe the author environment; a fresh installation elsewhere was not tested. Follow `EXTERNAL_DEPENDENCIES.md` to obtain the SSTBAN building blocks separately before constructing the temporal branch. No source downloads automatically.

The archived own-model file included upstream building-block definitions. The public file removes those definitions and loads selected classes from a separately obtained, pinned and hash-checked file. Default mask=None and device-aware calendar adaptations preserve the archived supervised branch. This audited dependency pin was selected during packaging; it was not recorded during the old training runs. The original research archive remains intact.

## Authorized data and retraining

Obtain authorized time-by-node CSVs and put `XMBRT_4320x44.csv`, `HZMetro_5616x80.csv`, `BJMetro_2700x276.csv` in `ts2img/origin_data/`. These observations have no documented redistribution grant and are not provided here. Use a new working copy before generating tensors:

```sh
python build_xmbrt_hzmetro_bjmetro_dataset.py --dataset all --history_step 12 --target_step 12 --img_size 128 --x_mode cwt --cwt_wavelet morl --cwt_signal_mode flatten --cwt_scale_min 1 --cwt_scale_max 128
python submission_tools/print_retraining_commands.py
```

The builder creates `datasets/<dataset>/{x,y,ts,te,splits,meta}/`. The second command prints 12 own-model commands without executing them. Choose fresh run names. `train.py` modes: `full`, `no_img` (temporal branch), `no_ts` (image branch), `no_fuse` (fixed 0.5/0.5 fusion). XMBRT/HZMetro use 216 daily slots and BJMetro 108. `build_transit_dataset.py` is an alternative historical path layout; the builder above matches this package.

Historical settings: 12 input/output steps, batch 16, learning rate 0.001, maximum 100 epochs, patience 20, seed 1, raw-unit Huber and AMP, AdamW inherited weight decay 0.01, cosine scheduler T_max=10. Early runs lacked complete contemporaneous CLI records; some values are reconstructed from source defaults and archive notes. ConvNeXt: depths [3,3,9,3], widths [96,192,384,768], GroupNorm. STBAN: L3, K16, d8, routing size3. Local-focus and halo options were disabled in the archived own runs.

Reported results use chronological `window_ratio` splits with shared target times at adjacent boundaries. This release does not claim target-disjoint evaluation. `target_time` is available for new experiments but does not harmonize old loss/regularization settings. HZMetro full/ablation results require removing the first test window for common-test alignment. Saved runs are single runs, not repeated random seeds. Labels STSGCN-L/SyncG4 denote local simplified baselines; their implementations are outside this release.

`saved_metrics/` and sanitized `run_index.json` describe all 33 archived experiments. The 24 comparison and 12 ablation table rows reuse three full-model runs. Checkpoint paths refer to the private archive. Full/ablation checkpoints store whole model objects; only trusted author-local files should be loaded with `weights_only=False`.

## Verification and figure replay

```sh
python -m unittest discover -s tests -v
python submission_tools/verify_saved_results.py --archive /path/to/authorized_archive --output verification_output
python -m pip install -r requirements-plot.txt
python figures_ch4/source/replay_figures.py --group all --output replay_output
```

Full numerical verification needs archive directories `datasets/`, `save/`, `ts2img/`, `manuscript_data/`. The result verifier does not require baseline implementations. Figure replay uses included aggregate CSV/JSON inputs, without raw observations or checkpoints. Matplotlib draws figures5–7/9–10; Pillow draws8/11. Raster plots prefer Windows Arial; other systems use fonts provided by installed Matplotlib. Font/version differences can change pixels, which the helper reports with a nonzero exit. Existing selected figure outputs are protected. Original `figures_ch4/Fig*.png` files provide the references.

See `AUDIT_SUMMARY.md`, `THIRD_PARTY_NOTICES.md` and `release_inventory.json`. No project-wide reuse license has been selected. External building blocks are documented and not bundled. No long retraining or new reported scientific results were created. `FIGURE_NUMBERING.md` maps the archived plotting names to the reorganized Scientific Reports main and supplementary displays.
