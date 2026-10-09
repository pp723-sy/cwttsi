# References and external dependencies

The author identifies the CWT-TSI model/preprocessing/experiment work as their own work. This release contains that scope and excludes baseline implementations and vendor archives. Packaging assigns no new project-wide license.

The archive cites ConvNeXt, SSTBAN and MM-TSI as methodological sources. The image backbone is a local GroupNorm ConvNeXt adaptation with a forecasting head. DropPath is provided by installed timm. PyTorch, timm, NumPy, PyWavelets, Matplotlib and Pillow are installed dependencies; their source is not bundled.

The archived `conv2d_` and `FC` class ASTs match [SSTBAN model/sstban_model.py](https://github.com/guoshnBJTU/SSTBAN/blob/30bafd9d27c99cf7efe311c76ac145b1b9f7fc0c/model/sstban_model.py) exactly; attention and calendar classes also show direct adaptations. Those definitions have been removed from the public file. `external_sstban.py` loads selected blocks from the user's separately acquired hash-checked file, with the archived supervised default mask and device-aware embedding adaptation. See `EXTERNAL_DEPENDENCIES.md`. The inspected upstream root/README exposed no explicit license grant. This documentation grants no rights to upstream source; resolve dependency use terms with its provider where required.

The archive does not record an exact MM-TSI source-code URL/commit, and this audit did not establish a direct code match to MM-TSI. No MM-TSI source archive is distributed. The missing identifier is a provenance limitation, not proof that every local line derives from MM-TSI.

Fonts are not bundled: rendering uses system fonts or font resources supplied by installed Matplotlib. Comparison tables are numerical outputs. Raw observations, adjacency, targets, predictions and checkpoints are excluded because provider redistribution conditions have not been documented.
