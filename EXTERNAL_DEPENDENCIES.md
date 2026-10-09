# Separately obtained SSTBAN building blocks

The public package does not redistribute SSTBAN source. Its supervised STBAN adapter uses selected blocks from [guoshnBJTU/SSTBAN](https://github.com/guoshnBJTU/SSTBAN), fixed to commit `30bafd9d27c99cf7efe311c76ac145b1b9f7fc0c`. Obtain this repository directly from the upstream author subject to the applicable terms:

```sh
git clone https://github.com/guoshnBJTU/SSTBAN.git external/SSTBAN
git -C external/SSTBAN checkout 30bafd9d27c99cf7efe311c76ac145b1b9f7fc0c
```

Do not upload `external/` as part of this repository. Alternatively set `CWT_TSI_SSTBAN_SOURCE` to your separately obtained `model/sstban_model.py`. The loader requires SHA256 `cc81572f3f1dcb01f1c59fcfd45c9f1b4df653a3d2f27b1c7b4cc5599e76424a` and does not download source automatically.

The archived own-model file contained `conv2d_` and `FC` blocks that are identical to upstream when parsed as Python AST, plus adaptations of the attention and embedding blocks. Those class definitions have been removed from the public model file. `external_sstban.py` selects upstream blocks in memory, gives mask arguments the archived supervised default of None, and uses a device-aware one-hot calendar adapter. No upstream self-supervised model or training code is included or invoked. The reproduction check compares the resulting own-model state dictionaries and predictions against all 12 original own-model checkpoints using genuine archived inputs.

The upstream root/README inspected on 2026-10-09 did not expose a software license grant. These instructions identify the dependency and avoid bundling it; they do not grant any rights to it. The author must resolve dependency access/use conditions with its provider where needed. The local code archive remains the complete record of original implementations.
