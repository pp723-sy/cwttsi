"""Replay the archived plotter; change only lettering in figures 9 and 12."""
from pathlib import Path
import ast
import json
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '02_图稿/原始实验结果图'
SOURCE = BASE / '04_生成代码与输入/原始稿生成链/chapter4_revision.py'
ORIGINAL = BASE / '02_高清源图/论文最终源图_图6-图12'
OUT = ROOT / '02_图稿/第四章原图标注调整_20260924'
QA = ROOT / 'tmp/ch4_original_figures_20260924/replay'
OUT.mkdir(exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
text = SOURCE.read_text(encoding='utf-8')
tree = ast.parse(text)
names = {'load_font', 'dashed_horizontal', 'draw_line_chart', 'generate_figures'}
code = '\n\n'.join(ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
aligned = json.loads((BASE / '04_生成代码与输入/论文最终对齐输入/common_test_alignment_audit.json').read_text(encoding='utf-8'))
evidence = {'full_runs': {}}
for ds, run in zip(['XMBRT', 'HZMetro', 'BJMetro'], ['xmbrt_full', 'hzmetro_full', 'bjmetro_full']):
    rd = ROOT / '03_可复现实验/save' / run
    saved = json.loads((rd / 'metrics.json').read_text(encoding='utf-8'))
    hist = np.load(rd / 'history.npz')
    evidence['full_runs'][ds] = {
        'metrics': aligned['aligned_hz']['CWT-TSI']['overall'] if ds == 'HZMetro' else {k: saved[k] for k in ['mae', 'rmse', 'mape']},
        'per_step_metrics': aligned['aligned_hz']['CWT-TSI']['per_step'] if ds == 'HZMetro' else saved['per_step_metrics'],
        'train_loss': hist['train_loss'].astype(float).tolist(),
        'val_loss': hist['val_loss'].astype(float).tolist(),
    }

def replay(source_code, directory):
    scope = dict(Path=Path, Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont,
                 FIGURE_DIR=directory, DATASETS=['XMBRT', 'HZMetro', 'BJMetro'],
                 METRICS=['mae', 'rmse', 'mape'], METRIC_LABELS={'mae': 'MAE', 'rmse': 'RMSE', 'mape': 'MAPE (%)'})
    exec(compile(source_code, str(SOURCE), 'exec'), scope)
    scope['generate_figures'](evidence)

replay(code, QA)
report = {'source_plotter': str(SOURCE), 'changes': 'Only font sizes and text/legend placement; redundant 12-step avg. text removed and explained accurately in the caption; data, axis ranges, plot coordinates, lines and colors unchanged.', 'figures': []}
for name in ['cwt_per_step_metrics.png', 'cwt_training_convergence.png']:
    original = np.asarray(Image.open(ORIGINAL / name).convert('RGB'))
    replayed = np.asarray(Image.open(QA / name).convert('RGB'))
    # Font rasterization differs between Pillow versions; verify every colored
    # curve/marker pixel, which is independent of that text rasterization.
    for rgb in [(47, 107, 154), (224, 122, 45), (61, 139, 109), (196, 71, 45)]:
        assert np.array_equal(np.all(original == rgb, axis=2), np.all(replayed == rgb, axis=2)), (name, rgb)
    report['figures'].append({'name': name, 'original_curve_pixels_reproduced_exactly': True, 'original_sha256': hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest()})

replacements = {
    'load_font(27, bold=True)': 'load_font(48, bold=True)',
    'load_font(21)': 'load_font(42)',
    'load_font(18)': 'load_font(40)',
    'load_font(17)': 'load_font(40)',
    'main_font = load_font(48, bold=True)': 'main_font = load_font(56, bold=True)',
    'y - 10)': 'y - 24)',
    'draw.text((plot_right - 145, y - 24), "12-step avg.", fill="#555555", font=small_font)': '# Overall reference line is defined in the manuscript caption.',
    'bottom - 30)': 'bottom - 24)',
    'plot_top - 34)': 'plot_top - 74)',
    'legend_y += 28': 'legend_y += 52',
}
adjusted = code
for before, after in replacements.items():
    assert before in adjusted, before
    adjusted = adjusted.replace(before, after)
replay(adjusted, OUT)
for r in report['figures']:
    r['adjusted_path'] = str(OUT / r['name'])
    r['adjusted_sha256'] = hashlib.sha256((OUT / r['name']).read_bytes()).hexdigest()
    assert hashlib.sha256((ORIGINAL / r['name']).read_bytes()).hexdigest() == r['original_sha256']
    original = np.asarray(Image.open(ORIGINAL / r['name']).convert('RGB'))
    changed = np.asarray(Image.open(OUT / r['name']).convert('RGB'))
    for rgb in [(47, 107, 154), (224, 122, 45), (61, 139, 109), (196, 71, 45)]:
        a, b = np.all(original == rgb, axis=2), np.all(changed == rgb, axis=2)
        if r['name'] == 'cwt_training_convergence.png':
            # Only the legend sample for the second line moves vertically.
            for left in [45, 1205, 2365]:
                a[170:240, left+115:left+154] = False
                b[170:240, left+115:left+154] = False
        assert np.array_equal(a, b), (r['name'], rgb, 'curve changed')
    r['adjusted_curve_pixels_unchanged'] = True
(OUT / 'provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
