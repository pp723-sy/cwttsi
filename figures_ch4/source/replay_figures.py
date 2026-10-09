"""Replay the archived plot logic from packaged inputs, without training or editing Word.

This helper was added during source retrieval on 2026-10-03. Original scripts
in fig05_07, fig08_11 and fig09_10 are kept byte-for-byte unchanged.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import sys

PACKAGE = Path(__file__).resolve().parents[1]
DATASETS = ['XMBRT', 'HZMetro', 'BJMetro']
METRICS = ['mae', 'rmse', 'mape']


def source_module(path, skip_assignments, skip_results_import=False):
    """Keep original functions; replace only project-specific directory globals."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.If):
            continue  # main guard: invoke main explicitly after globals are supplied
        if isinstance(node, ast.ImportFrom) and node.module == 'results' and skip_results_import:
            continue
        if isinstance(node, ast.Assign):
            names = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if names & skip_assignments:
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr=='mkdir':
            continue  # make the isolated replay directories ourselves
        body.append(node)
    return compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec')


def render_comparison(output):
    path = PACKAGE/'source/fig05_07/plots.py'
    scope = dict(__file__=str(path), __name__='archived_comparison',
                 ROOT=PACKAGE, OUT=output, DATASETS=DATASETS, METRICS=METRICS,
                 MODELS=['LSTM','ASTGCN-r','GraphWaveNet','STSGCN-L','StemGNN','SyncG4','ASTGNN','CWT-TSI'])
    exec(source_module(path, set(), True), scope)
    scope['main']()


def render_ablation(output):
    path = PACKAGE/'source/fig09_10/plots.py'
    scope = dict(__file__=str(path), __name__='archived_ablation', ROOT=PACKAGE,
                 OUT=output, FIG=output/'figures', SOURCE=PACKAGE/'data/ablation_table_common_test.csv')
    exec(source_module(path, {'ROOT','OUT','FIG','SOURCE'}), scope)
    scope['main']()


def render_raster(output):
    from PIL import Image, ImageDraw, ImageFont
    path = PACKAGE/'source/fig08_11/chapter4_revision.py'
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    names = {'load_font','dashed_horizontal','draw_line_chart','generate_figures'}
    selected = '\n\n'.join(ast.get_source_segment(text,n) for n in tree.body
                           if isinstance(n,ast.FunctionDef) and n.name in names)
    adjustment = ast.parse((PACKAGE/'source/fig08_11/adjust_original_ch4_labels.py').read_text(encoding='utf-8'))
    assignment = next(n for n in adjustment.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='replacements' for t in n.targets))
    replacements = ast.literal_eval(assignment.value)
    for before, after in replacements.items():
        if before not in selected:
            raise ValueError('Archived label adjustment could not be applied: '+before)
        selected = selected.replace(before,after)
    scope = dict(Path=Path,Image=Image,ImageDraw=ImageDraw,ImageFont=ImageFont,
                 FIGURE_DIR=output/'figures',DATASETS=DATASETS,METRICS=METRICS,
                 METRIC_LABELS={'mae':'MAE','rmse':'RMSE','mape':'MAPE (%)'})
    exec(compile(selected,str(path),'exec'),scope)
    evidence=json.loads((PACKAGE/'data/fig08_fig11_evidence.json').read_text(encoding='utf-8'))
    # Preserve Windows Arial when present; use bundled DejaVu on other hosts.
    original_load_font = scope['load_font']
    def portable_font(size, bold=False):
        if Path('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf').exists():
            return original_load_font(size, bold)
        try:
            from matplotlib.font_manager import findfont, FontProperties
            return ImageFont.truetype(findfont(FontProperties(family='DejaVu Sans', weight='bold' if bold else 'normal')), size=size)
        except ImportError:
            return ImageFont.load_default(size=size)
    scope['load_font'] = portable_font
    scope['generate_figures'](evidence)
    for old,new in [('cwt_per_step_metrics.png','Fig08_逐步预测误差.png'),
                    ('cwt_training_convergence.png','Fig11_训练验证损失.png')]:
        (output/'figures'/old).rename(output/'figures'/new)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=['comparison','ablation','raster','all'],default='all')
    parser.add_argument('--output',type=Path,default=PACKAGE/'replay_output')
    args=parser.parse_args()
    output=args.output.resolve()
    if output==PACKAGE or PACKAGE.is_relative_to(output):
        raise ValueError('Choose a new subdirectory or separate directory for replay outputs.')
    groups={'comparison','ablation','raster'} if args.group=='all' else {args.group}
    selected_numbers=[]
    if 'comparison' in groups:selected_numbers.extend([5,6,7])
    if 'ablation' in groups:selected_numbers.extend([9,10])
    if 'raster' in groups:selected_numbers.extend([8,11])
    # Protect the retrieved originals and existing replay results from overwriting.
    for n in selected_numbers:
        if any((output/'figures').glob(f'Fig{n:02d}_*')):
            raise FileExistsError('Output already contains this figure; choose a new output directory.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    (output/'data').mkdir(exist_ok=True)
    import shutil
    for name in ['comparison_updated.csv','summary.json']:
        shutil.copy2(PACKAGE/'data'/name,output/'data'/name)
    if 'comparison' in groups:render_comparison(output)
    if 'ablation' in groups:render_ablation(output)
    if 'raster' in groups:render_raster(output)
    from PIL import Image
    record=json.loads((PACKAGE/'manifest.json').read_text(encoding='utf-8'))
    checks=[]
    for f in record['figures']:
        if f['figure'] not in selected_numbers:continue
        name=next(p for p in f['files'] if p.endswith('.png'))
        original=PACKAGE/name;generated=output/'figures'/name
        with Image.open(original) as a,Image.open(generated) as b:
            pixels_equal=a.size==b.size and a.convert('RGBA').tobytes()==b.convert('RGBA').tobytes()
        checks.append(dict(figure=f['figure'],path=str(generated),
                           file_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                           byte_equal=original.read_bytes()==generated.read_bytes(),
                           pixels_equal=pixels_equal))
    versions=dict(python=sys.version.split()[0],Pillow=__import__('PIL').__version__)
    if 'comparison' in groups or 'ablation' in groups:
        versions.update(matplotlib=__import__('matplotlib').__version__,numpy=__import__('numpy').__version__)
    report=dict(helper_added_on='2026-10-03',original_plot_logic_preserved=True,release_font_fallback_added=True,versions=versions,checks=checks)
    (output/f'replay_verification_{args.group}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not all(c['pixels_equal'] for c in checks):
        raise SystemExit('Replayed pixels differ; retrieved originals remain unchanged. Check fonts and dependency versions.')


if __name__=='__main__':main()
