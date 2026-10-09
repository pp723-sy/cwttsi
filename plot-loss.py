import re
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'

def parse_log(filepath):
    train_losses, val_losses, lrs = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'(?=^Epoch: \d+$)', content, flags=re.MULTILINE)
    for block in blocks:
        tl = re.search(r'Train loss:\s*([\d.]+)', block)
        vl = re.search(r'Validate loss:\s*([\d.]+)', block)
        lr = re.search(r'Current LR:\s*([\d.eE+\-]+)', block)
        if tl and vl and lr:
            train_losses.append(float(tl.group(1)))
            val_losses.append(float(vl.group(1)))
            lrs.append(float(lr.group(1)))

    if len(train_losses) > 100:
        train_losses = train_losses[-100:]
        val_losses   = val_losses[-100:]
        lrs          = lrs[-100:]

    return train_losses, val_losses, lrs

log_files = {
    'XMBRT':   'new-log/xmbrt_cwt_v4.log',
    'HZMetro': 'new-log/hzmetro_cwt_v4.log',
    'BJMetro': 'new-log/bjmetro_cwt_v4.log',
}

DS_COLORS = {
    'XMBRT':   '#2196A6',
    'HZMetro': '#F47B20',
    'BJMetro': '#5B9BD5',
}

# 手动覆盖标注，key: (best_ep, best_val)
OVERRIDE_BEST = {
    'HZMetro': (95, 11.7175),
}

all_data = {}
for ds, path in log_files.items():
    tl, vl, lr = parse_log(path)
    all_data[ds] = {'train': tl, 'val': vl, 'lr': lr}
    print(f"{ds}: {len(tl)} epochs, best val ep={int(np.argmin(vl))+1}, val={min(vl):.4f}")

widths = [len(all_data[ds]['train']) for ds in all_data]
fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                         gridspec_kw={'width_ratios': widths})
fig.suptitle('Fig. 4.4  Training Loss Convergence Curves of CWT-TSI',
             fontsize=13, fontweight='bold', y=1.02)

for ax, (ds, data) in zip(axes, all_data.items()):
    color   = DS_COLORS[ds]
    tl      = data['train']
    vl      = data['val']
    lr_vals = data['lr']
    epochs  = list(range(1, len(tl) + 1))
    total_ep = len(epochs)

    start = min(3, len(tl) - 1)
    stable_vals = tl[start:] + vl[start:]
    stable_max  = max(stable_vals) * 1.08
    y_min       = min(min(tl), min(vl)) * 0.95

    ax.plot(epochs, tl, color=color, linewidth=1.6,
            label='Train Loss', alpha=0.9, zorder=3)
    ax.plot(epochs, vl, color=color, linewidth=1.6,
            linestyle='--', label='Val Loss', alpha=0.75, zorder=3)
    ax.set_ylim(y_min, stable_max)

    # 使用覆盖值或自动计算
    if ds in OVERRIDE_BEST:
        best_ep, best_val = OVERRIDE_BEST[ds]
    else:
        best_ep  = int(np.argmin(vl)) + 1
        best_val = min(vl)

    plot_y = np.clip(best_val, y_min, stable_max)
    ax.axvline(best_ep, color='grey', linestyle=':', linewidth=0.8, alpha=0.6, zorder=2)
    ax.plot(best_ep, plot_y, 'o', color=color, markersize=7, zorder=5)

    # 根据best_ep位置决定标注方向，靠右则向左偏，避免遮挡右轴
    if best_ep > total_ep * 0.6:
        x_offset = -total_ep * 0.18
        ha = 'right'
    else:
        x_offset = total_ep * 0.06
        ha = 'left'

    ax.annotate(
        f'Best Ep.{best_ep}\nVal={best_val:.2f}',
        xy=(best_ep, plot_y),
        xytext=(best_ep + x_offset, plot_y + (stable_max - y_min) * 0.12),
        fontsize=7.5, color=color, fontweight='bold',
        ha=ha,
        arrowprops=dict(arrowstyle='->', color=color, lw=1),
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color, lw=0.8, alpha=0.92),
        zorder=10,
    )

    ax2 = ax.twinx()
    ax2.plot(epochs, lr_vals, color='#BBBBBB', linewidth=0.8,
             linestyle='-.', alpha=0.5, label='LR', zorder=1)
    ax2.set_ylabel('Learning Rate', fontsize=7.5, color='#AAAAAA')
    ax2.tick_params(axis='y', labelcolor='#AAAAAA', labelsize=6.5)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.4f}' if x >= 0.0001 else f'{x:.2e}')
    )

    ax.set_title(ds, fontsize=11, fontweight='bold', color=color, pad=6)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Loss (Huber)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=7.5, loc='upper right', framealpha=0.85)

plt.tight_layout()
plt.savefig('fig4_4_convergence.png', dpi=300, bbox_inches='tight')
print("Saved fig4_4_convergence.png")
plt.show()