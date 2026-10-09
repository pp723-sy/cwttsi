import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

steps = list(range(1, 13))

step_data = {
    'XMBRT': {
        'MAE':  [5.3802,5.3109,5.3333,5.3585,5.3392,5.3364,5.3580,5.3817,5.3872,5.4111,5.4539,5.5374],
        'RMSE': [8.8227,8.6572,8.7168,8.7925,8.7322,8.6898,8.7135,8.7363,8.7402,8.7831,8.8590,8.9682],
        'MAPE': [36.09,35.96,36.01,36.12,35.98,36.24,36.71,36.72,36.63,36.48,36.63,37.57],
    },
    'HZMetro': {
        'MAE':  [13.3684,13.2908,13.1307,13.0855,13.2195,13.5428,13.8561,13.9939,14.0401,14.2067,14.4591,14.7582],
        'RMSE': [23.7507,23.5272,23.2332,23.1646,23.3697,24.0240,24.7919,25.2861,25.6104,26.1123,26.7570,27.6271],
        'MAPE': [27.03,26.94,26.52,26.38,26.33,26.55,26.91,27.18,27.18,27.06,27.69,27.82],
    },
    'BJMetro': {
        'MAE':  [21.1772,20.9803,21.0737,21.2065,21.2907,21.4615,21.6284,21.5936,21.7540,21.9495,22.0590,22.3169],
        'RMSE': [42.3561,41.8205,41.8362,41.8330,41.9799,42.3225,42.8204,42.7567,43.3735,44.0957,44.2508,44.5477],
        'MAPE': [20.70,20.35,20.83,21.55,21.91,22.04,22.02,22.00,22.02,22.41,23.55,25.31],
    },
}

datasets  = ['XMBRT', 'HZMetro', 'BJMetro']
metrics   = ['MAE', 'RMSE', 'MAPE']
m_labels  = ['MAE', 'RMSE', 'MAPE (%)']

best_step = {
    'XMBRT':   {'MAE': 6,  'RMSE': 6,  'MAPE': 2},
    'HZMetro': {'MAE': 4,  'RMSE': 4,  'MAPE': 5},
    'BJMetro': {'MAE': 2,  'RMSE': 2,  'MAPE': 2},
}

DS_COLORS = {'XMBRT': '#2196A6', 'HZMetro': '#F47B20', 'BJMetro': '#5B9BD5'}

# 每个子图标注的偏移方向 (dx, dy_ratio) — dy_ratio 是相对于数据范围的比例
# 正dy = 向上，负dy = 向下；dx正 = 向右
annot_offset = {
    'XMBRT':   {'MAE': (0.6,  0.25), 'RMSE': (0.6,  0.25), 'MAPE': (0.6, -0.35)},
    'HZMetro': {'MAE': (0.6,  0.20), 'RMSE': (0.6,  0.20), 'MAPE': (0.6,  0.20)},
    'BJMetro': {'MAE': (0.6, -0.35), 'RMSE': (0.6, -0.35), 'MAPE': (0.6, -0.35)},
}

fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)


for row, ds in enumerate(datasets):
    for col, (metric, mlabel) in enumerate(zip(metrics, m_labels)):
        ax = axes[row][col]
        vals  = step_data[ds][metric]
        color = DS_COLORS[ds]
        bs    = best_step[ds][metric]

        ax.plot(steps, vals, 'o-', color=color, linewidth=2,
                markersize=4, markerfacecolor='white', markeredgewidth=1.5)

        # 最优点实心圆
        ax.plot(bs, vals[bs-1], 'o', color=color, markersize=8, zorder=5)

        # 标注偏移
        data_range = max(vals) - min(vals)
        dx, dy_r   = annot_offset[ds][metric]
        dy         = dy_r * data_range

        ax.annotate(
            f'Step {bs}\n{vals[bs-1]:.2f}',
            xy=(bs, vals[bs-1]),
            xytext=(bs + dx, vals[bs-1] + dy),
            fontsize=7.5, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=1, shrinkA=0, shrinkB=3),
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color,
                      lw=0.8, alpha=0.92),  # 白色背景框，彻底解决遮挡
            zorder=10,
        )

        # 平均值虚线
        avg = np.mean(vals)
        ax.axhline(avg, color='grey', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(12.15, avg, f'avg\n{avg:.2f}', fontsize=6.5,
                color='grey', va='center')

        ax.set_xlim(0.5, 13.8)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps], fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

        if row == 0:
            ax.set_title(metric, fontsize=11, fontweight='bold', pad=6)
        if row == 2:
            ax.set_xlabel('Prediction Step', fontsize=9)

    axes[row][0].set_ylabel(f'{ds}\n{m_labels[0]}', fontsize=9, fontweight='bold')

handles = [mpatches.Patch(color=DS_COLORS[ds], label=ds) for ds in datasets]
fig.legend(handles=handles, loc='upper center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, 1.03), frameon=False)

plt.tight_layout()
plt.savefig('fig4_3_stepwise.png', dpi=300, bbox_inches='tight')
print("Saved fig4_3_stepwise.png")
plt.show()