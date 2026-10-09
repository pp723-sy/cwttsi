"""Figures 5--7 only. Other manuscript figures contain no changed baseline.
Uses audited comparison_updated.csv; no smoothing, error bars or data suppression.
"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from results import ROOT,OUT,DATASETS,MODELS,METRICS

def main():
    data=OUT/'data'; out=OUT/'figures';out.mkdir(exist_ok=True)
    with (data/'comparison_updated.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    values={(r['dataset'],r['model'],k):float(r[k]) for r in rows for k in METRICS}
    summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
    metadata=[]
    def save(fig,name):
        for ext in ['png','svg','pdf']:
            path=out/(name+'.'+ext);fig.savefig(path,dpi=600,facecolor='white')
        metadata.append(dict(name=name,size_inches=list(fig.get_size_inches()),dpi=600))
        plt.close(fig)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42}):
        colors={'ours':'#C8B1B2','best':'#93AAA0','others':'#BCCAD5'}
        fig,axes=plt.subplots(3,3,figsize=(10.2,6.5),layout='constrained')
        for r,ds in enumerate(DATASETS):
            for c,k in enumerate(METRICS):
                ax=axes[r,c];v=np.array([values[ds,m,k] for m in MODELS]);best=int(np.argmin(v[:-1]));lowest=int(np.argmin(v))
                fills=[colors['ours'] if i==7 else colors['best'] if i==best else colors['others'] for i in range(8)]
                bars=ax.bar(range(8),v,color=fills,width=.72,edgecolor='#666666',linewidth=.45)
                bars[lowest].set_linewidth(1.2);bars[lowest].set_edgecolor('#333333')
                assert np.array_equal([b.get_height() for b in bars],v)
                for i in [best,7]:ax.text(i,v[i]+v.max()*.025,f'{v[i]:.2f}',ha='center',va='bottom',fontsize=8.5,fontweight='bold' if i==lowest else 'normal')
                ax.set_ylim(0,v.max()*1.17);ax.set_xticks(range(8),MODELS,rotation=48,ha='right',fontsize=8)
                ax.tick_params(axis='y',labelsize=8)
                if r==0:ax.set_title(['MAE','RMSE','MAPE (%)'][c],fontsize=11,fontweight='bold')
                if c==0:ax.set_ylabel(ds,fontsize=10,fontweight='bold')
                ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E4E4E4',linewidth=.4);ax.set_axisbelow(True)
        fig.legend(handles=[Patch(facecolor=colors[k],edgecolor='#666666',label=lab) for k,lab in [('ours','CWT-TSI'),('best','Best comparator'),('others','Other comparators')]],loc='outside lower center',ncol=3,frameon=False,fontsize=10)
        fig.suptitle('Prediction errors on three datasets',fontsize=13,fontweight='bold')
        save(fig,'Fig05_整体预测误差_ASTGNN更新')
        ranks=np.empty((8,9))
        for j,(ds,k) in enumerate((d,k) for d in DATASETS for k in METRICS):
            v=[values[ds,m,k] for m in MODELS]
            for i in range(8):ranks[i,j]=1+sum(x<v[i] for x in v)
        fig,ax=plt.subplots(figsize=(10.2,4.64),layout='constrained')
        cmap=LinearSegmentedColormap.from_list('soft_rank',['#F1F5F8','#9EB3C5'])
        img=ax.imshow(ranks,cmap=cmap,vmin=1,vmax=8,aspect='auto')
        ax.set_yticks(range(8),MODELS)
        ax.set_xticks(range(9),[d+'\n'+['MAE','RMSE','MAPE'][METRICS.index(k)] for d in DATASETS for k in METRICS],fontsize=9)
        for i in range(8):
            for j in range(9):ax.text(j,i,str(int(ranks[i,j])),ha='center',va='center',fontweight='bold' if ranks[i,j]==1 else 'normal',color='#26333C')
        for x in [2.5,5.5]:ax.axvline(x,color='white',linewidth=2)
        ax.add_patch(Rectangle((-.5,6.5),9,1,fill=False,edgecolor='#A5767C',linewidth=1.8))
        ax.set_title('Metric ranks across datasets',fontweight='bold',pad=12)
        fig.colorbar(img,ax=ax,ticks=range(1,9),label='Rank (1 = lowest error)',fraction=.04,pad=.02)
        save(fig,'Fig06_模型指标排名_ASTGNN更新')
        fig,ax=plt.subplots(figsize=(10.2,3.9),layout='constrained')
        palette=['#547C9A','#71937C','#AA7880'];markers=['o','s','^']
        for i,k in enumerate(METRICS):
            v=[summary['by_dataset'][ds][k]['improve_pct'] for ds in DATASETS]
            ax.plot(range(3),v,color=palette[i],marker=markers[i],linewidth=1.8,markersize=6,label=k.upper())
        ax.axhline(0,color='#555555',linestyle='--',linewidth=.8)
        ax.set_xticks(range(3),DATASETS);ax.set_xlim(-.15,2.15)
        ax.set_ylabel('Error reduction relative to best comparator (%)')
        ax.set_title('CWT-TSI relative error reduction',fontweight='bold')
        ax.grid(axis='y',color='#E5E5E5',linewidth=.5);ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='best',ncol=3,frameon=False)
        save(fig,'Fig07_相对误差降幅_ASTGNN更新')
    (OUT/'figure_manifest.json').write_text(json.dumps(dict(figures=metadata,
        source='data/comparison_updated.csv',derived=['data/summary.json','data/rankings.csv','data/improvement.csv'],
        rules='Figure 5 zero-based bars; Figure 6 per-column rank; Figure 7 reference chosen separately for each dataset/metric; negative improvements retained; no uncertainty intervals (one run)',
        unchanged_figures='Figures 1-4 and 8-11 preserved from input DOCX'),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
