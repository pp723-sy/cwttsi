"""Rename ablation variants in data-derived publication figures; keep all values."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'04_核验与说明/批注核定_20261001'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
SOURCE=ROOT/'04_核验与说明/图表交付_20260929/data/ablation_table_common_test.csv'
MAPPING={'No-Img':'w/o Image','No-TS':'w/o TS','No-Fuse':'w/o Learnable Fusion','CWT-TSI':'CWT-TSI'}
DATASETS=['XMBRT','HZMetro','BJMetro'];METRICS=['mae','rmse','mape'];MODELS=list(MAPPING)

def main():
    with SOURCE.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    values={(r['dataset'],r['model'],k):float(r[k]) for r in rows for k in METRICS}
    records=[]
    def save(fig,name):
        for ext in ['png','svg','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=600,facecolor='white')
        records.append(dict(name=name,size_inches=list(fig.get_size_inches()),png_dpi=600))
        plt.close(fig)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42}):
        fig,axes=plt.subplots(3,3,figsize=(11.8,7.0),layout='constrained')
        palette=['#839BA9','#9BAD9E','#B7A98D','#B8878C'];markers=['o','s','^','D']
        labels=['w/o Image','w/o TS','w/o Learnable\nFusion','CWT-TSI']
        for r,ds in enumerate(DATASETS):
            for c,k in enumerate(METRICS):
                ax=axes[r,c];v=np.array([values[ds,m,k] for m in MODELS]);lo,hi=v.min(),v.max();span=max(hi-lo,hi*.015)
                ax.set_xlim(max(0,lo-span*.2),hi+span*.2);ax.set_ylim(3.6,-.7)
                for i,m in enumerate(MODELS):
                    best=v[i]==v.min()
                    ax.scatter(v[i],i,s=45,color=palette[i],marker=markers[i],edgecolor='#333333',linewidth=1.5 if best else .6,zorder=3)
                    ax.text(1.015,i,f'{v[i]:.2f}',transform=ax.get_yaxis_transform(),va='center',fontsize=9,fontweight='bold' if best else 'normal')
                ax.set_yticks(range(4),labels,fontsize=9)
                ax.set_title(ds+'  '+['MAE','RMSE','MAPE (%)'][c],fontsize=11,fontweight='bold')
                ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(axis='x',labelsize=8)
                ax.tick_params(axis='y',length=0);ax.spines[['top','right','left']].set_visible(False)
                ax.grid(axis='x',color='#E4E4E4',linewidth=.5);ax.set_axisbelow(True)
        fig.suptitle('Ablation prediction errors',fontsize=14,fontweight='bold')
        fig.supxlabel('Lower is better; bold values mark the minimum. Panels use independent ranges.',fontsize=10)
        save(fig,'Fig09_消融误差_w_o命名')
        change=np.array([[(values[ds,m,k]-values[ds,'CWT-TSI',k])/values[ds,'CWT-TSI',k]*100 for ds in DATASETS for k in METRICS] for m in MODELS[:-1]])
        fig,ax=plt.subplots(figsize=(11.8,3.1),layout='constrained')
        cmap=LinearSegmentedColormap.from_list('soft_change',['#9CB5C9','#F7F7F7','#CAA0A4'])
        bound=50;im=ax.imshow(change,cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
        ax.set_yticks(range(3),labels[:3],fontsize=10)
        ax.set_xticks(range(9),[ds+'\n'+['MAE','RMSE','MAPE (%)'][METRICS.index(k)] for ds in DATASETS for k in METRICS],fontsize=9)
        for i in range(3):
            for j in range(9):ax.text(j,i,f'{change[i,j]:.1f}',ha='center',va='center',fontsize=10,color='#26333B')
        for x in [2.5,5.5]:ax.axvline(x,color='white',linewidth=2)
        ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title('Relative error change of ablation variants vs. CWT-TSI',fontsize=13,fontweight='bold',pad=10)
        fig.colorbar(im,ax=ax,label='Relative error change (%)',ticks=[-50,-25,0,25,50],fraction=.035,pad=.025)
        save(fig,'Fig10_消融相对误差_w_o命名')
    with (OUT/'消融显示数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','model','mae','rmse','mape']);writer.writeheader()
        for r in rows:writer.writerow(dict(dataset=r['dataset'],model=MAPPING[r['model']],**{k:r[k] for k in METRICS}))
    (OUT/'图源生成记录.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        names=MAPPING,figures=records,relative_change_formula='100 * (variant - full) / full',
        all_source_values_preserved=True,figure10_color_scale=[-50,0,50],uncertainty='none; one run per configuration'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created Fig9 and Fig10, PNG/SVG/PDF; all 36 ablation values retained.')
if __name__=='__main__':main()
