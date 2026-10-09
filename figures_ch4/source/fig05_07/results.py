"""Recompute new predictions; preserve reported values of all untouched runs."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'04_核验与说明/ASTGNN补跑_20260930'
DATASETS=['XMBRT','HZMetro','BJMetro']
MODELS=['LSTM','ASTGCN-r','GraphWaveNet','STSGCN-L','StemGNN','SyncG4','ASTGNN','CWT-TSI']
METRICS=['mae','rmse','mape']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(target,prediction):
    t=np.asarray(target,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64)
    if t.shape!=p.shape or not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError('Shape mismatch or nonfinite prediction/target')
    error=p-t; nz=np.abs(t)>=1e-4
    if not nz.any():raise ValueError('MAPE undefined: no nonzero target')
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.square(error).mean())),
                mape=float(np.mean(np.abs(error[nz]/t[nz]))*100))

def improvement(best,ours):
    return (best-ours)/best*100

def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    status=json.loads((OUT/'status_v2.json').read_text(encoding='utf-8'))
    assert status.get('completed_at') and all(r['exit_code']==0 for r in status['runs'])
    assert len(status['runs'])==3
    protocol=json.loads((OUT/'protocol_v2.json').read_text(encoding='utf-8'))
    for filename,digest in protocol['source_hashes'].items():
        assert sha(ROOT/'03_可复现实验'/filename)==digest, f'Source changed during run: {filename}'
    source=ROOT/'04_核验与说明/审核修订交付_20260929/data/comparison_seven_baselines_display_names.csv'
    with source.open(encoding='utf-8-sig',newline='') as f: old=list(csv.DictReader(f))
    rows=[dict(r) for r in old if r['model']!='GT-Local']
    audit={};steps=[];new_metrics=[]
    for ds,n in zip(DATASETS,[44,80,276]):
        run=next(r['run'] for r in status['runs'] if r['dataset']==ds)
        base=ROOT/'03_可复现实验/save'/run
        record=json.loads((base/'metrics.json').read_text(encoding='utf-8'))
        y=np.load(base/'test_target.npy');p=np.load(base/'test_predict.npy')
        expected=np.load(ROOT/f'03_可复现实验/save/comparison_{ds.lower()}_astgnn/test_target.npy')
        assert np.array_equal(y,expected)
        ours=np.load(ROOT/f'03_可复现实验/save/{ds.lower()}_full/test_target.npy').reshape(-1,n,12)
        assert np.array_equal(y.reshape(-1,n,12),ours[-len(y)//n:])
        computed=metrics(y,p)
        # Float32 archived sklearn averaging and float64 global averaging differ slightly.
        for key in METRICS:
            assert abs(computed[key]-record[key]) < max(.01,abs(computed[key])*2e-4)
        assert record['model']=='ASTGNN-official' and record['seed']==1
        assert record['amp']==False and record['num_layers']==4
        assert record['best_epoch']==int(np.argmin(record['val_loss_history']))+1
        assert (base/f"{record['best_epoch']}.pt").exists()
        assert len(record['val_loss_history'])<=100
        row=dict(dataset=ds,model='ASTGNN',**{k:f'{computed[k]:.4f}' for k in METRICS},
                 best_epoch=str(record['best_epoch']),run=run,source=f'save/{run}/test_predict.npy')
        rows.append(row)
        new_metrics.append(dict(dataset=ds,**computed,best_epoch=record['best_epoch'],
                                trained_epochs=len(record['val_loss_history'])))
        for step in range(12):
            steps.append(dict(dataset=ds,model='ASTGNN',step=step+1,**metrics(y[:,step],p[:,step])))
        audit[ds]=dict(target_shape=list(y.shape),all_finite=True,
                      exact_legacy_baseline_targets=True,exact_common_cwt_targets=True,
                      recomputed_float64=computed,training_reported_metrics={k:record[k] for k in METRICS},
                      best_epoch=record['best_epoch'],epochs=len(record['val_loss_history']),
                      hashes={f:sha(base/f) for f in ['test_target.npy','test_predict.npy','metrics.json',f"{record['best_epoch']}.pt"]})
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),MODELS.index(r['model'])))
    assert len(rows)==24 and len(new_metrics)==3
    for r in rows:
        if r['model']!='ASTGNN':
            original=next(x for x in old if x['model']==r['model'] and x['dataset']==r['dataset'])
            assert r==original
    rankings=[];improvements=[];by_ds={};wins=0;rank_total=0
    for ds in DATASETS:
        sub={r['model']:r for r in rows if r['dataset']==ds}; by_ds[ds]={}
        for key in METRICS:
            values={m:float(sub[m][key]) for m in MODELS}
            best=min(MODELS[:-1],key=lambda m:values[m]); ours=values['CWT-TSI']
            rank=1+sum(v<ours for v in values.values());wins+=rank==1;rank_total+=rank
            item=dict(dataset=ds,metric=key,best_comparator=best,best_error=values[best],
                      cwt_error=ours,improve_pct=improvement(values[best],ours),cwt_rank=rank)
            improvements.append(item);by_ds[ds][key]=item
            for model in MODELS:
                rankings.append(dict(dataset=ds,metric=key,model=model,rank=1+sum(v<values[model] for v in values.values())))
    data=OUT/'data'; data.mkdir(exist_ok=True)
    write_csv(data/'comparison_updated.csv',rows)
    write_csv(data/'astgnn_metrics_float64.csv',new_metrics)
    write_csv(data/'astgnn_per_step_float64.csv',steps)
    write_csv(data/'rankings.csv',rankings)
    write_csv(data/'improvement.csv',improvements)
    summary=dict(cwt_wins=int(wins),cwt_mean_rank=rank_total/9,by_dataset=by_ds,
                 astgnn=new_metrics,retained_numeric_rows_unchanged=True,
                 rounding='new ASTGNN recomputed in float64; displayed to 4 decimals; other rows preserved; Improve uses displayed values')
    (data/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'result_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
