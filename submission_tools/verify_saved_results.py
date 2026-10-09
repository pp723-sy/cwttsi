from pathlib import Path
import json,csv,hashlib,re,sys,argparse
import numpy as np
E=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Verify archived targets and table values, without training or network access.')
parser.add_argument('--output',type=Path,default=E/'verification_output')
parser.add_argument('--archive',type=Path,default=E,help='Authorized local archive containing datasets/, save/, ts2img/ and manuscript_data/.')
args=parser.parse_args(); OUT=args.output.resolve()
E=args.archive.resolve()
if not all((E/name).is_dir() for name in ['datasets','save','ts2img','manuscript_data']):
    raise SystemExit('Full verification requires the authorized local archive. Supply --archive PATH; restricted data and checkpoints are not distributed in this code package.')
if OUT.exists(): raise FileExistsError('Choose a new output directory to protect existing results.')
OUT.mkdir(parents=True)
ROOT=E
def dump(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def metrics(y,p,dtype):
    y=y.astype(dtype);p=p.astype(dtype);e=p-y;flat=e.reshape(-1,12)
    # Match sklearn's multi-output columnwise FP32 reductions for original runs.
    mae=float(np.mean(np.mean(np.abs(flat),axis=0)))
    rmse=float(np.sqrt(np.mean(np.mean(flat**2,axis=0))))
    mask=np.abs(y)>=1e-4
    mape=float(np.mean(np.abs((p[mask]-y[mask])/y[mask]))*100)
    return dict(mae=mae,rmse=rmse,mape=mape)
comp=readcsv(E/'manuscript_data/comparison_updated.csv'); abl=readcsv(E/'manuscript_data/ablation_table_common_test.csv')
runs={r['run']:r for r in comp+abl}; records=[]; steps=[]; checks=[]; overlaps=[]; dataset_records=[]
for ds in ['XMBRT','HZMetro','BJMetro']:
    base=E/'datasets'/ds;raw=np.loadtxt(E/'ts2img/origin_data'/f'{ds}_{dict(XMBRT="4320x44",HZMetro="5616x80",BJMetro="2700x276")[ds]}.csv',delimiter=',',dtype=np.float32)
    y=np.load(base/'y/y_hs12_ts12.npy',mmap_mode='r'); ts=np.load(base/'ts/ts_hs12_ts12.npy',mmap_mode='r');te=np.load(base/'te/te_hs12_ts12.npy',mmap_mode='r');xfile=next((base/'x').glob('*cwt*.npy')); x=np.load(xfile,mmap_mode='r'); split=np.load(base/'splits/split_6_2_2_hs12_ts12.npz')
    expected=np.lib.stride_tricks.sliding_window_view(raw,24,axis=0)
    checks += [{'check':ds+' history matches raw','pass':bool(np.array_equal(ts,expected[:,:,:12]))},{'check':ds+' targets match raw','pass':bool(np.array_equal(y,expected[:,:,12:]))},{'check':ds+' synthetic sequence embeddings','pass':bool(np.array_equal(te[:,0,0],(np.arange(len(te))//(108 if ds=='BJMetro' else 216))%7))}]
    dataset_records.append({'dataset':ds,'raw_shape':list(raw.shape),'windows':len(ts),'images':list(x.shape),'dtype':str(x.dtype),'image_MiB':x.nbytes/2**20,'raw_min':float(raw.min()),'raw_max':float(raw.max()),'finite':bool(np.isfinite(raw).all()),'noninteger_fraction':float(np.mean(raw!=np.round(raw))),'common_test_start_window':int(split['test_idx'][0]),'train_std_ddof1':float(np.std(np.asarray(ts[split['train_idx']]),ddof=1))})
    full_train=np.arange(int(len(ts)*.6));full_val=np.arange(int(len(ts)*.6),int(len(ts)*.6)+int(len(ts)*.2))
    def target_times(ids):return np.unique((ids[:,None]+np.arange(12,24)).reshape(-1))
    for protocol,tr,va in [('baseline',split['train_idx'],split['val_idx']),('full_and_ablation',full_train,full_val)]:
        for label,a,b in [('train_validation',tr,va),('validation_common_test',va,split['test_idx'])]:
            ov=np.intersect1d(target_times(a),target_times(b))
            overlaps.append({'dataset':ds,'pipeline':protocol,'boundary':label,'shared_target_timesteps':len(ov),'first':int(ov[0]) if len(ov) else None,'last':int(ov[-1]) if len(ov) else None})
for run,r in runs.items():
    d=E/'save'/run;m=json.loads((d/'metrics.json').read_text(encoding='utf-8'));p=np.load(d/'test_predict.npy');y=np.load(d/'test_target.npy');shape=list(y.shape)
    # Original train.py serializes (sample*node,horizon), baseline pipeline serializes (sample,node,horizon).
    p=p.reshape(-1,m['num_nodes'],12);y=y.reshape(p.shape)
    ds=r['dataset'];split=np.load(E/'datasets'/ds/'splits/split_6_2_2_hs12_ts12.npz');start=int(split['test_idx'][0]);yt=np.load(E/'datasets'/ds/'y/y_hs12_ts12.npy',mmap_mode='r')[start:]
    trimmed=len(y)-len(yt);p=p[trimmed:];y=y[trimmed:]
    checks.append({'check':run+' targets equal common test','pass':bool(np.array_equal(y,yt))})
    f32=metrics(y,p,np.float32);f64=metrics(y,p,np.float64)
    is_recomputed=trimmed>0 or 'official' in run
    for key in ['mae','rmse','mape']:
        ref=f64[key] if is_recomputed else float(m[key]);shown=float(r[key]);checks.append({'check':run+' '+key+' table matches declared evaluation','pass':round(ref,4)==round(shown,4)})
    ck=d/f'{m["best_epoch"]}.pt';h=np.load(d/'history.npz'); best=int(np.argmin(h['val_loss']))+1
    checks.append({'check':run+' best checkpoint exists','pass':ck.exists()})
    records.append({'dataset':ds,'run':run,'display_model':r['model'],'input_prediction_shape':shape,'test_samples':len(y),'removed_first_windows':trimmed,'table_metrics':{k:float(r[k]) for k in ['mae','rmse','mape']},'recalculated_fp32':f32,'recalculated_fp64':f64,'best_epoch_recorded':m['best_epoch'],'argmin_val_loss_epoch':best,'epochs_completed':len(h['val_loss']),'checkpoint':str(ck.relative_to(E)),'checkpoint_bytes':ck.stat().st_size if ck.exists() else 0,'environment':m.get('environment',{}),'seed':m.get('seed','see original log'),'amp':m.get('amp','see original log')})
    if r['model']=='CWT-TSI':
        for i in range(12):
            yy=y[:,:,i].astype(np.float64);pp=p[:,:,i].astype(np.float64);mask=np.abs(yy)>=1e-4
            steps.append(dict(dataset=ds,step=i+1,mae=float(np.mean(np.abs(pp-yy))),rmse=float(np.sqrt(np.mean((pp-yy)**2))),mape=float(np.mean(np.abs((pp[mask]-yy[mask])/yy[mask]))*100)))
dump('numeric_checks.json',{'checks':checks,'passed':sum(c['pass'] for c in checks),'failed':[c for c in checks if not c['pass']],'datasets':dataset_records,'runs':records,'target_overlap':overlaps})
with (OUT/'per_horizon_fp64.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(steps[0]));w.writeheader();w.writerows(steps)
print(json.dumps({'checks':len(checks),'failed':[c for c in checks if not c['pass']],'unique_runs':len(runs),'overlap':overlaps},ensure_ascii=False))

if any(not c['pass'] for c in checks): raise SystemExit('Archived numeric checks failed')
