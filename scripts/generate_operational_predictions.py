import argparse, hashlib, json, sqlite3, subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
WEEKLY = ROOT/'data/processed/weekly_timeseries_features.csv'
CLS_MODEL = ROOT/'models/classification/corrected/selected_classifier_deployment.joblib'
CLS_META = ROOT/'models/classification/corrected/selected_classifier_deployment_metadata.json'
FC_MODEL = ROOT/'models/forecasting/selected_forecaster_pretest.joblib'
REC_MODEL = ROOT/'models/recommendation/selected_recommender_pretest.joblib'
REC_TRAIN = ROOT/'data/modeling/recommendation/recommendation_train.csv'
REC_VALID = ROOT/'data/modeling/recommendation/recommendation_validation.csv'
DEFAULT_DB = ROOT/'database/generated/github_herd_phase3_predictions.db'
DEFAULT_OUTPUT = ROOT/'outputs/phase3/operational_predictions'
MIN_CLEAN_WEEKS = 4


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--database-path', type=Path, default=DEFAULT_DB)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--run-id')
    p.add_argument('--batch-size', type=int, default=5000)
    p.add_argument('--validate-only', action='store_true')
    return p.parse_args()


def now(): return datetime.now(timezone.utc).isoformat()
def git(*a): return subprocess.run(['git',*a],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
def loadj(p): return json.loads(p.read_text(encoding='utf-8'))
def writej(p,v): p.write_text(json.dumps(v,indent=2,default=str)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def hashes():
    ps=[WEEKLY,CLS_MODEL,CLS_META,FC_MODEL,REC_MODEL,REC_TRAIN,REC_VALID]
    miss=[str(p.relative_to(ROOT)) for p in ps if not p.is_file()]
    if miss: raise FileNotFoundError('\n'.join(miss))
    return {str(p.relative_to(ROOT)):sha(p) for p in ps}


def numeric(df, cols, label):
    miss=sorted(set(cols)-set(df.columns))
    if miss: raise ValueError(f'{label} missing {miss}')
    x=df[cols].apply(pd.to_numeric,errors='raise')
    if x.isna().any().any() or not np.isfinite(x.to_numpy(float)).all():
        raise ValueError(f'{label} has missing/nonfinite values')
    return x


def latest_rows():
    d=pd.read_csv(WEEKLY)
    d['week_start']=pd.to_datetime(d['week_start'],utc=True,errors='raise')
    d=d.sort_values(['repo_id','week_start']).reset_index(drop=True)
    d['repository_row_index']=d.groupby('repo_id').cumcount()
    d['history_length']=d['repository_row_index']+1
    d['initial_snapshot_rows']=1
    d['clean_history_weeks']=d['history_length']-1
    latest=d.groupby('repo_id',as_index=False).tail(1).sort_values('repo_id').reset_index(drop=True)
    ok=latest['clean_history_weeks']>=MIN_CLEAN_WEEKS
    sel=latest.loc[ok].copy().reset_index(drop=True)
    return sel, {'weekly_rows':len(d),'weekly_repositories':d.repo_id.nunique(),'latest_repositories':len(latest),'eligible_repositories':len(sel),'ineligible_repositories':int((~ok).sum()),'minimum_clean_history_weeks':MIN_CLEAN_WEEKS}


def repo_predictions(latest, run_id, created):
    clf=joblib.load(CLS_MODEL); meta=loadj(CLS_META)
    cf=list(clf.feature_names_in_)
    if cf!=list(meta['feature_columns']): raise RuntimeError('classification feature order mismatch')
    prob=clf.predict_proba(numeric(latest,cf,'classification'))[:,1]
    thr=float(meta['selected_threshold']); pred=(prob>=thr).astype(int)
    art=joblib.load(FC_MODEL)
    if art['artifact_version']!='forecasting_pretest_v1': raise RuntimeError('bad forecasting artifact')
    xf=numeric(latest,list(art['feature_columns']),'forecasting')
    base=float(art['baseline_multiplier'])*latest[art['baseline_column']].to_numpy(float)
    resid=art['residual_estimator'].predict(xf)
    fc=np.maximum(float(art['prediction_minimum']),base+resid)
    out=latest[['repo_id','repo_full_name','week_start','history_length','initial_snapshot_rows','clean_history_weeks']].copy()
    out.insert(0,'run_id',run_id); out=out.rename(columns={'week_start':'prediction_cutoff_week'})
    out['prediction_cutoff_week']=out['prediction_cutoff_week'].astype(str)
    out['growth_surge_probability']=prob; out['growth_surge_threshold']=thr; out['growth_surge_prediction']=pred
    out['raw_rolling_baseline']=base; out['residual_correction']=resid; out['predicted_future_4week_stars']=fc; out['generated_at_utc']=created
    return out, {'rows':len(out),'predicted_surges':int(pred.sum()),'classification_probability_mean':float(prob.mean()),'classification_threshold':thr,'forecast_mean':float(fc.mean()),'forecast_median':float(np.median(fc)),'forecast_minimum':float(fc.min()),'forecast_maximum':float(fc.max())}

def rec_inputs():
    a=joblib.load(REC_MODEL)
    if a['artifact_version']!='recommendation_pretest_v1' or a['freeze_status']!='recommendation_pretest_frozen': raise RuntimeError('bad recommendation artifact')
    c=a['scoring_contract']
    if float(c['forecast_weight'])!=0 or bool(c['static_content_in_primary']): raise RuntimeError('recommendation contract changed')
    inter=pd.concat([pd.read_csv(REC_TRAIN,usecols=['user_id','repo_id']),pd.read_csv(REC_VALID,usecols=['user_id','repo_id'])],ignore_index=True)
    if inter.duplicated(['user_id','repo_id']).any(): raise RuntimeError('duplicate fit interactions')
    users=np.asarray(a['fit_user_ids'],dtype=np.int64); repos=np.asarray(a['catalog_repo_ids'],dtype=np.int64)
    if not np.array_equal(users,np.sort(inter.user_id.astype(np.int64).unique())): raise RuntimeError('fit user mismatch')
    um={int(v):i for i,v in enumerate(users)}; rm={int(v):i for i,v in enumerate(repos)}
    r=inter.user_id.map(um); col=inter.repo_id.map(rm)
    if r.isna().any() or col.isna().any(): raise RuntimeError('unknown user/repo')
    m=sparse.csr_matrix((np.ones(len(inter)),(r.to_numpy(np.int64),col.to_numpy(np.int64))),shape=(len(users),len(repos)),dtype=float)
    counts=np.asarray(m.sum(axis=0)).ravel(); frozen=np.asarray(a['item_counts'],float)
    if float(np.max(np.abs(counts-frozen)))>0: raise RuntimeError('item count mismatch')
    catalog=pd.DataFrame({'repo_id':repos,'repo_full_name':np.asarray(a['catalog_repo_names'],object),'training_interaction_count':frozen})
    return a,catalog,m


def mask(scores,seen):
    x=np.asarray(scores,float).copy(); x[~np.isfinite(x)]=-np.inf; x[seen]=-np.inf; return x

def order(scores): return np.argsort(-scores,axis=1,kind='stable')
def ranknorm(scores,seen):
    o=order(mask(scores,seen)); ranks=np.empty_like(o,dtype=np.int32); rows=np.arange(len(o))[:,None]
    ranks[rows,o]=np.arange(o.shape[1],dtype=np.int32)[None,:]
    n=(~seen).sum(axis=1).astype(float)
    z=1-ranks.astype(float)/np.maximum(n-1,1)[:,None]; z[seen]=0; return z


def score_batch(a,m,start,end):
    h=m[start:end]; seen=h.toarray()>0; candidates=(~seen).sum(axis=1); k=int(a['scoring_contract']['top_k'])
    if np.any(candidates<k): raise RuntimeError('not enough candidates')
    item=np.asarray(h@np.asarray(a['component_artifacts']['item_similarity'],float),float)
    length=np.asarray(h.sum(axis=1)).ravel(); agg=a['internal_family_winners']['item_item_cosine']['parameters']['aggregation']
    if agg=='mean': item/=np.maximum(length,1)[:,None]
    elif agg!='sum': raise RuntimeError('unknown aggregation')
    inv=np.divide(1,length,out=np.zeros_like(length),where=length>0)
    graph=np.asarray((sparse.diags(inv)@h)@np.asarray(a['component_artifacts']['graph_propagation'],float),float)
    pop=np.broadcast_to(np.asarray(a['item_counts'],float)[None,:],item.shape).copy()
    ir,gr,pr=ranknorm(item,seen),ranknorm(graph,seen),ranknorm(pop,seen)
    w=a['hybrid_parameters']['weights']
    if float(w['truncated_svd'])!=0 or float(w['training_popularity'])!=0: raise RuntimeError('unexpected hybrid weight')
    hybrid=float(w['item_item_cosine'])*ir+float(w['graph_personalized_pagerank'])*gr
    final=ranknorm(hybrid,seen)-float(a['herd_parameters']['popularity_penalty'])*pr
    return order(mask(final,seen))[:,:k],final,candidates


def schema(conn):
    conn.executescript('''
    PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS model_runs(run_id TEXT PRIMARY KEY,started_at_utc TEXT NOT NULL,completed_at_utc TEXT,git_commit TEXT NOT NULL,git_branch TEXT NOT NULL,status TEXT NOT NULL,repository_prediction_rows INTEGER,recommendation_rows INTEGER,input_hashes_json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS repository_predictions(run_id TEXT NOT NULL,repo_id INTEGER NOT NULL,repo_full_name TEXT NOT NULL,prediction_cutoff_week TEXT NOT NULL,history_length INTEGER NOT NULL,initial_snapshot_rows INTEGER NOT NULL,clean_history_weeks INTEGER NOT NULL,growth_surge_probability REAL NOT NULL,growth_surge_threshold REAL NOT NULL,growth_surge_prediction INTEGER NOT NULL,raw_rolling_baseline REAL NOT NULL,residual_correction REAL NOT NULL,predicted_future_4week_stars REAL NOT NULL,generated_at_utc TEXT NOT NULL,PRIMARY KEY(run_id,repo_id),FOREIGN KEY(run_id) REFERENCES model_runs(run_id) ON DELETE CASCADE);
    CREATE TABLE IF NOT EXISTS user_recommendations(run_id TEXT NOT NULL,user_id INTEGER NOT NULL,rank INTEGER NOT NULL,repo_id INTEGER NOT NULL,repo_full_name TEXT NOT NULL,score REAL NOT NULL,training_interaction_count REAL NOT NULL,generated_at_utc TEXT NOT NULL,PRIMARY KEY(run_id,user_id,rank),FOREIGN KEY(run_id) REFERENCES model_runs(run_id) ON DELETE CASCADE);
    CREATE INDEX IF NOT EXISTS idx_repo_prob ON repository_predictions(run_id,growth_surge_probability DESC);
    CREATE INDEX IF NOT EXISTS idx_repo_fc ON repository_predictions(run_id,predicted_future_4week_stars DESC);
    CREATE INDEX IF NOT EXISTS idx_rec_repo ON user_recommendations(run_id,repo_id);
    ''')


def validate(repo_summary,a,catalog,m,batch):
    end=min(batch,m.shape[0]); top,scores,n=score_batch(a,m,0,end)
    if not np.isfinite(scores).all(): raise RuntimeError('nonfinite scores')
    print('='*100); print('OPERATIONAL PREDICTION VALIDATION'); print('='*100)
    print('Repository prediction rows:',repo_summary['rows']); print('Predicted growth surges:',repo_summary['predicted_surges']); print('Mean four-week forecast:',repo_summary['forecast_mean'])
    print('Recommendation fitted users:',m.shape[0]); print('Recommendation catalog:',len(catalog)); print('Validated recommendation users:',end); print('Top-k:',top.shape[1])
    print('Candidate minimum / median / maximum:',int(n.min()),float(np.median(n)),int(n.max())); print('Expected full recommendation rows:',m.shape[0]*top.shape[1]); print('Validation status: passed_no_files_written')

def full_run(ns,run_id,started,hs,pred,latest_summary,repo_summary,a,catalog,m):
    db=ns.database_path.resolve(); outdir=ns.output_root.resolve()/run_id
    if outdir.exists(): raise RuntimeError(f'output exists: {outdir}')
    db.parent.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True)
    conn=sqlite3.connect(db); freq=np.zeros(len(catalog),dtype=np.int64); rec_rows=0; cmin=None; cmax=None; csum=0; cn=0
    try:
        schema(conn)
        if conn.execute('SELECT COUNT(*) FROM model_runs WHERE run_id=?',(run_id,)).fetchone()[0]: raise RuntimeError('run id exists')
        conn.execute('INSERT INTO model_runs(run_id,started_at_utc,git_commit,git_branch,status,input_hashes_json) VALUES(?,?,?,?,?,?)',(run_id,started,git('rev-parse','HEAD'),git('branch','--show-current'),'running',json.dumps(hs,sort_keys=True)))
        pred.to_sql('repository_predictions',conn,if_exists='append',index=False,method='multi')
        users=np.asarray(a['fit_user_ids'],dtype=np.int64); repo_ids=catalog.repo_id.to_numpy(np.int64); names=catalog.repo_full_name.astype(str).to_numpy(); counts=catalog.training_interaction_count.to_numpy(float)
        created=pred.generated_at_utc.iloc[0]; batch=int(ns.batch_size)
        if batch<=0: raise ValueError('batch must be positive')
        sql='INSERT INTO user_recommendations VALUES(?,?,?,?,?,?,?,?)'
        for b,start in enumerate(range(0,len(users),batch),1):
            end=min(start+batch,len(users)); top,final,n=score_batch(a,m,start,end); k=top.shape[1]; rows=np.arange(end-start)[:,None]; flat=top.ravel()
            records=list(zip([run_id]*((end-start)*k),np.repeat(users[start:end],k).tolist(),np.tile(np.arange(1,k+1),end-start).tolist(),repo_ids[flat].tolist(),names[flat].tolist(),final[rows,top].ravel().tolist(),counts[flat].tolist(),[created]*((end-start)*k)))
            conn.executemany(sql,records); conn.commit(); rec_rows+=len(records); freq+=np.bincount(flat,minlength=len(catalog))
            cmin=int(n.min()) if cmin is None else min(cmin,int(n.min())); cmax=int(n.max()) if cmax is None else max(cmax,int(n.max())); csum+=int(n.sum()); cn+=len(n)
            print(f'[Batch {b}] users {start:,}-{end-1:,}; rows {rec_rows:,}')
        expected=len(users)*int(a['scoring_contract']['top_k'])
        if rec_rows!=expected: raise RuntimeError('recommendation row mismatch')
        repo_rows=conn.execute('SELECT COUNT(*) FROM repository_predictions WHERE run_id=?',(run_id,)).fetchone()[0]
        db_rec_rows=conn.execute('SELECT COUNT(*) FROM user_recommendations WHERE run_id=?',(run_id,)).fetchone()[0]
        rec_users=conn.execute('SELECT COUNT(DISTINCT user_id) FROM user_recommendations WHERE run_id=?',(run_id,)).fetchone()[0]
        dups=conn.execute('SELECT COUNT(*) FROM (SELECT user_id,rank,COUNT(*) n FROM user_recommendations WHERE run_id=? GROUP BY user_id,rank HAVING n>1)',(run_id,)).fetchone()[0]
        completed=now(); conn.execute("UPDATE model_runs SET completed_at_utc=?,status='complete',repository_prediction_rows=?,recommendation_rows=? WHERE run_id=?",(completed,repo_rows,db_rec_rows,run_id)); conn.commit()
    finally: conn.close()
    f=catalog.copy(); f['recommendation_count']=freq; f=f.sort_values(['recommendation_count','repo_id'],ascending=[False,True]).reset_index(drop=True)
    pred.to_csv(outdir/'repository_predictions.csv',index=False); f.to_csv(outdir/'recommendation_repository_frequency.csv',index=False)
    with sqlite3.connect(db) as conn:
        sample=pd.read_sql_query('SELECT * FROM user_recommendations WHERE run_id=? ORDER BY user_id,rank LIMIT 1000',conn,params=(run_id,))
    sample.to_csv(outdir/'recommendation_sample_first_100_users.csv',index=False)
    summary={'status':'operational_predictions_complete','run_id':run_id,'started_at_utc':started,'completed_at_utc':completed,'git_commit':git('rev-parse','HEAD'),'git_branch':git('branch','--show-current'),'database_path':str(db.relative_to(ROOT)),'tracked_phase2_database_modified':False,'input_hashes':hs,'latest_repository_diagnostics':latest_summary,'repository_predictions':repo_summary,'recommendations':{'artifact_version':a['artifact_version'],'selected_model':a['selected_model_name'],'fit_users':len(a['fit_user_ids']),'catalog_repositories':len(catalog),'top_k':int(a['scoring_contract']['top_k']),'rows_written':db_rec_rows,'candidate_count_minimum':cmin,'candidate_count_mean':csum/cn,'candidate_count_maximum':cmax,'top_repositories':f.head(20).to_dict(orient='records')},'database_verification':{'repository_prediction_rows':repo_rows,'recommendation_rows':db_rec_rows,'recommendation_users':rec_users,'duplicate_user_rank_groups':dups}}
    writej(outdir/'operational_run_summary.json',summary)
    oh={p.name:sha(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    writej(outdir/'operational_run_manifest.json',{'run_id':run_id,'status':'operational_predictions_frozen','created_at_utc':completed,'git_commit':summary['git_commit'],'database_path':summary['database_path'],'database_sha256':sha(db),'input_hashes':hs,'output_hashes':oh,'repository_prediction_rows':repo_rows,'recommendation_rows':db_rec_rows,'recommendation_users':rec_users})
    print('='*100); print('OPERATIONAL PREDICTIONS COMPLETE'); print('='*100); print('Run ID:',run_id); print('Repository prediction rows:',repo_rows); print('Recommendation users:',rec_users); print('Recommendation rows:',db_rec_rows); print('Database:',db); print('Output directory:',outdir)


def main():
    ns=args()
    if ns.database_path.resolve()==(ROOT/'database/github_herd.db').resolve(): raise RuntimeError('refusing to modify tracked Phase 2 database')
    hs=hashes(); started=now(); run_id=ns.run_id or ('operational_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+git('rev-parse','--short','HEAD'))
    latest,latest_summary=latest_rows(); pred,repo_summary=repo_predictions(latest,run_id,now()); a,catalog,m=rec_inputs()
    if ns.validate_only: validate(repo_summary,a,catalog,m,int(ns.batch_size))
    else: full_run(ns,run_id,started,hs,pred,latest_summary,repo_summary,a,catalog,m)


if __name__=='__main__': main()
