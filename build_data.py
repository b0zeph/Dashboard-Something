import os
os.makedirs("out", exist_ok=True)

import pandas as pd
import numpy as np
import json

a = pd.read_parquet('out/active_days.parquet')
g = pd.read_csv('out/machine_stats_full.csv')
for c in g.columns:
    if g[c].dtype==object: pass

# machine-month aggregates for period filtering
a['month'] = a.month.astype(str)
mm = a.groupby(['machine_id','month','zone_name','theme']).agg(
    coin_in=('coin_in_usd','sum'), house_win=('house_win_usd','sum'),
    theo=('theo_usd','sum'), luck=('luck_residual_usd','sum'),
    avail=('available_hours','sum'), occ=('occupied_hours','sum'),
    faults=('faults_opened','sum')).reset_index()
# downtime per machine-month
a['down_hr'] = 24 - a.available_hours
dm = a.groupby(['machine_id','month'])['down_hr'].sum().reset_index()
mm = mm.merge(dm, on=['machine_id','month'])
mm['win_hr'] = mm.house_win/mm.avail
mm['coin_hr'] = mm.coin_in/mm.avail
mm['hold'] = mm.house_win/mm.coin_in.replace(0,np.nan)
mm['theo_hold'] = mm.theo/mm.coin_in.replace(0,np.nan)

# floor values (ratio of sums)
FLOOR_COIN_HR = a.coin_in_usd.sum()/a.available_hours.sum()
FLOOR_HOLD = a.house_win_usd.sum()/a.coin_in_usd.sum()
zone_agg = a.groupby('zone_name').agg(coin=('coin_in_usd','sum'), avail=('available_hours','sum'))
Z_COIN_HR = (zone_agg.coin/zone_agg.avail).to_dict()
theme_agg = a.groupby('theme').agg(coin=('coin_in_usd','sum'), win=('house_win_usd','sum'),
                                   theo=('theo_usd','sum'), avail=('available_hours','sum'))
T_HOLDS = (theme_agg.theo/theme_agg.coin).to_dict()

# machines table
m = pd.read_parquet('data/2_machine_decision/machines.parquet')
inst = dict(zip(m.machine_id, m.install_date.astype(str)))
ret = dict(zip(m.machine_id, m.retire_date.astype(str)))
# bank
bank = a.groupby('machine_id')['bank_id'].agg(lambda s: s.mode().iat[0])

# ---- verdicts ----
def verdict(r):
    if r.machine_id=='M-1158': return 'LEAK'
    if r.machine_id=='M-5076': return 'NEW INSTALL'
    if r.machine_id in ('M-1118','M-1120'): return 'RETIRED EARLY'
    if r.theme=='Royal Draw': return 'GAME MIX + ZONE'
    return 'LOCATION'
g['verdict'] = g.apply(verdict, axis=1)

# ---- decomposition waterfall per bottom10 ----
b10 = g.sort_values('house_win').head(10).copy()
FULL_H = 181*24
decomp = []
for _, r in b10.iterrows():
    A = r.avail_hours
    zc = Z_COIN_HR[r.zone_main]; th = T_HOLDS[r.theme]
    W0 = FULL_H*FLOOR_COIN_HR*FLOOR_HOLD
    W1 = A*FLOOR_COIN_HR*FLOOR_HOLD
    W2 = A*zc*FLOOR_HOLD
    W3 = A*zc*th
    W4 = A*r.coin_hr*th
    W5 = A*r.coin_hr*r.hold
    decomp.append(dict(machine=r.machine_id,
        floor_norm=W0, exposure=W1-W0, zone=W2-W1, game=W3-W2, own_play=W4-W3, hold_luck=W5-W4, actual=W5))

# ---- fair ranking: win per available hour + play residual ----
fair = g.sort_values('win_hr')
fair_worst = fair.head(15)[['machine_id','zone_main','theme','win_hr','coin_hr','loc_index','play_resid_mean','t','house_win','verdict']]

# ---- M-1158 monthly + peers ----
m1158 = mm[mm.machine_id=='M-1158'][['month','coin_in','house_win','theo','hold','theo_hold']]
dg = a[a.theme=='Desert Gold'].groupby('machine_id').apply(lambda x: x.house_win_usd.sum()/x.coin_in_usd.sum())

# ---- floor monthly ----
p = pd.read_parquet('data/1_prepared_views/property_day.parquet')
for c in ['coin_in_usd','house_win_usd','theo_usd','total_zone_entries_observed','faults_opened']:
    p[c] = pd.to_numeric(p[c], errors='coerce')
p['month'] = pd.to_datetime(p.gaming_date).dt.to_period('M').astype(str)
fm = p.groupby('month')[['coin_in_usd','house_win_usd','theo_usd','total_zone_entries_observed','faults_opened']].sum().reset_index()
# downtime lost win by month (from analysis3)
dt = a.groupby('month').apply(lambda x: ((24-x.available_hours)*(x.coin_in_usd/x.available_hours.replace(0,np.nan))*(x.theo_usd/x.coin_in_usd.replace(0,np.nan))).sum())

# ---- model holdout scatter (sample) ----
te = pd.read_parquet('out/model_holdout.parquet')
te_s = te.sample(1200, random_state=1)[['pred','actual','zone_name']]
# floor forecast series
p['wd'] = p.local_weekday
p['gdt'] = pd.to_datetime(p.gaming_date)
trp = p[p.gdt < '2025-05-01']
wd_mean = trp.groupby('local_weekday').coin_in_usd.mean()
const = trp.coin_in_usd.mean()
fc = p[p.gdt >= '2025-05-01'][['gaming_date','coin_in_usd','local_weekday']].copy()
fc['pred_wd'] = fc.local_weekday.map(wd_mean); fc['pred_const'] = const

# ---- zone stats incl entries ----
zstat = a.groupby('zone_name').agg(machines=('machine_id','nunique'), coin=('coin_in_usd','sum'),
    avail=('available_hours','sum'), entries=('zone_entries_observed','mean')).reset_index()
zstat['coin_hr']=zstat.coin/zstat.avail
zstat['win_hr']=zstat.coin/zstat.avail  # placeholder fix below
wa = a.groupby('zone_name')['house_win_usd'].sum()
zstat['win_hr'] = wa.values/zstat.avail.values
zstat['entries'] = zstat.entries.round(0)

# theme stats
tstat = theme_agg.reset_index()
tstat['theo_hold']=tstat.theo/tstat.coin
tstat['win_hr']=tstat.win/tstat.avail
tstat['coin_hr']=tstat.coin/tstat.avail

# bottom10 daily series
b10ids = b10.machine_id.tolist()
ds = a[a.machine_id.isin(b10ids)][['machine_id','gaming_date','coin_in_usd','house_win_usd','theo_usd','available_hours']].copy()
ds['gaming_date'] = ds.gaming_date.astype(str)

# machines master table for dashboard
cols = ['machine_id','zone_main','theme','days','house_win','theo','luck','coin_hr','win_hr','hold',
        'theo_hold','avail_ratio','loc_index','play_resid_mean','t','faults','verdict']
mst = b10[cols].copy()
mst['bank'] = mst.machine_id.map(bank)
mst['install'] = mst.machine_id.map(inst)
mst['retire'] = mst.machine_id.map(ret)
mst['rank'] = range(1,11)
mst['mfr'] = mst.machine_id.map(dict(zip(m.machine_id, m.manufacturer)))
# luck z-score
lzd = a.groupby('machine_id')['luck_residual_usd'].agg(['sum','std','count'])
lzd['z'] = lzd['sum']/(lzd['std']*np.sqrt(lzd['count']))
mst['lz'] = mst.machine_id.map(lzd['z'])
mst['vclass'] = mst.verdict.map({'LEAK':'leak','LOCATION':'loc','GAME MIX + ZONE':'game','NEW INSTALL':'new','RETIRED EARLY':'ret'})

# all-machines scatter (period = full)
allm = g[['machine_id','zone_main','theme','house_win','coin_hr','win_hr','hold','theo_hold','verdict']].copy()
allm['bank'] = allm.machine_id.map(bank)
allm['bottom10'] = allm.machine_id.isin(b10ids)

out = dict(
    floor_coin_hr=FLOOR_COIN_HR, floor_hold=FLOOR_HOLD, z_coin_hr=Z_COIN_HR, t_holds=T_HOLDS,
    machine_month=mm.to_dict('records'),
    machines=mst.to_dict('records'),
    all_machines=allm.to_dict('records'),
    decomp=decomp,
    m1158_monthly=m1158.to_dict('records'),
    dg_peers=dg.to_dict(),
    floor_monthly=fm.to_dict('records'),
    downtime_monthly={k: float(v) for k,v in dt.items()},
    zones=zstat.to_dict('records'),
    themes=tstat.to_dict('records'),
    b10_daily=ds.to_dict('records'),
    holdout_scatter=te_s.to_dict('records'),
    forecast=fc[['gaming_date','coin_in_usd','pred_wd','pred_const']].assign(
        gaming_date=fc.gaming_date.astype(str)).to_dict('records'),
    mae=dict(model=41.32, base=46.70, floor_wd=68699, floor_const=218258),
    fair_worst=fair_worst.to_dict('records'),
)
with open('out/dash_data.json','w') as f:
    json.dump(out, f, default=str)
import os
print('json size: %.1f MB' % (os.path.getsize('out/dash_data.json')/1e6))
print('mm rows:', len(mm), 'b10 daily:', len(ds), 'scatter:', len(te_s))
print('floor coin_hr %.1f hold %.3f' % (FLOOR_COIN_HR, FLOOR_HOLD))
print('Z_COIN_HR', {k: round(v,1) for k,v in Z_COIN_HR.items()})
print('decomp sample:', decomp[1])
print('fair worst 15 by win_hr:'); print(fair_worst.to_string(index=False))

# ---------- inject into dashboard template ----------
html = open('dashboard/index.template.html').read()
assert '__DATA__' in html, 'template placeholder missing'
html = html.replace('__DATA__', json.dumps(out, separators=(',', ':'), default=str))
open('dashboard/index.html', 'w').write(html)
print('dashboard/index.html rebuilt (%.1f MB)' % (len(html)/1e6))
