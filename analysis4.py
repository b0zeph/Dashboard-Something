import os
os.makedirs("out", exist_ok=True)

import pandas as pd
import numpy as np
import json

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)

a = pd.read_parquet('out/active_days.parquet')
g = pd.read_csv('out/machine_stats.csv')
ac = pd.read_parquet('data/2_machine_decision/action_catalog.parquet')
print('=== action catalog ===')
print(ac.to_string(index=False))

a = a[a.available_hours > 0].copy()
a['coin_hr_day'] = a.coin_in_usd / a.available_hours
a['date'] = pd.to_datetime(a.gaming_date)
a['month_n'] = a.date.dt.month

# ---------- Model: expected play intensity from zone+theme+weekday+month ----------
train = a[a.date < '2025-05-01']
test  = a[a.date >= '2025-05-01']
feat_tr = train[['zone_name','theme','local_weekday','month_n','coin_hr_day']].dropna()
feat_tr = feat_tr[feat_tr.coin_hr_day > 0]
feat_te = test[['machine_id','zone_name','theme','local_weekday','month_n','coin_hr_day']].dropna()

# group means on log scale
lg = np.log(feat_tr.coin_hr_day)
def addmeans(keys):
    m = lg.groupby([feat_tr[k] for k in keys]).mean()
    return m
m_zone = addmeans(['zone_name'])
m_theme = addmeans(['theme'])
m_wd = addmeans(['local_weekday'])
m_month = addmeans(['month_n'])
mu = lg.mean()

def predict(df):
    p = mu.copy()
    # additive in log space
    z = df.zone_name.map(m_zone).fillna(mu)
    t = df.theme.map(m_theme).fillna(mu)
    w = df.local_weekday.map(m_wd).fillna(mu)
    mo = df.month_n.map(m_month).fillna(mu)
    pred = mu + (z-mu) + (t-mu) + (w-mu) + (mo-mu)
    return np.exp(pred)

# baseline: per-machine training mean coin_hr
mach_mean = feat_tr.assign(mid=train['machine_id'][feat_tr.index]).groupby('mid').coin_hr_day.mean()

te = feat_te.copy()
te['pred'] = predict(te)
te['mid'] = test['machine_id'][te.index]
te['base'] = te.mid.map(mach_mean)
te['actual'] = te.coin_hr_day
mae_model = (te.pred - te.actual).abs().mean()
mae_base = (te.base - te.actual).abs().mean()
print('\nMODEL MAE (May-Jun, $/avail-hr): %.2f  vs baseline %.2f  (n=%d)' % (mae_model, mae_base, len(te)))

# daily floor forecast comparison too (guide route): weekday mean vs constant
p = pd.read_parquet('data/1_prepared_views/property_day.parquet')
p['coin_in_usd'] = pd.to_numeric(p.coin_in_usd, errors='coerce')
p['date'] = pd.to_datetime(p.gaming_date)
trp = p[p.date < '2025-05-01']; tep = p[p.date >= '2025-05-01']
wd_mean = trp.groupby('local_weekday').coin_in_usd.mean()
const = trp.coin_in_usd.mean()
pred_wd = tep.local_weekday.map(wd_mean)
print('FLOOR daily coin-in MAE: weekday model $%.0f vs constant-mean $%.0f' % ((pred_wd-tep.coin_in_usd).abs().mean(), (const-tep.coin_in_usd).abs().mean()))

# ---------- machine-specific residual: own play intensity vs model expectation ----------
a2 = a.dropna(subset=['coin_hr_day']).copy()
a2['pred_coin_hr'] = predict(a2)
a2['play_resid'] = a2.coin_hr_day - a2.pred_coin_hr  # >0: draws more play than location/game/weekday predict
mr = a2.groupby('machine_id').agg(play_resid_mean=('play_resid','mean'),
                                  play_resid_se=('play_resid','sem'))
mr['t'] = mr.play_resid_mean / mr.play_resid_se
g2 = g.merge(mr, left_on='machine_id', right_index=True)

# ---------- final bottom-10 table with all diagnostics ----------
b10 = g2.sort_values('house_win').head(10).copy()
print('\n=== BOTTOM 10 final diagnostics ===')
print(b10[['machine_id','zone_main','theme','days','house_win','theo','luck','avail_ratio',
           'coin_hr','loc_index','play_resid_mean','t','faults']].to_string(index=False))

# ---------- counterfactual: expected coin_hr if machine were in each zone ----------
zones = sorted(a.zone_name.unique())
def cf(machine, zone):
    row = a[a.machine_id==machine]
    base = row[['theme','local_weekday','month_n']].copy()
    base['zone_name'] = zone
    return predict(base).mean()
print('\n=== counterfactual mean coin_hr by zone for selected machines ===')
for mch in ['M-1181','M-1265','M-1179','M-1035','M-1017','M-1238']:
    vals = {z: cf(mch,z) for z in zones}
    print(mch, {k: round(v,1) for k,v in sorted(vals.items(), key=lambda x:-x[1])})

g2.to_csv('out/machine_stats_full.csv', index=False)
te.to_parquet('out/model_holdout.parquet')
print('\nsaved machine_stats_full.csv, model_holdout.parquet')
