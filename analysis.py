import os
os.makedirs("out", exist_ok=True)

import pandas as pd
import numpy as np

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)

d = pd.read_parquet('data/1_prepared_views/prepared_daily_view.parquet')
for c in ['coin_in_usd','payout_usd','house_win_usd','theo_usd','luck_residual_usd',
          'carded_coin_in_usd','lease_cost_usd','service_cost_usd','power_cost_usd',
          'repair_cost_usd','contribution_after_supplied_costs_usd','configured_rtp']:
    d[c] = pd.to_numeric(d[c], errors='coerce')
d['gaming_date'] = pd.to_datetime(d['gaming_date'])

a = d[d.active & d.location_id.notna()].copy()
print('active machine-days:', len(a), 'machines:', a.machine_id.nunique())

# ---- scheduled seconds for availability denominator ----
cal = pd.read_parquet('data/2_machine_decision/calendar_day.parquet')
d['sched_hours'] = d.gaming_date.map(dict(zip(cal.gaming_date, cal.scheduled_seconds/3600.0)))

# ---- floor-level machine stats ----
g = a.groupby('machine_id').agg(
    days=('gaming_date','count'),
    first_day=('gaming_date','min'), last_day=('gaming_date','max'),
    zone_main=('zone_name', lambda s: s.mode().iat[0]),
    zones=('zone_name','nunique'),
    theme=('theme','first'),
    rtp=('configured_rtp','mean'),
    vol=('volatility_class','first'),
    mfr=('manufacturer','first'),
    coin_in=('coin_in_usd','sum'),
    house_win=('house_win_usd','sum'),
    theo=('theo_usd','sum'),
    luck=('luck_residual_usd','sum'),
    avail_hours=('available_hours','sum'),
    faults=('faults_opened','sum'),
    occ=('occupied_hours','sum'),
    repair_cost=('repair_cost_usd','sum'),
    costs=('contribution_after_supplied_costs_usd','sum'),
).reset_index()
g['theo_hold'] = g.theo/g.coin_in
g['hold'] = g.house_win/g.coin_in
g['win_hr'] = g.house_win/g.avail_hours          # $ win per available hour
g['coin_hr'] = g.coin_in/g.avail_hours           # play per available hour
g['avail_ratio'] = g.avail_hours/(g.days*24)
g['occ_ratio'] = g.occ/g.avail_hours

# ---- zone-normalized location effect ----
# a machine's play intensity vs the average of its zone (per available hour)
a['coin_hr_day'] = a.coin_in_usd / a.available_hours.replace(0, np.nan)
zone_hr = a.groupby('zone_name')['coin_hr_day'].mean()
g['zone_coin_hr'] = g.zone_main.map(zone_hr)
g['loc_index'] = g.coin_hr / g.zone_coin_hr   # >1: machine draws more play than zone average

g_sorted = g.sort_values('house_win')
bottom10 = g_sorted.head(10)
print('\n=== MANAGER BOTTOM 10 (total house win, on-floor days) ===')
print(bottom10[['machine_id','zone_main','theme','days','coin_in','house_win','theo','luck',
                'avail_ratio','win_hr','coin_hr','loc_index','faults','repair_cost']].to_string(index=False))

# diagnostics
print('\n=== floor medians for context ===')
print(g[['coin_in','house_win','theo','luck','win_hr','coin_hr','avail_ratio','loc_index']].describe().loc[['mean','50%']].to_string())

# luck: is actual win below theo (machine under-earned vs its own math)?
# exposure: coin_hr vs floor median
# location: loc_index vs zone
# downtime: avail_ratio, faults
print('\n=== per-machine verdict table ===')
out = bottom10.copy()
def verdict(r):
    tags=[]
    # luck: luck residual negative AND large vs theo
    if r.luck < -0.5*r.theo: tags.append('UNLUCKY (actual << theo)')
    # exposure: low play per available hour
    if r.coin_hr < g.coin_hr.median()*0.7: tags.append('LOW PLAY/EXPOSURE')
    # location: draws less play than zone average
    if r.loc_index < 0.7: tags.append('BAD LOCATION (vs zone peers)')
    # downtime
    if r.avail_ratio < 0.9: tags.append('OFTEN BROKEN (downtime)')
    # hold: actual hold far below configured RTP implies
    if r.hold < 0.5*r.theo_hold: tags.append('PAYOUT LEAK? (hold << theo hold)')
    return '; '.join(tags) if tags else 'no single flag'
out['verdict'] = out.apply(verdict, axis=1)
print(out[['machine_id','luck','loc_index','avail_ratio','hold','theo_hold','faults','verdict']].to_string(index=False))

# luck significance: binomial-ish std of luck residual
# daily luck std per machine, z-score of cumulative luck
a['luck_day'] = a.luck_residual_usd
lz = a.groupby('machine_id')['luck_day'].agg(['sum','std','count'])
lz['z'] = lz['sum']/(lz['std']*np.sqrt(lz['count']))
out2 = bottom10.merge(lz[['z']], left_on='machine_id', right_index=True)
print('\n=== luck z-scores (cumulative luck vs daily variance) ===')
print(out2[['machine_id','luck','z']].to_string(index=False))

g.to_csv('out/machine_stats.csv', index=False)
a.to_parquet('out/active_days.parquet')
print('\nsaved machine_stats.csv and active_days.parquet')
