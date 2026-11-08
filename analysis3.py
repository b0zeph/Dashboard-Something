import pandas as pd
import numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)

a = pd.read_parquet('out/active_days.parquet')
g = pd.read_csv('out/machine_stats.csv')

# ---- monthly floor demand ----
p = pd.read_parquet('data/1_prepared_views/property_day.parquet')
for c in ['coin_in_usd','house_win_usd','theo_usd','available_machine_hours','faults_opened',
          'total_zone_entries_observed','entrance_zone_entries_observed']:
    p[c] = pd.to_numeric(p[c], errors='coerce')
p['month'] = pd.to_datetime(p.gaming_date).dt.to_period('M')
pm = p.groupby('month')[['coin_in_usd','house_win_usd','theo_usd','available_machine_hours','faults_opened','total_zone_entries_observed']].sum()
print('=== monthly floor ==='); print(pm.T.to_string())

# ---- Royal Draw theme: is it structurally low-hold? ----
t = a.groupby('theme').agg(coin=('coin_in_usd','sum'), win=('house_win_usd','sum'),
                           theo=('theo_usd','sum'), avail=('available_hours','sum'))
t['win_hr']=t.win/t.avail; t['coin_hr']=t.coin/t.avail; t['theo_hold']=t.theo/t.coin
print('\n=== themes ==='); print(t.sort_values('win_hr').to_string())

# ---- downtime: floor-wide lost play ----
a['down_hr'] = 24 - a.available_hours
tot_down = a.down_hr.sum()
# lost play estimate: coin_hr of that machine-day's zone * downtime
a['coin_hr_day'] = a.coin_in_usd/a.available_hours.replace(0,np.nan)
a['lost_play'] = a.down_hr * a.coin_hr_day
a['lost_win'] = a.lost_play * (a.theo_usd/a.coin_in_usd.replace(0,np.nan))
print('\n=== downtime by month ===')
dm = a.groupby('month').agg(down_hours=('down_hr','sum'), lost_play=('lost_play','sum'), lost_win=('lost_win','sum'))
print(dm.to_string())
print('total lost theo win to downtime: $%.0f over %d machine-hours down' % (a.lost_win.sum(), tot_down))

# bottom10 downtime
b10 = g.sort_values('house_win').head(10).machine_id.tolist()
bb = a[a.machine_id.isin(b10)].groupby('machine_id').agg(down=('down_hr','sum'), lost_win=('lost_win','sum'), faults=('faults_opened','sum'))
print('\nbottom10 downtime:'); print(bb.to_string())

# ---- zone traffic (Back Forty entries) ----
zt = pd.read_parquet('data/4_context/zone_traffic_hourly.parquet')
print('\nzone traffic cols:', list(zt.columns)[:20])
print(zt.head(3).to_string())

# ---- events ----
ev = pd.read_parquet('data/4_context/events.parquet')
print('\nevents:'); print(ev.to_string(index=False))
