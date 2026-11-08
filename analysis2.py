import pandas as pd
import numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)

a = pd.read_parquet('out/active_days.parquet')
g = pd.read_csv('out/machine_stats.csv')

# ---- zone-level picture ----
z = a.groupby('zone_name').agg(
    machines=('machine_id','nunique'),
    coin_in=('coin_in_usd','sum'), house_win=('house_win_usd','sum'),
    avail=('available_hours','sum'), entries=('zone_entries_observed','mean'),
).reset_index()
z['coin_hr'] = z.coin_in/z.avail
z['win_hr'] = z.house_win/z.avail
print('=== ZONES ===')
print(z.sort_values('coin_hr').to_string(index=False))

# ---- machines table: install/retire ----
m = pd.read_parquet('data/2_machine_decision/machines.parquet')
print('\n=== machines detail for bottom10 ===')
b10 = g.sort_values('house_win').head(10).machine_id.tolist()
print(m[m.machine_id.isin(b10)].to_string(index=False))

# ---- M-1158 deep dive: daily hold over time ----
print('\n=== M-1158 monthly: hold vs theo_hold ===')
s = a[a.machine_id=='M-1158']
print(s.groupby('month').agg(coin=('coin_in_usd','sum'), win=('house_win_usd','sum'),
      theo=('theo_usd','sum'), luck=('luck_residual_usd','sum')).assign(
      hold=lambda x: x.win/x.coin, theo_hold=lambda x: x.theo/x.coin).to_string())

# compare with other Desert Gold machines
print('\n=== all Desert Gold machines: hold ===')
dg = a[a.theme=='Desert Gold']
dd = dg.groupby('machine_id').agg(coin=('coin_in_usd','sum'), win=('house_win_usd','sum'), theo=('theo_usd','sum'))
dd['hold']=dd.win/dd.coin; dd['theo_hold']=dd.theo/dd.coin
print(dd.sort_values('hold').to_string())

# game_math for Desert Gold
gm = pd.read_parquet('data/2_machine_decision/game_math.parquet')
print('\ngame math:'); print(gm.to_string(index=False))

# ---- monthly floor trend ----
p = pd.read_parquet('data/1_prepared_views/property_day.parquet')
for c in p.columns:
    if p[c].dtype==object and c not in ('gaming_date',):
        try: p[c]=pd.to_numeric(p[c], errors='ignore')
        except: pass
print('\n=== property_day columns ===')
print(list(p.columns))
num = p.select_dtypes('number')
pm = p.groupby(p['month'] if 'month' in p else pd.to_datetime(p.gaming_date).dt.to_period('M'))[num.columns].sum()
print(pm.T.to_string())
