# The Gilded Ace — Luck, Location, or a Leak?

Interactive dashboard for the TAMU Datathon **Dashboard Royale** challenge (data release 0.9.0-draft).
Question: are the casino's 10 lowest-earning slot machines truly underperforming, or just
unlucky, badly placed, or often broken?

## What's in this bundle

```
gilded-ace-submission/
├── dashboard/
│   ├── index.html            the finished, self-contained dashboard (open in any browser)
│   └── index.template.html   same file with the data blob replaced by __DATA__
├── code/
│   ├── analysis.py           bottom-10 list, per-machine stats, luck z-scores, verdicts
│   ├── analysis2.py          zone economics, M-1158 vs Desert Gold peers, floor trends
│   ├── analysis3.py          downtime cost, theme/game-mix economics, events
│   ├── analysis4.py          play-intensity model, time-based holdout validation,
│   │                         counterfactual "what if moved to zone X" projections
│   └── build_data.py         aggregates everything, emits out/dash_data.json and
│                             rebuilds dashboard/index.html from the template
└── README.md
```

## How to reproduce

Requires Python 3.11+ with `pandas`, `numpy`, `pyarrow` (`pip install pandas numpy pyarrow`).

1. Get the challenge data:
   ```sh
   git clone --depth 1 https://github.com/tamu-datathon-org/dashboard-royale.git
   ```
2. Unzip this bundle **inside** the cloned `dashboard-royale` folder, so you have:
   ```
   dashboard-royale/
   ├── data/          (from the clone)
   ├── code/          (from this bundle)
   └── dashboard/     (from this bundle)
   ```
3. From the `dashboard-royale` folder, run the scripts in order:
   ```sh
   python3 code/analysis.py     # writes out/machine_stats.csv, out/active_days.parquet
   python3 code/analysis2.py
   python3 code/analysis3.py
   python3 code/analysis4.py    # writes out/machine_stats_full.csv, out/model_holdout.parquet
   python3 code/build_data.py   # rebuilds dashboard/index.html (1.1 MB, no server needed)
   ```
4. Open `dashboard/index.html` in a browser. Everything (data, charts, fonts via CDN,
   Chart.js via CDN) is embedded or loaded from public CDNs — no backend required.

The rebuilt `dashboard/index.html` is byte-identical to the deployed version.

## Method summary

- **Manager's list**: bottom 10 by SUM(house_win_usd) over active machine-days (Jan 1 – Jun 30, 2025),
  matching the guide's reference query. Pre-install/post-retire rows excluded.
- **Luck**: cumulative (actual − theo) win, z-scored against each machine's own daily variance;
  |z| < 2 treated as noise.
- **Location**: coin-in per available machine-hour vs zone and floor norms; location index =
  machine play intensity ÷ zone average.
- **Leak**: M-1158 hold vs configured theo hold and vs 48 Desert Gold peers (3.0% vs 10% theo,
  peers 7–14%, z = −4.5).
- **Exposure / downtime / game mix**: days on floor, availability ratio, fault counts,
  downtime cost (downtime hours × play rate × theo hold), theme theo-hold table.
- **Model**: additive log-linear model of daily play intensity (zone + theme + weekday + month),
  trained Jan–Apr, tested May–Jun. MAE $41.32 vs per-machine-mean baseline $46.70.
  Floor forecast: weekday means (MAE $68,699) vs constant mean ($218,258). No leakage.
- **Budget**: 3 action types, $13,360 of $100,000 — inspect M-1158 ($1,200), move 3 machines
  out of Back Forty ($5,400), lease-swap 2 Royal Draw titles ($6,760). Each with cost,
  expected effect, uncertainty, and a diff-in-diff measurement plan.

## Dashboard features (what judges can test)

- Change the analysis period (month-range filter recomputes KPIs, table, zone and scatter charts)
- Drill into one bank (scatter filter) or one machine (click any table row → case-file drawer
  with gap decomposition waterfall and daily trend)
- Find the model validation (Section 06: holdout scatter, MAE cards, forecast chart)
- Trace a recommendation to its evidence (each budget card links to its section;
  Claims & Evidence appendix maps all 10 headline claims to table → result → alternative → limitation)

