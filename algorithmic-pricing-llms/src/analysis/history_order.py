"""History-order ablation: run-level comparison over a common window (rounds FIRST..LAST).

Run from algorithmic-pricing-llms/:  python src/analysis/history_order.py
RUNS matches the 2026-10-24 Llama-3.3-70B batch (6 seeds per arm); the 10 runs that stopped early
still all cover rounds 1-73, hence the window. One observation per run (both firms averaged), as in the thesis.
  price index  = (p - p_Nash) / (p_mono - p_Nash)          0 = competitive, 1 = monopoly price
  profit index = (pi - pi_Nash) / (pi_mono - pi_Nash)      0 = competitive, 1 = joint-monopoly profit
"""

import glob
import json

import polars as pl
from scipy import stats

FIRST, LAST = 24, 73
RUNS = "experiments_synthetic/experiments_runs/2_agents/179285103[78]_duopoly_setting_P1_mem100*_seed[1-6]"

rows = []
for d in sorted(glob.glob(RUNS)):
    with open(f"{d}/metadata.json") as f:
        m = json.load(f)
    env = m["environment"]["environment_params"]
    pn, pm = env["nash_prices"][0], env["monopoly_prices"][0]
    qn, qm = env["nash_profits"][0], env["monopoly_profits"][0]
    h = pl.read_parquet(f"{d}/environment_history.parquet").filter(
        pl.col("round").is_between(FIRST, LAST)
    )
    assert h["round"].n_unique() == LAST - FIRST + 1, f"{d} is missing rounds"
    per_round = h.group_by("round").agg(
        pl.col("price").mean().alias("p"),
        pl.col("profit").mean().alias("pi"),
        (pl.col("price").max() - pl.col("price").min()).alias("gap"),
    )
    rows.append(
        {
            "arm": m["agents_history_order"]["Firm A"],
            "seed": m["seed"],
            "mean_price": h["price"].mean(),
            "price_index": (h["price"].mean() - pn) / (pm - pn),
            "profit_index": (h["profit"].mean() - qn) / (qm - qn),
            "share_rounds_above_monopoly": (per_round["p"] > pm).mean(),
            "price_sd_over_time": per_round["p"].std(),
            "mean_gap_between_firms": per_round["gap"].mean(),
        }
    )

df = pl.DataFrame(rows).sort("arm", "seed")
pl.Config.set_tbl_rows(20)
pl.Config.set_float_precision(3)
pl.Config.set_tbl_width_chars(200)
print(f"Per-run metrics, rounds {FIRST}-{LAST} (Nash p={pn:.3f}, monopoly p={pm:.3f})")
print(df)

print("\nArm comparison (n=6 runs per arm): mean ± sd | diff (oldest - newest) | Welch t p | Mann-Whitney p")
for col in [c for c in df.columns if c not in ("arm", "seed")]:
    a = df.filter(pl.col("arm") == "newest")[col].to_list()
    b = df.filter(pl.col("arm") == "oldest")[col].to_list()
    t = stats.ttest_ind(b, a, equal_var=False)
    u = stats.mannwhitneyu(b, a, alternative="two-sided")
    sa, sb = pl.Series(a), pl.Series(b)
    print(
        f"  {col:28s} newest {sa.mean():7.3f} ± {sa.std():.3f} | oldest {sb.mean():7.3f} ± {sb.std():.3f}"
        f" | diff {sb.mean() - sa.mean():+7.3f} | t p={t.pvalue:.3f} | MW p={u.pvalue:.3f}"
    )
