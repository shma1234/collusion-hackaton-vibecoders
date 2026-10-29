"""Merge the history-order batch into one long parquet for text analysis.

Run from algorithmic-pricing-llms/:  python src/analysis/history_order_merge.py
One row per (run, round, firm). Runs stopped at different rounds (73-100); every run covers 1-73,
so filter round <= 73 for a balanced comparison (see history_order.py).
"""

import glob
import json
from pathlib import Path

import polars as pl

RUNS = "experiments_synthetic/experiments_runs/2_agents/179285103[78]_duopoly_setting_P1_mem100*_seed[1-6]"
OUT = Path("data/results/history_order_llama.parquet")

rows = []
for d in sorted(glob.glob(RUNS)):
    with open(f"{d}/metadata.json") as f:
        m = json.load(f)
    with open(f"{d}/results.json") as f:
        results = json.load(f)
    env = m["environment"]["environment_params"]
    idx = {name: v["env_index"] for name, v in m["agent_environment_mapping"].items()}
    # only rounds every firm completed (a run can stop mid-round)
    rounds = set.intersection(*(set(map(int, r)) for r in results.values()))
    for agent, per_round in results.items():
        for rnd in sorted(rounds):
            r = per_round[str(rnd)]
            rows.append(
                {
                    "arm": m["agents_history_order"][agent],
                    "seed": m["seed"],
                    "round": rnd,
                    "agent": agent,
                    "chosen_price": float(r["chosen_price"]),
                    "profit": float(r["profit"]),
                    "observations": r["observations"],
                    "plans": r["plans"],
                    "insights": r["insights"],
                    "nash_price": env["nash_prices"][idx[agent]],
                    "monopoly_price": env["monopoly_prices"][idx[agent]],
                }
            )

df = pl.DataFrame(rows).sort("arm", "seed", "round", "agent")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(OUT)
print(f"wrote {OUT}: {df.height:,} rows")
print(df.group_by("arm", "seed").agg(pl.col("round").max().alias("last_round")).sort("arm", "seed"))
