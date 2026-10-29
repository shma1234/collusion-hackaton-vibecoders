"""
Shared runner for the synthetic pricing experiments (hackathon edition).

Every script in this folder (monopoly.py, duopoly.py, oligopoly_*.py) is a thin
wrapper around ``run_setting``. The CLI follows the "Collusion Research Sprint"
project 1.1 recipe:

  1. Reduce the thesis sweep to ONE prompt, ONE demand parameter, ONE seed
     (these are the defaults) and verify sustained prices above the Nash
     benchmark - the end-of-run summary prints PASS/FAIL for that.
  2. Vary exactly one feature while holding the economics fixed:
       --memory 0|10|100      visible rounds of history in the prompt
       --history-order newest|oldest   order of those rounds in the prompt
       --no-notes            persistent PLANS/INSIGHTS scratchpad off
       --shuffle-names        competitor labels re-shuffled every round
       --temperature 0.7      sampling temperature
  3. Expand --seeds only after the positive control works.

Inference: the shared endpoint (Nebius or OpenRouter; set NEBIUS_API_KEY in the repo-root compute.env).
--model / MODEL_NAME is any open model id the shared config can resolve; left unset it defaults to the
thesis model, mistralai/Mistral-Large-Instruct-2411. Nothing is ever substituted.

    python experiments_synthetic/duopoly.py --rounds 30          # smoke test (~60 calls)
    python experiments_synthetic/duopoly.py                      # positive control, 600 calls
    python experiments_synthetic/duopoly.py --memory 10 --seeds 1 2 3
    python experiments_synthetic/duopoly.py --no-notes
    python experiments_synthetic/oligopoly_3.py --shuffle-names
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.agents.LLM_agent import LLMAgent  # noqa: E402
from src.agents.llm_providers import (  # noqa: E402
    PRESETS,
    estimate_cost_note,
    make_provider,
    resolve_effective_model,
)
from src.environment.calvano import CalvanoDemandEnvironment  # noqa: E402
from src.experiment.experiment import Experiment  # noqa: E402
from src.prompts.prompts import GENERAL_PROMPT, GENERAL_PROMPT_NO_NOTES, P1, P2  # noqa: E402
from src.prompts.prompts_models import create_pricing_response_model  # noqa: E402


PROMPTS = {"P1": P1, "P2": P2}
FIRM_NAMES = ["Firm A", "Firm B", "Firm C", "Firm D", "Firm E", "Firm F", "Firm G", "Firm H"]
RUNS_DIR = Path(__file__).resolve().parent / "experiments_runs"
PROVIDER_CHOICES = ["mistral", *sorted(PRESETS)]


async def run_setting(
    n_firms: int,
    prompt: str = "P1",
    alpha: float = 1.0,
    rounds: int = 300,
    seed: Optional[int] = 0,
    memory_length: int = 100,
    history_order: str = "newest",
    notes: bool = True,
    shuffle_names: bool = False,
    temperature: float = 0.7,
    provider_kind: Optional[str] = None,
    model: Optional[str] = None,
    experiment_name: Optional[str] = None,
    runs_dir: Path = RUNS_DIR,
    plot_every: int = 10,
) -> Experiment:
    provider = make_provider(provider_kind, model=model)
    response_model = create_pricing_response_model(
        include_wtp=True, wtp_value=4.51 * alpha, include_notes=notes
    )
    template = GENERAL_PROMPT if notes else GENERAL_PROMPT_NO_NOTES
    env_params = {"a": 2.0, "alpha": alpha, "c": 1.0}

    agents = [
        LLMAgent(
            name,
            prefix=PROMPTS[prompt],
            provider=provider,
            response_model=response_model,
            memory_length=memory_length,
            history_order=history_order,
            temperature=temperature,
            seed=seed,
            prompt_template=template,
            env_params=env_params,
        )
        for name in FIRM_NAMES[:n_firms]
    ]

    label = {1: "monopoly", 2: "duopoly"}.get(n_firms, f"oligopoly_{n_firms}_firms")
    if experiment_name is None:
        tags = [label, "setting", prompt, f"mem{memory_length}"]
        if history_order != "newest":
            tags.append(f"hist{history_order}")
        if not notes:
            tags.append("nonotes")
        if shuffle_names:
            tags.append("shuffled")
        if temperature != 0.7:
            tags.append(f"t{temperature}")
        tags.append(f"seed{seed}")
        experiment_name = "_".join(tags)

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description=f"{label} environment with Calvano 2020 demand",
    )
    experiment = Experiment(
        name=experiment_name,
        agents=agents,
        num_rounds=rounds,
        environment=env,
        experiment_dir=runs_dir,
        plot_every=plot_every,
        shuffle_names=shuffle_names,
        seed=seed,
    )
    await experiment.run()
    return experiment


def make_parser(n_firms_default: int, description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    g = ap.add_argument_group("economics (hold fixed while ablating)")
    g.add_argument("--firms", type=int, default=n_firms_default, help="number of firms")
    g.add_argument("--rounds", type=int, default=300, help="game length (thesis: 300)")
    g.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0],
        help="demand parameter(s); thesis grid was 1 3.2 10",
    )
    g.add_argument(
        "--prompts",
        nargs="+",
        default=["P1"],
        choices=["P1", "P2"],
        help="P1 = profit-focused, P2 = exploration-friendly",
    )

    a = ap.add_argument_group("ablations (change one at a time)")
    a.add_argument(
        "--memory",
        type=int,
        default=100,
        help="visible rounds of history in the prompt (try 0, 10, 100)",
    )
    a.add_argument(
        "--history-order",
        default="newest",
        choices=["newest", "oldest"],
        help="order of the visible past rounds in the prompt (thesis: newest first)",
    )
    a.add_argument(
        "--no-notes", action="store_true", help="remove the persistent PLANS/INSIGHTS scratchpad"
    )
    a.add_argument(
        "--shuffle-names",
        action="store_true",
        help="re-label competitors randomly every round (matters with 3+ firms)",
    )
    a.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    a.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="one run per seed; start with 1, expand to 3-5 after the control passes",
    )

    m = ap.add_argument_group("model")
    m.add_argument(
        "--provider",
        default=None,
        choices=PROVIDER_CHOICES,
        help="LLM provider (default: env LLM_PROVIDER or openrouter)",
    )
    m.add_argument(
        "--model",
        default=None,
        help="model id (default: env MODEL_NAME, else the thesis model mistralai/Mistral-Large-Instruct-2411)",
    )

    o = ap.add_argument_group("output")
    o.add_argument(
        "--plot-every",
        type=int,
        default=10,
        help="re-render results_plot.svg every N rounds (0 = only at the end)",
    )
    o.add_argument("-y", "--yes", action="store_true", help="skip the cost confirmation")
    return ap


def main_from_cli(n_firms_default: int, description: str) -> None:
    args = make_parser(n_firms_default, description).parse_args()
    n_runs = len(args.seeds) * len(args.alphas) * len(args.prompts)
    n_calls = n_runs * args.rounds * args.firms
    provider = args.provider or os.getenv("LLM_PROVIDER", "openrouter")
    # Show the model the run will actually resolve to (respects LLM_MODEL / OpenRouter
    # id-translation), matching estimate_cost_note below — no silent swap.
    model = resolve_effective_model(args.model)
    print(
        f"{n_runs} run(s) × {args.rounds} rounds × {args.firms} firms via {provider}/{model}\n"
        f"{estimate_cost_note(n_calls, args.model)}"
    )
    if not args.yes and provider != "ollama" and n_calls > 100 and sys.stdin.isatty():
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("aborted")
    try:
        _run_all(args)
    except (FileNotFoundError, ValueError, KeyError, ConnectionError, RuntimeError) as e:
        sys.exit(f"\n❌ {e}")


def _run_all(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        for alpha in args.alphas:
            for prompt in args.prompts:
                flags = [f"mem={args.memory}"]
                if args.history_order != "newest":
                    flags.append(f"history={args.history_order}-first")
                if args.no_notes:
                    flags.append("notes=off")
                if args.shuffle_names:
                    flags.append("names=shuffled")
                if args.temperature != 0.7:
                    flags.append(f"T={args.temperature}")
                print(
                    f"▶ {args.firms} firms | {prompt} | alpha={alpha} | seed={seed} | "
                    f"{args.rounds} rounds | {' '.join(flags)}"
                )
                asyncio.run(
                    run_setting(
                        n_firms=args.firms,
                        prompt=prompt,
                        alpha=alpha,
                        rounds=args.rounds,
                        seed=seed,
                        memory_length=args.memory,
                        history_order=args.history_order,
                        notes=not args.no_notes,
                        shuffle_names=args.shuffle_names,
                        temperature=args.temperature,
                        provider_kind=args.provider,
                        model=args.model,
                        plot_every=args.plot_every,
                    )
                )
                print("✅ done\n")
