from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import REPORT_ASSETS_DIR, RESULTS_DIR
from utils.io import ensure_dir


EXCLUDED_COLUMNS = {
    "algorithm",
    "model_path",
    "metrics_path",
    "progress_path",
    "plot_path",
    "summary_path",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, str]], preferred_columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows found._"
    if preferred_columns is not None:
        columns = [column for column in preferred_columns if column in rows[0]]
    else:
        columns = [column for column in rows[0].keys() if column not in EXCLUDED_COLUMNS]

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    data_rows = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *data_rows])


def load_best_models() -> dict[str, dict[str, object]]:
    return json.loads((RESULTS_DIR / "best_models.json").read_text(encoding="utf-8"))


def load_final_demo() -> dict[str, object]:
    return json.loads((RESULTS_DIR / "final_demo_model.json").read_text(encoding="utf-8"))


def best_run_paragraph(best_models: dict[str, dict[str, object]], final_demo: dict[str, object]) -> str:
    lines = [
        "## Best Performing Models",
        "",
        f"- `DQN`: `{best_models['dqn']['run_name']}` with mean reward `{best_models['dqn']['mean_reward']:.2f}` and success rate `{best_models['dqn']['success_rate']:.2f}`.",
        f"- `REINFORCE`: `{best_models['reinforce']['run_name']}` with mean reward `{best_models['reinforce']['mean_reward']:.2f}` and success rate `{best_models['reinforce']['success_rate']:.2f}`.",
        f"- `PPO`: `{best_models['ppo']['run_name']}` with mean reward `{best_models['ppo']['mean_reward']:.2f}` and success rate `{best_models['ppo']['success_rate']:.2f}`.",
        f"- `A2C`: `{best_models['a2c']['run_name']}` with mean reward `{best_models['a2c']['mean_reward']:.2f}` and success rate `{best_models['a2c']['success_rate']:.2f}`.",
        "",
        f"The final demo/deployment model is `{final_demo['algorithm'].upper()} {final_demo['run_name']}` because it achieved the strongest held-out performance with mean reward `{final_demo['mean_reward']:.2f}` and success rate `{final_demo['success_rate']:.2f}`.",
        "",
    ]
    return "\n".join(lines)


def discussion_paragraph(best_models: dict[str, dict[str, object]]) -> str:
    return "\n".join(
        [
            "## Results Discussion Draft",
            "",
            "The cumulative reward plots show a clear separation between the strongest actor-critic methods and the weaker baselines in this environment. A2C achieved the best overall held-out performance, followed by PPO, while DQN obtained relatively high reward but failed to convert that reward into successful deliveries. This suggests that the value-based agent benefited from reward shaping but did not learn a reliable end-to-end delivery policy under the chosen configuration.",
            "",
            "PPO and A2C displayed the most practical behavior for the emergency medical drone mission because they achieved non-zero generalization success on unseen states. A2C was the strongest model with a success rate of "
            f"`{best_models['a2c']['success_rate']:.2f}` and mean reward `{best_models['a2c']['mean_reward']:.2f}`, while PPO reached a success rate of `{best_models['ppo']['success_rate']:.2f}` and mean reward `{best_models['ppo']['mean_reward']:.2f}`. REINFORCE learned more slowly and remained unstable, which is consistent with the higher variance normally associated with Monte Carlo policy-gradient methods.",
            "",
            "The convergence metrics also support the qualitative plots. PPO converged earlier than A2C in episode count, but A2C achieved the highest final quality and the best final demo behavior. DQN converged according to reward trend, but because its success rate remained zero, convergence should be interpreted carefully: it converged to a suboptimal behavior rather than to the actual mission objective. This difference between reward optimization and task completion is an important finding for the report discussion.",
            "",
        ]
    )


def build_report_summary() -> str:
    best_models = load_best_models()
    final_demo = load_final_demo()

    dqn_rows = read_csv(RESULTS_DIR / "dqn" / "sweep_summary.csv")
    reinforce_rows = read_csv(RESULTS_DIR / "reinforce" / "sweep_summary.csv")
    ppo_rows = read_csv(RESULTS_DIR / "ppo" / "sweep_summary.csv")
    a2c_rows = read_csv(RESULTS_DIR / "a2c" / "sweep_summary.csv")

    sections = [
        "# Generated Report Summary",
        "",
        "This file was generated from the completed experiment outputs and is intended to help fill the final PDF report.",
        "",
        best_run_paragraph(best_models, final_demo),
        discussion_paragraph(best_models),
        "## Figure References",
        "",
        f"- Cumulative rewards subplot: `{PROJECT_ROOT / 'plots/overview/cumulative_rewards.png'}`",
        f"- Training stability subplot: `{PROJECT_ROOT / 'plots/overview/training_stability.png'}`",
        f"- Convergence and generalization summary: `{PROJECT_ROOT / 'plots/overview/summary_bars.png'}`",
        f"- Random-action visualization: `{PROJECT_ROOT / 'results/random_policy_contact_sheet.png'}`",
        "",
        "## DQN Table",
        "",
        markdown_table(
            dqn_rows,
            [
                "run_name",
                "learning_rate",
                "gamma",
                "buffer_size",
                "batch_size",
                "exploration_fraction",
                "mean_reward",
                "success_rate",
                "convergence_episode",
            ],
        ),
        "",
        "## REINFORCE Table",
        "",
        markdown_table(
            reinforce_rows,
            [
                "run_name",
                "learning_rate",
                "gamma",
                "hidden_dim",
                "entropy_coef",
                "normalize_returns",
                "episodes",
                "mean_reward",
                "success_rate",
            ],
        ),
        "",
        "## PPO Table",
        "",
        markdown_table(
            ppo_rows,
            [
                "run_name",
                "learning_rate",
                "gamma",
                "n_steps",
                "batch_size",
                "gae_lambda",
                "ent_coef",
                "clip_range",
                "mean_reward",
                "success_rate",
                "convergence_episode",
            ],
        ),
        "",
        "## A2C Table",
        "",
        markdown_table(
            a2c_rows,
            [
                "run_name",
                "learning_rate",
                "gamma",
                "n_steps",
                "ent_coef",
                "vf_coef",
                "mean_reward",
                "success_rate",
                "convergence_episode",
            ],
        ),
        "",
        "## Final Demo Command",
        "",
        "```bash",
        ".venv/bin/python main.py --registry-path results/final_demo_model.json --export-trace",
        "```",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    ensure_dir(REPORT_ASSETS_DIR)
    output_path = REPORT_ASSETS_DIR / "generated_report_summary.md"
    output_path.write_text(build_report_summary(), encoding="utf-8")
    print(f"Saved report summary to {output_path}")


if __name__ == "__main__":
    main()
