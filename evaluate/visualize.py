import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

LOG_DIR = "data/logs"
FIG_DIR = "data/figures"

# Ordered prompt depth axis for all plots
PROMPT_ORDER = ["Baseline", "Zero-Shot", "Few-Shot", "CoT", "CoT+Few-Shot", "SR-CoT", "SR-CoT+Few-Shot"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    match_df = pd.read_csv(os.path.join(LOG_DIR, "tournament_results.csv"))
    turn_df  = pd.read_csv(os.path.join(LOG_DIR, "tournament_results_turns.csv"))

    # Back-fill columns that older CSVs may not have
    for col, default in [("Model_Name", "qwen2.5"), ("Model_Label", "Qwen-2.5"),
                         ("Temperature", 0.0), ("Prompt_Strategy", "Unknown"),
                         ("Shot", 0)]:
        if col not in match_df.columns:
            match_df[col] = default
        if col not in turn_df.columns:
            turn_df[col] = default

    match_df["Red_Win"] = (match_df["Win"] == "Red").astype(int)
    return match_df, turn_df


# ---------------------------------------------------------------------------
# Paper Plot 1 — Win Rate & Game Length by Agent (grouped bar)
# ---------------------------------------------------------------------------

def plot_win_rate_and_length(match_df):
    """
    Side-by-side grouped bar: Win Rate (%) and Avg Turns to Win.
    Only renders bars for (strategy, model) pairs that have actual data —
    no placeholder empty bars for missing combinations.
    Bar width within each strategy group scales to however many models
    are present for that strategy, so groups always look full.
    """
    agg = (
        match_df.groupby(["Prompt_Strategy", "Model_Label"])
        .agg(Win_Rate=("Red_Win", "mean"), Avg_Turns=("Turns_Taken", "mean"), N=("Red_Win", "count"))
        .reset_index()
    )

    strategies  = [s for s in PROMPT_ORDER if s in agg["Prompt_Strategy"].unique()]
    all_models  = sorted(agg["Model_Label"].unique())
    palette     = sns.color_palette("tab10", len(all_models))
    model_color = dict(zip(all_models, palette))

    # Maximum models in any one strategy — sets the finest bar width
    max_per_group = max(
        len(agg[agg["Prompt_Strategy"] == s]) for s in strategies
    )
    bar_width = 0.8 / max(max_per_group, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    legend_handles: dict = {}

    for strat_idx, strategy in enumerate(strategies):
        group = agg[agg["Prompt_Strategy"] == strategy].reset_index(drop=True)
        n = len(group)
        for j, row in group.iterrows():
            model  = row["Model_Label"]
            offset = (j - n / 2 + 0.5) * bar_width
            xpos   = strat_idx + offset

            b0 = axes[0].bar(xpos, row["Win_Rate"],  width=bar_width * 0.9,
                             color=model_color[model])
            axes[1].bar(xpos, row["Avg_Turns"], width=bar_width * 0.9,
                        color=model_color[model])

            if model not in legend_handles:
                legend_handles[model] = b0[0]

    for ax, title, ylabel in [
        (axes[0], "Win Rate (Red team) by Prompt Strategy", "Win Rate"),
        (axes[1], "Average Game Length by Prompt Strategy",  "Avg Turns"),
    ]:
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    axes[0].set_ylim(0, 1.05)
    baseline_line = axes[0].axhline(
        0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    handles = list(legend_handles.values()) + [baseline_line]
    labels  = list(legend_handles.keys())   + ["50% baseline"]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.08), fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save("win_rate_by_strategy.png")


# ---------------------------------------------------------------------------
# Paper Plot 2 — Avg Turns to Win vs. Prompt Depth (boxplot)
# ---------------------------------------------------------------------------

def plot_turns_vs_prompt_depth(match_df):
    """
    Boxplot: distribution of game length across prompt strategies.
    Answers: does more prompt depth make games faster or slower?
    """
    strategies = [s for s in PROMPT_ORDER if s in match_df["Prompt_Strategy"].unique()]
    if len(strategies) < 2:
        print("  Skipping turns_vs_prompt_depth — need ≥2 prompt strategies in data.")
        return

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=match_df,
        x="Prompt_Strategy", y="Turns_Taken",
        order=strategies,
        palette="tab10",
        width=0.5,
    )
    plt.title("Game Length Distribution by Prompt Depth")
    plt.xlabel("Prompt Strategy")
    plt.ylabel("Turns to Finish")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    _save("turns_vs_prompt_depth.png")


# ---------------------------------------------------------------------------
# Paper Plot 3 — Invalid Clue / Fallback Rate vs. Prompt Depth (bar)
# ---------------------------------------------------------------------------

def plot_invalid_rate_vs_prompt_depth(turn_df):
    """
    Grouped bar: Invalid Clue Rate and Fallback Rate by Prompt Strategy.
    Answers: does chain-of-thought improve rule compliance?
    """
    strategies = [s for s in PROMPT_ORDER if s in turn_df["Prompt_Strategy"].unique()]

    agg = (
        turn_df.groupby("Prompt_Strategy")
        .agg(Invalid_Rate=("Invalid_Clue", "mean"), Fallback_Rate=("Fallback", "mean"))
        .reindex(strategies)
        .reset_index()
    )

    x     = range(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([xi - width / 2 for xi in x], agg["Invalid_Rate"],  width=width, label="Invalid Clue Rate",  color=sns.color_palette("tab10")[0])
    ax.bar([xi + width / 2 for xi in x], agg["Fallback_Rate"], width=width, label="Fallback Rate", color=sns.color_palette("tab10")[1])

    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Rule Compliance: Invalid Clue & Fallback Rate by Prompt Depth")
    ax.set_ylabel("Rate (fraction of turns)")
    ax.legend()
    plt.tight_layout()
    _save("invalid_rate_vs_prompt_depth.png")


# ---------------------------------------------------------------------------
# Paper Plot 4 — Assassin Hit Rate & Invalid Clue Rate by Model (grouped bar)
# ---------------------------------------------------------------------------

def plot_safety_by_model(match_df, turn_df):
    """
    Grouped bar: Assassin Hit Rate and Invalid Clue Rate per model.
    Answers: which model architecture is safest?
    """
    # Assassin hit rate per match, then averaged per model
    match_assassin = (
        turn_df.groupby(["Match_ID", "Model_Label"])["Assassin_Hit"]
        .max().reset_index()
    )
    assassin_rate = match_assassin.groupby("Model_Label")["Assassin_Hit"].mean()
    invalid_rate  = turn_df.groupby("Model_Label")["Invalid_Clue"].mean()

    models = sorted(set(assassin_rate.index) | set(invalid_rate.index))
    x      = range(len(models))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([xi - width / 2 for xi in x],
           [assassin_rate.get(m, 0) for m in models],
           width=width, label="Assassin Hit Rate", color=sns.color_palette("tab10")[3])
    ax.bar([xi + width / 2 for xi in x],
           [invalid_rate.get(m, 0) for m in models],
           width=width, label="Invalid Clue Rate", color=sns.color_palette("tab10")[1])

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Safety Metrics by Model Architecture")
    ax.set_ylabel("Rate")
    ax.legend()
    plt.tight_layout()
    _save("safety_by_model.png")


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

def run_statistical_tests(match_df):
    groups = {
        label: group["Red_Win"].values
        for label, group in match_df.groupby("Prompt_Strategy")
    }

    print("\n" + "=" * 55)
    print("Statistical Significance Tests")
    print("=" * 55)

    if "Baseline" in groups:
        baseline = groups["Baseline"]
        llm_wins = match_df[match_df["Prompt_Strategy"] != "Baseline"]["Red_Win"].values
        if len(llm_wins) > 0 and len(baseline) > 0:
            t, p = stats.ttest_ind(baseline, llm_wins, equal_var=False)
            sig = "*" if p < 0.05 else "n.s."
            print(f"T-test  Baseline vs all LLM:  t={t:.3f},  p={p:.4f}  {sig}")

    anova_groups = [v for v in groups.values() if len(v) > 1]
    if len(anova_groups) >= 2:
        f, p = stats.f_oneway(*anova_groups)
        sig = "*" if p < 0.05 else "n.s."
        print(f"ANOVA   all prompt strategies: F={f:.3f},  p={p:.4f}  {sig}")

    print("=" * 55)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(filename):
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_visualization():
    os.makedirs(FIG_DIR, exist_ok=True)
    match_df, turn_df = load_data()
    print(f"Loaded {len(match_df)} matches, {len(turn_df)} turns.")
    print(f"Prompt strategies: {sorted(match_df['Prompt_Strategy'].unique())}")
    print(f"Models:            {sorted(match_df['Model_Label'].unique())}")
    print("\nGenerating plots...")

    plot_win_rate_and_length(match_df)
    plot_turns_vs_prompt_depth(match_df)
    plot_invalid_rate_vs_prompt_depth(turn_df)
    plot_safety_by_model(match_df, turn_df)
    run_statistical_tests(match_df)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    run_visualization()
