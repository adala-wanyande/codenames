import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

LOG_DIR = "data/logs"
FIG_DIR = "data/figures"

# Canonical left-to-right ordering for prompt depth axis.
# Temperature ablation variant is kept separate so it doesn't pollute the
# main strategy axis.
PROMPT_ORDER = [
    "Baseline", "Zero-Shot", "Few-Shot",
    "CoT", "CoT+Few-Shot", "SR-CoT", "SR-CoT+Few-Shot",
]

# Consistent model colour map used across all figures
MODEL_PALETTE = {
    "Word2Vec": "#4e79a7",
    "Qwen-2.5": "#f28e2b",
    "Qwen-3":   "#59a14f",
    "Mistral":  "#e15759",
    "LLaMA-3":  "#76b7b2",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    match_df = pd.read_csv(os.path.join(LOG_DIR, "tournament_results.csv"))
    turn_df  = pd.read_csv(os.path.join(LOG_DIR, "tournament_results_turns.csv"))

    for col, default in [("Model_Name", "qwen2.5"), ("Model_Label", "Qwen-2.5"),
                         ("Temperature", 0.0), ("Prompt_Strategy", "Unknown"),
                         ("Shot", 0)]:
        if col not in match_df.columns:
            match_df[col] = default
        if col not in turn_df.columns:
            turn_df[col] = default

    match_df["Red_Win"] = (match_df["Win"] == "Red").astype(int)

    # Split temperature ablation rows into their own label so they don't
    # pollute the main CoT+Few-Shot bar in Figure 1.
    match_df["Config_Label"] = match_df.apply(
        lambda r: r["Prompt_Strategy"] + " (T=0.7)"
        if r["Temperature"] == 0.7 else r["Prompt_Strategy"],
        axis=1,
    )
    turn_df["Config_Label"] = turn_df.apply(
        lambda r: r["Prompt_Strategy"] + " (T=0.7)"
        if r["Temperature"] == 0.7 else r["Prompt_Strategy"],
        axis=1,
    )

    return match_df, turn_df


def _model_color(model):
    return MODEL_PALETTE.get(model, "#bab0ac")


# ---------------------------------------------------------------------------
# Plot 1 — Win Rate & Game Length by Prompt Strategy
# ---------------------------------------------------------------------------

def plot_win_rate_and_length(match_df):
    """
    Side-by-side grouped bar using Config_Label (so T=0.7 is a separate bar).
    Only renders bars for combinations that have actual data.
    """
    # Exclude temp-ablation from the main strategy plot; it gets its own figure.
    df = match_df[match_df["Temperature"] != 0.7].copy()

    agg = (
        df.groupby(["Prompt_Strategy", "Model_Label"])
        .agg(Win_Rate=("Red_Win", "mean"), Avg_Turns=("Turns_Taken", "mean"),
             N=("Red_Win", "count"))
        .reset_index()
    )

    strategies = [s for s in PROMPT_ORDER if s in agg["Prompt_Strategy"].unique()]
    max_per_group = max(len(agg[agg["Prompt_Strategy"] == s]) for s in strategies)
    bar_width = 0.8 / max(max_per_group, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    legend_handles: dict = {}

    for si, strategy in enumerate(strategies):
        group = agg[agg["Prompt_Strategy"] == strategy].reset_index(drop=True)
        n = len(group)
        for j, row in group.iterrows():
            model  = row["Model_Label"]
            offset = (j - n / 2 + 0.5) * bar_width
            xpos   = si + offset
            color  = _model_color(model)

            b0 = axes[0].bar(xpos, row["Win_Rate"],  width=bar_width * 0.9, color=color)
            axes[1].bar(xpos, row["Avg_Turns"], width=bar_width * 0.9, color=color)

            if model not in legend_handles:
                legend_handles[model] = b0[0]

    for ax, title, ylabel in [
        (axes[0], "Win Rate by Prompt Strategy",   "Win Rate"),
        (axes[1], "Average Game Length by Prompt Strategy", "Avg Turns"),
    ]:
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0, 1.05)
    baseline_line = axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    handles = list(legend_handles.values()) + [baseline_line]
    labels  = list(legend_handles.keys())   + ["50% chance"]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save("win_rate_by_strategy.png")


# ---------------------------------------------------------------------------
# Plot 2 — Game Length Distribution by Prompt Strategy (boxplot)
# ---------------------------------------------------------------------------

def plot_turns_vs_prompt_depth(match_df):
    df = match_df[match_df["Temperature"] != 0.7].copy()
    strategies = [s for s in PROMPT_ORDER if s in df["Prompt_Strategy"].unique()]
    if len(strategies) < 2:
        print("  Skipping turns_vs_prompt_depth — need ≥2 prompt strategies.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df, x="Prompt_Strategy", y="Turns_Taken",
        order=strategies,
        palette=[_model_color(
            df[df["Prompt_Strategy"] == s]["Model_Label"].iloc[0]
        ) for s in strategies],
        width=0.5, ax=ax,
        hue="Prompt_Strategy", legend=False,
    )
    ax.set_title("Game Length Distribution by Prompt Strategy",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Prompt Strategy")
    ax.set_ylabel("Turns to Finish")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save("turns_vs_prompt_depth.png")


# ---------------------------------------------------------------------------
# Plot 3 — Invalid Clue & Fallback Rate by Prompt Strategy
# ---------------------------------------------------------------------------

def plot_invalid_rate_vs_prompt_depth(turn_df):
    df = turn_df[turn_df["Temperature"] != 0.7].copy()
    strategies = [s for s in PROMPT_ORDER if s in df["Prompt_Strategy"].unique()]

    agg = (
        df.groupby("Prompt_Strategy")
        .agg(Invalid_Rate=("Invalid_Clue", "mean"), Fallback_Rate=("Fallback", "mean"))
        .reindex(strategies)
        .reset_index()
    )

    x, width = range(len(strategies)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([xi - width / 2 for xi in x], agg["Invalid_Rate"],
           width=width, label="Invalid Clue Rate",
           color=sns.color_palette("tab10")[0], alpha=0.85)
    ax.bar([xi + width / 2 for xi in x], agg["Fallback_Rate"],
           width=width, label="Fallback Rate",
           color=sns.color_palette("tab10")[1], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Rule Compliance: Invalid Clue & Fallback Rate by Prompt Strategy",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Rate (fraction of turns)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    _save("invalid_rate_vs_prompt_depth.png")


# ---------------------------------------------------------------------------
# Plot 4 — Safety Metrics by Model Architecture
# ---------------------------------------------------------------------------

def plot_safety_by_model(match_df, turn_df):
    # Use only T=0.0 to avoid double-counting Qwen-2.5
    mdf = match_df[match_df["Temperature"] != 0.7]
    tdf = turn_df[turn_df["Temperature"] != 0.7]

    match_assassin = (
        tdf.groupby(["Match_ID", "Model_Label"])["Assassin_Hit"]
        .max().reset_index()
    )
    assassin_rate = match_assassin.groupby("Model_Label")["Assassin_Hit"].mean()
    invalid_rate  = tdf.groupby("Model_Label")["Invalid_Clue"].mean()

    models = sorted(set(assassin_rate.index) | set(invalid_rate.index))
    x, width = range(len(models)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([xi - width / 2 for xi in x],
           [assassin_rate.get(m, 0) for m in models],
           width=width, label="Assassin Hit Rate",
           color=sns.color_palette("tab10")[3], alpha=0.85)
    ax.bar([xi + width / 2 for xi in x],
           [invalid_rate.get(m, 0) for m in models],
           width=width, label="Invalid Clue Rate",
           color=sns.color_palette("tab10")[1], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Safety Metrics by Model Architecture",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Rate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    _save("safety_by_model.png")


# ---------------------------------------------------------------------------
# Plot 5 — Temperature Ablation (T=0.0 vs T=0.7)
# ---------------------------------------------------------------------------

def plot_temperature_ablation(match_df, turn_df):
    """
    Side-by-side bar: win rate and compliance metrics for
    CoT+Few-Shot Qwen-2.5 at T=0.0 vs T=0.7.
    """
    ablation = match_df[
        (match_df["Model_Name"] == "qwen2.5") &
        (match_df["Prompt_Strategy"] == "CoT+Few-Shot")
    ].copy()

    agg = ablation.groupby("Temperature").agg(
        Win_Rate=("Red_Win", "mean"),
        Avg_Turns=("Turns_Taken", "mean"),
        N=("Red_Win", "count"),
    ).reset_index()

    tabl = turn_df[
        (turn_df["Model_Name"] == "qwen2.5") &
        (turn_df["Prompt_Strategy"] == "CoT+Few-Shot")
    ].groupby("Temperature").agg(
        Invalid_Rate=("Invalid_Clue", "mean"),
        Fallback_Rate=("Fallback", "mean"),
    ).reset_index()

    agg = agg.merge(tabl, on="Temperature")
    labels = [f"T={t:.1f}\n(n={int(n)})" for t, n in zip(agg["Temperature"], agg["N"])]
    x = range(len(agg))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4e79a7", "#e15759"]

    for ax, col, title, fmt in [
        (axes[0], "Win_Rate",     "Win Rate",           "pct"),
        (axes[1], "Invalid_Rate", "Invalid Clue Rate",  "pct"),
        (axes[2], "Fallback_Rate","Fallback Rate",       "pct"),
    ]:
        bars = ax.bar(x, agg[col], color=colors[:len(agg)], width=0.5, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Temperature Ablation: CoT+Few-Shot (Qwen-2.5)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save("temperature_ablation.png")


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

def run_statistical_tests(match_df):
    # Exclude T=0.7 from strategy comparison
    df = match_df[match_df["Temperature"] != 0.7]
    groups = {l: g["Red_Win"].values for l, g in df.groupby("Prompt_Strategy")}

    print("\n" + "=" * 55)
    print("Statistical Significance Tests (T=0.0 only)")
    print("=" * 55)

    if "Baseline" in groups:
        baseline = groups["Baseline"]
        llm_wins = df[df["Prompt_Strategy"] != "Baseline"]["Red_Win"].values
        if len(llm_wins) > 0 and len(baseline) > 0:
            t, p = stats.ttest_ind(baseline, llm_wins, equal_var=False)
            print(f"T-test  Baseline vs all LLM:  t={t:.3f},  p={p:.4f}  {'*' if p<0.05 else 'n.s.'}")

    anova_groups = [v for v in groups.values() if len(v) > 1]
    if len(anova_groups) >= 2:
        f, p = stats.f_oneway(*anova_groups)
        print(f"ANOVA   all prompt strategies: F={f:.3f},  p={p:.4f}  {'*' if p<0.05 else 'n.s.'}")

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
# Main
# ---------------------------------------------------------------------------

def run_visualization():
    os.makedirs(FIG_DIR, exist_ok=True)
    match_df, turn_df = load_data()
    print(f"Loaded {len(match_df)} matches, {len(turn_df)} turns.")
    print(f"Strategies: {sorted(match_df['Prompt_Strategy'].unique())}")
    print(f"Models:     {sorted(match_df['Model_Label'].unique())}")
    print("\nGenerating plots...")

    plot_win_rate_and_length(match_df)
    plot_turns_vs_prompt_depth(match_df)
    plot_invalid_rate_vs_prompt_depth(turn_df)
    plot_safety_by_model(match_df, turn_df)
    plot_temperature_ablation(match_df, turn_df)
    run_statistical_tests(match_df)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    run_visualization()
