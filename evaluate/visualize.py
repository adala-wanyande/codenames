import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf
import numpy as np

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
    match_df = pd.read_csv(
        os.path.join(
            LOG_DIR,
            "tournament_results_vs_baseline.csv"
        )
    )

    turn_df = pd.read_csv(
        os.path.join(
            LOG_DIR,
            "tournament_results_vs_baseline_turns.csv"
        )
    )

    # =====================================================
    # BACKWARD COMPATIBILITY
    # =====================================================

    defaults = {
        "Model_Name": "qwen2.5",
        "Model_Label": "Qwen-2.5",
        "Temperature": 0.0,
        "Prompt_Strategy": "Unknown",
        "Shot": 0,
        "Winner": "Unknown",
        "Red_Model": "unknown",
        "Blue_Model": "word2vec",
        "Fallback": 0,
        "Invalid_Clue": 0,
        "Assassin_Hit": 0,
    }
    for col, default in defaults.items():

        if col not in match_df.columns:
            match_df[col] = default

        if col not in turn_df.columns:
            turn_df[col] = default

    # =====================================================
    # WIN FLAGS
    # =====================================================

    match_df["Red_Win"] = (
        match_df["Winner"] == "Red"
    ).astype(int)

    match_df["Blue_Win"] = (
        match_df["Winner"] == "Blue"
    ).astype(int)

    # =====================================================
    # CONFIG LABEL
    # =====================================================

    match_df["Config_Label"] = match_df.apply(
        lambda r:
            f"{r['Prompt_Strategy']} (T=0.7)"
            if r["Temperature"] == 0.7
            else r["Prompt_Strategy"],
        axis=1
    )

    turn_df["Config_Label"] = turn_df.apply(
        lambda r:
            f"{r['Prompt_Strategy']} (T=0.7)"
            if r["Temperature"] == 0.7
            else r["Prompt_Strategy"],
        axis=1
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
    _save("win_rate_by_strategy_vs_baseline.png")


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
    plt.setp(
        ax.get_xticklabels(),
        rotation=20,
        ha="right",
        fontsize=9
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save("turns_vs_prompt_depth_vs_baseline.png")


# ---------------------------------------------------------------------------
# Plot 3 — Invalid Clue & Fallback Rate by Prompt Strategy
# ---------------------------------------------------------------------------

def plot_invalid_rate_vs_prompt_depth(turn_df):
    df = turn_df[
        (turn_df["Temperature"] != 0.7) &
        (turn_df["Team"] == "Red")
    ].copy()

    strategies = [s for s in PROMPT_ORDER if s in df["Prompt_Strategy"].unique()]

    agg = (
        df.groupby("Prompt_Strategy")
        .agg(Invalid_Rate=("Invalid_Clue", "mean"), Fallback_Rate=("Fallback_Clue", "mean"))
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
    _save("invalid_rate_vs_prompt_depth_vs_baseline.png")


# ---------------------------------------------------------------------------
# Plot 4 — Safety Metrics by Model Architecture
# ---------------------------------------------------------------------------

def plot_safety_by_model(match_df, turn_df):
    # Use only T=0.0 to avoid double-counting Qwen-2.5
    mdf = match_df[match_df["Temperature"] != 0.7]
    tdf = turn_df[turn_df["Temperature"] != 0.7]

    match_assassin = (
        tdf.groupby(["Match_ID", "Red_Model"])["Assassin_Hit"]
        .max()
        .reset_index()
    )

    assassin_rate = (
        match_assassin.groupby("Red_Model")["Assassin_Hit"]
        .mean()
    )

    invalid_rate = (
        tdf.groupby("Red_Model")["Invalid_Clue"]
        .mean()
    )

    models = sorted(
        set(assassin_rate.index)
        | set(invalid_rate.index)
    )

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
    _save("safety_by_model_vs_baseline.png")


# ---------------------------------------------------------------------------
# Plot 5 — Temperature Ablation (T=0.0 vs T=0.7)
# ---------------------------------------------------------------------------

def plot_temperature_ablation(match_df, turn_df):
    """
    Side-by-side bar: win rate and compliance metrics for
    CoT+Few-Shot Qwen-2.5 at T=0.0 vs T=0.7.
    """
    ablation = match_df[
        (match_df["Red_Model"] == "qwen2.5") &
        (match_df["Prompt_Strategy"] == "CoT+Few-Shot")
    ].copy()

    agg = ablation.groupby("Temperature").agg(
        Win_Rate=("Red_Win", "mean"),
        Avg_Turns=("Turns_Taken", "mean"),
        N=("Red_Win", "count"),
    ).reset_index()

    tabl = turn_df[
        (turn_df["Red_Model"] == "qwen2.5") &
        (turn_df["Prompt_Strategy"] == "CoT+Few-Shot") &
        (turn_df["Team"] == "Red")
    ].groupby("Temperature").agg(
        Invalid_Rate=("Invalid_Clue", "mean"),
        Fallback_Rate=("Fallback_Clue", "mean"),
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
    _save("temperature_ablation_vs_baseline.png")


def plot_red_vs_baseline(match_df):

    df = match_df.copy()

    agg = (
        df.groupby("Prompt_Strategy")
        .agg(
            Win_Rate=("Red_Win", "mean"),
            Avg_Turns=("Turns_Taken", "mean"),
            Games=("Red_Win", "count"),
        )
        .reset_index()
    )

    strategies = [
        s for s in PROMPT_ORDER
        if s in agg["Prompt_Strategy"].values
    ]

    agg = (
        agg.set_index("Prompt_Strategy")
        .loc[strategies]
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        agg["Prompt_Strategy"],
        agg["Win_Rate"],
        color="#4e79a7",
        alpha=0.85,
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        color="black"
    )

    ax.set_ylim(0, 1.05)

    ax.yaxis.set_major_formatter(
        mticker.PercentFormatter(xmax=1)
    )

    ax.set_title(
        "Experimental Agent Win Rate vs Word2Vec Baseline",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel("Win Rate")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, agg["Win_Rate"]):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.0%}",
            ha="center",
            fontsize=9
        )

    plt.xticks(rotation=20)

    plt.tight_layout()

    _save("red_vs_baseline_winrate_vs_baseline.png")

# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

def run_statistical_tests(match_df):
    # Exclude T=0.7 from strategy comparison
    df = match_df[match_df["Temperature"] != 0.7]
    groups = {
        label: group["Red_Win"].values
        for label, group in df.groupby("Prompt_Strategy")
    }

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

# =========================================================
# EFFECT SIZE INTERPRETATION
# =========================================================

def interpret_odds_ratio(or_value):
    """
    Interprets odds ratios symmetrically around 1.0.

    OR > 1  -> positive effect
    OR < 1  -> negative effect
    """

    if np.isnan(or_value):
        return "undefined"

    # ---------------------------------------------
    # Neutral
    # ---------------------------------------------

    if 0.95 <= or_value <= 1.05:
        return "negligible"

    # ---------------------------------------------
    # Positive effects
    # ---------------------------------------------

    elif 1.05 < or_value <= 1.25:
        return "very small positive"

    elif 1.25 < or_value <= 1.75:
        return "small positive"

    elif 1.75 < or_value <= 2.50:
        return "moderate positive"

    elif or_value > 2.50:
        return "large positive"

    # ---------------------------------------------
    # Negative effects
    # ---------------------------------------------

    elif 0.80 <= or_value < 0.95:
        return "very small negative"

    elif 0.57 <= or_value < 0.80:
        return "small negative"

    elif 0.40 <= or_value < 0.57:
        return "moderate negative"

    else:
        return "large negative"


# =========================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================

def bootstrap_winrate(samples, n_boot=10000, confidence=0.95):
    """
    Bootstrap confidence interval for binary win-rate data.
    """

    samples = np.asarray(samples)

    if len(samples) == 0:
        return np.nan, np.nan

    means = np.empty(n_boot)

    for i in range(n_boot):
        resample = np.random.choice(
            samples,
            size=len(samples),
            replace=True
        )
        means[i] = np.mean(resample)

    alpha = 1 - confidence

    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return lower, upper


# =========================================================
# COHEN'S H EFFECT SIZE
# =========================================================

def cohens_h(p1, p2):
    """
    Cohen's h effect size for proportions.
    """

    return 2 * (
        np.arcsin(np.sqrt(p1)) -
        np.arcsin(np.sqrt(p2))
    )


def interpret_cohens_h(h):

    h = abs(h)

    if h < 0.20:
        return "negligible"

    elif h < 0.50:
        return "small"

    elif h < 0.80:
        return "medium"

    else:
        return "large"


# =========================================================
# MAIN STATISTICAL ANALYSIS
# =========================================================

def run_statistical_tests_baseline(match_df):

    # =====================================================
    # FILTER
    # =====================================================

    df = match_df[
        match_df["Temperature"] != 0.7
    ].copy()

    print("\n" + "=" * 85)
    print("PAIRWISE STATISTICAL TESTS VS BASELINE")
    print("=" * 85)

    # =====================================================
    # BASELINE
    # =====================================================

    baseline_df = df[
        df["Prompt_Strategy"] == "Baseline"
    ]

    baseline_samples = baseline_df["Red_Win"].values

    baseline_wins = baseline_samples.sum()

    baseline_total = len(baseline_samples)

    baseline_losses = baseline_total - baseline_wins

    baseline_rate = np.mean(baseline_samples)

    # Wilson interval
    baseline_ci_low, baseline_ci_high = proportion_confint(
        baseline_wins,
        baseline_total,
        alpha=0.05,
        method="wilson"
    )

    # Bootstrap interval
    boot_low, boot_high = bootstrap_winrate(baseline_samples)

    print(
        f"\nBaseline Win Rate: "
        f"{baseline_rate:.1%}   "
        f"Wilson CI=[{baseline_ci_low:.1%}, {baseline_ci_high:.1%}]   "
        f"Bootstrap CI=[{boot_low:.1%}, {boot_high:.1%}]"
    )

    # =====================================================
    # PAIRWISE TESTS
    # =====================================================

    print("\n" + "-" * 85)
    print("Pairwise Fisher Exact Tests")
    print("-" * 85)

    strategies = [
        s for s in sorted(df["Prompt_Strategy"].unique())
        if s != "Baseline"
    ]

    summary_rows = []

    for strategy in strategies:

        strat_df = df[
            df["Prompt_Strategy"] == strategy
        ]

        samples = strat_df["Red_Win"].values

        strat_wins = samples.sum()

        strat_total = len(samples)

        strat_losses = strat_total - strat_wins

        strat_rate = np.mean(samples)

        # =================================================
        # Confidence intervals
        # =================================================

        ci_low, ci_high = proportion_confint(
            strat_wins,
            strat_total,
            alpha=0.05,
            method="wilson"
        )

        boot_low, boot_high = bootstrap_winrate(samples)

        # =================================================
        # Fisher exact test
        # =================================================

        table = [
            [baseline_wins, baseline_losses],
            [strat_wins, strat_losses]
        ]

        odds_ratio, p = stats.fisher_exact(table)

        # =================================================
        # Effect sizes
        # =================================================

        delta = strat_rate - baseline_rate

        h = cohens_h(strat_rate, baseline_rate)

        effect_label = interpret_odds_ratio(odds_ratio)

        h_label = interpret_cohens_h(h)

        # =================================================
        # Interpretation
        # =================================================

        practical = (
            "meaningful"
            if abs(delta) >= 0.10
            else "small"
        )

        significance = "*" if p < 0.05 else "n.s."

        # =================================================
        # Print
        # =================================================

        print(
            f"{strategy:<20}"
            f"Win={strat_rate:>6.1%}   "
            f"Δ={delta:+6.1%}   "
            f"Wilson=[{ci_low:.1%}, {ci_high:.1%}]   "
            f"Boot=[{boot_low:.1%}, {boot_high:.1%}]   "
            f"OR={odds_ratio:>6.2f} ({effect_label})   "
            f"h={h:>5.2f} ({h_label})   "
            f"p={p:.4f}   "
            f"{significance}"
        )

        summary_rows.append({
            "Strategy": strategy,
            "Win_Rate": strat_rate,
            "Delta": delta,
            "Odds_Ratio": odds_ratio,
            "Odds_Interpretation": effect_label,
            "Cohens_h": h,
            "Effect_Size": h_label,
            "P_Value": p,
        })

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    print("\n" + "=" * 85)
    print("LOGISTIC REGRESSION")
    print("=" * 85)

    try:

        regression_df = df.copy()

        regression_df["Prompt_Strategy"] = (
            regression_df["Prompt_Strategy"]
            .astype("category")
        )

        # Explicit baseline reference
        regression_df["Prompt_Strategy"] = (
            regression_df["Prompt_Strategy"]
            .cat.reorder_categories(
                [
                    "Baseline",
                    "Zero-Shot",
                    "Few-Shot",
                    "CoT",
                    "CoT+Few-Shot",
                    "SR-CoT",
                    "SR-CoT+Few-Shot",
                ],
                ordered=True
            )
        )

        # =================================================
        # MODEL
        # =================================================

        model = smf.logit(
            formula="""
                Red_Win ~ C(Prompt_Strategy)
            """,
            data=regression_df
        ).fit(disp=False)

        print(model.summary())

        # =================================================
        # ODDS RATIOS
        # =================================================

        print("\n" + "-" * 85)
        print("Odds Ratios")
        print("-" * 85)

        odds_ratios = pd.DataFrame({
            "Odds_Ratio": np.exp(model.params),
            "CI_Low": np.exp(model.conf_int()[0]),
            "CI_High": np.exp(model.conf_int()[1]),
            "P_Value": model.pvalues,
        })

        odds_ratios["Effect"] = odds_ratios[
            "Odds_Ratio"
        ].apply(interpret_odds_ratio)

        print(
            odds_ratios.round(4)
        )

        # =================================================
        # MODEL INTERPRETATION
        # =================================================

        print("\n" + "-" * 85)
        print("Model Interpretation")
        print("-" * 85)

        llr_p = model.llr_pvalue

        print(
            f"Likelihood Ratio Test p-value: {llr_p:.4f}"
        )

        if llr_p < 0.05:
            print(
                "Overall prompt strategy has a statistically "
                "significant relationship with win probability."
            )
        else:
            print(
                "No statistically significant overall relationship "
                "between prompt strategy and win probability was detected."
            )

        print(
            f"Pseudo R² (McFadden): {model.prsquared:.4f}"
        )

    except Exception as e:

        print(f"Regression failed: {e}")

    print("=" * 85)
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
    plot_red_vs_baseline(match_df)
    run_statistical_tests_baseline(match_df)
    

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    run_visualization()
