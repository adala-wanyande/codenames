import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


LOG_DIR = "data/logs"
FIG_DIR = "data/figures"

def build_global_color_map(turn_df):
    groups = sorted(
        turn_df[["Spymaster_Type", "Shot"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )

    palette = sns.color_palette("tab10", len(groups))

    return {group: palette[i] for i, group in enumerate(groups)}

def load_data():
    match_path = os.path.join(LOG_DIR, "tournament_results.csv")
    turn_path = os.path.join(LOG_DIR, "tournament_results_turns.csv")

    match_df = pd.read_csv(match_path)
    turn_df = pd.read_csv(turn_path)

    return match_df, turn_df


# -------------------------
# 1. WIN RATE ANALYSIS
# -------------------------
def plot_win_rates(match_df):
    plt.figure()

    win_rates = match_df.groupby("Spymaster_Type")["Win"].apply(
        lambda x: (x == "Red").mean()  # adjust if Red is not "win"
    )

    win_rates.plot(kind="bar")
    plt.title("Win Rate by Spymaster Type")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45)

    save_path = os.path.join(FIG_DIR, "win_rates.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
# -------------------------
# 2. AVERAGE GAME LENGTH
# -------------------------
def plot_game_dynamics(match_df, turn_df, color_map):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -------------------------
    # LEFT: Words Left over turns
    # -------------------------
    turn_base = (
        turn_df
        .groupby(["Spymaster_Type", "Shot", "Turn"])["Words_Left"]
        .mean()
        .reset_index()
    )

    for (spymaster, shot), group in turn_base.groupby(["Spymaster_Type", "Shot"]):
        group = group.sort_values("Turn")
        color = color_map[(spymaster, shot)]

        axes[0].plot(
            group["Turn"],
            group["Words_Left"],
            color=color,
            label=f"{spymaster}|{shot}"
        )

    axes[0].set_title("Words Left over Turns")
    axes[0].set_xlabel("Turn")
    axes[0].set_ylabel("Words Left")

    # -------------------------
    # RIGHT: Avg Game Length
    # -------------------------
    pivot = (
        match_df
        .groupby(["Spymaster_Type", "Shot"])["Turns_Taken"]
        .mean()
        .reset_index()
    )

    for (spymaster, shot), group in pivot.groupby(["Spymaster_Type", "Shot"]):
        color = color_map[(spymaster, shot)]
        axes[1].bar(
            f"{spymaster}|{shot}",
            group["Turns_Taken"].values[0],
            color=color
        )

    axes[1].set_title("Average Game Length")
    axes[1].set_ylabel("Turns")

    # -------------------------
    # FIXED LEGEND (OUTSIDE PLOTS)
    # -------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.15)  # push below figure
    )

    # reserve space at top for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom

    save_path = os.path.join(FIG_DIR, "game_dynamics.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
def plot_false_positives(turn_df):
    plt.figure()

    fp = (
        turn_df
        .groupby(["Spymaster_Type", "Shot"])["Invalid_Clue"]
        .mean()
        .reset_index()
    )

    groups = list(fp[["Spymaster_Type", "Shot"]].apply(tuple, axis=1))
    color_map = build_global_color_map(turn_df)

    for _, row in fp.iterrows():
        key = (row["Spymaster_Type"], row["Shot"])
        plt.bar(
            str(key),
            row["Invalid_Clue"],
            color=color_map[key]
        )

    plt.title("Invalid Clue Rate by Agent")
    plt.ylabel("Rate")
    plt.xticks(rotation=45)

    save_path = os.path.join(FIG_DIR, "false_positive_rate.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def extract_shot_and_type(turn_df):
    def parse(match_id):
        parts = match_id.split("_")

        if len(parts) < 3:
            return pd.Series([None, None])

        shot = int(parts[-2])                     # always second-to-last
        spymaster_type = "_".join(parts[:-2])     # everything before

        return pd.Series([spymaster_type, shot])

    turn_df[["Spymaster_Type", "Shot"]] = turn_df["Match_ID"].apply(parse)

    return turn_df




def make_agent_key(spymaster, shot):
    return f"{spymaster}|{shot}"
def plot_clue_vs_guesses(turn_df, color_map):
    base = (
        turn_df
        .groupby(["Spymaster_Type", "Shot", "Turn"])
        .mean(numeric_only=True)
        .reset_index()
    )

    groups = list(base.groupby(["Spymaster_Type", "Shot"]).groups.keys())
    color_map = color_map

    plt.figure()

    for (spymaster, shot), group in base.groupby(["Spymaster_Type", "Shot"]):
        group = group.sort_values("Turn")
        color = color_map[(spymaster, shot)]

        label = f"{spymaster} | shot={shot}"

        # clue
        plt.plot(
            group["Turn"],
            group["Clue_Count"],
            marker="o",
            linestyle="-",
            color=color,
            label=label
        )

        # guesses (NO legend entry)
        plt.plot(
            group["Turn"],
            group["Guesses"],
            marker="x",
            linestyle="--",
            color=color,
            label="_nolegend_"
        )

    plt.title("Clue Count vs Nb of Guesses")
    plt.xlabel("Turn")
    plt.ylabel("Count")
    plt.legend(ncol=3, fontsize=8)

    save_path = os.path.join(FIG_DIR, "clue_vs_guesses_over_turns.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_guess_quality(turn_df, color_map):
    base = (
        turn_df
        .groupby(["Spymaster_Type", "Shot", "Turn"])
        .mean(numeric_only=True)
        .reset_index()
    )

    groups = list(base.groupby(["Spymaster_Type", "Shot"]).groups.keys())
    color_map = color_map

    plt.figure()

    for (spymaster, shot), group in base.groupby(["Spymaster_Type", "Shot"]):
        group = group.sort_values("Turn")
        color = color_map[(spymaster, shot)]

        label = f"{spymaster} | shot={shot}"

        # correct
        plt.plot(
            group["Turn"],
            group["Correct_Guesses"],
            marker="^",
            linestyle="-",
            color=color,
            label=label
        )

        # wrong (NO legend entry)
        plt.plot(
            group["Turn"],
            group["Wrong_Guesses"],
            marker="x",
            linestyle="--",
            color=color,
            label="_nolegend_"
        )

    plt.title("Correct vs Wrong guesses")
    plt.xlabel("Turn")
    plt.ylabel("Count")
    plt.legend(ncol=3, fontsize=8)

    save_path = os.path.join(FIG_DIR, "guess_quality_over_turns.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def plot_assassin_suite(turn_df, color_map):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------------------------
    # 1. HIT RATE PER GAME
    # -------------------------
    match_assassin = (
        turn_df
        .groupby(["Match_ID", "Spymaster_Type", "Shot"])["Assassin_Hit"]
        .max()
        .reset_index()
    )

    hit_rate = (
        match_assassin
        .groupby(["Spymaster_Type", "Shot"])["Assassin_Hit"]
        .mean()
        .reset_index()
    )

    x_labels = []

    for (spymaster, shot), group in hit_rate.groupby(["Spymaster_Type", "Shot"]):
        color = color_map[(spymaster, shot)]
        label = f"{spymaster}|{shot}"
        x_labels.append(label)

        axes[0].bar(
            label,
            group["Assassin_Hit"].values[0],
            color=color
        )

    axes[0].set_title("Assassin Hit Rate")
    axes[0].tick_params(axis='x', rotation=45)   # ✅ FIX HERE

    # -------------------------
    # 2. AVERAGE TURN HIT
    # -------------------------
    assassin_turns = (
        turn_df[turn_df["Assassin_Hit"] == 1]
        .groupby(["Match_ID", "Spymaster_Type", "Shot"])["Turn"]
        .min()
        .reset_index()
    )

    avg_turn = (
        assassin_turns
        .groupby(["Spymaster_Type", "Shot"])["Turn"]
        .mean()
        .reset_index()
    )

    for (spymaster, shot), group in avg_turn.groupby(["Spymaster_Type", "Shot"]):
        color = color_map[(spymaster, shot)]

        axes[1].bar(
            f"{spymaster}|{shot}",
            group["Turn"].values[0],
            color=color
        )

    axes[1].set_title("Avg Turn of Assassin Hit")
    axes[1].tick_params(axis='x', rotation=45)   # ✅ FIX HERE

    # -------------------------
    # 3. DISTRIBUTION (unchanged)
    # -------------------------
    for (spymaster, shot), group in assassin_turns.groupby(["Spymaster_Type", "Shot"]):
        color = color_map[(spymaster, shot)]

        if group["Turn"].nunique() < 2:
            continue

        sns.kdeplot(
            data=group,
            x="Turn",
            ax=axes[2],
            fill=True,
            alpha=0.25,
            color=color,
            label=f"{spymaster}|{shot}",
            warn_singular=False
        )

    axes[2].set_title("Assassin Turn Distribution")

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "assassin_suite.png")
    plt.savefig(save_path)
    plt.close()
# -------------------------
# MAIN ENTRY
# -------------------------
def run_visualization():
    os.makedirs(FIG_DIR, exist_ok=True)

    match_df, turn_df = load_data()
    turn_df = extract_shot_and_type(turn_df)

    color_map = build_global_color_map(turn_df)

    print("📊 Loaded datasets:")
    print("Match rows:", len(match_df))
    print("Turn rows:", len(turn_df))

    print("📈 Generating plots...")

    plot_game_dynamics(match_df, turn_df, color_map)
    plot_guess_quality(turn_df, color_map)
    plot_false_positives(turn_df)
    plot_clue_vs_guesses(turn_df, color_map)
    plot_assassin_suite(turn_df, color_map)

    print(f"✅ All figures saved to {FIG_DIR}/")
if __name__ == "__main__":
    run_visualization()