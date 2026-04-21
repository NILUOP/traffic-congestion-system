"""
CARLA Traffic Congestion Visualization Script
=============================================
Directory structure expected:
  CARLA_0.9.16/
    big_data/
      visualize_traffic.py   <-- this script
    data/raw/                <-- CSV / Parquet session files

Run from inside  CARLA_0.9.16/big_data/:
    python visualize_traffic.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from collections import Counter

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#2d3142",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#3a3d4d",
    "font.family":      "DejaVu Sans",
})

ACCENT   = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff",
            "#ffa657", "#79c0ff", "#ff7b72", "#56d364"]
CONG_MAP = {0: "#3fb950", 1: "#ffa657", 2: "#f78166"}   # free / moderate / congested
CONG_LBL = {0: "Free Flow", 1: "Moderate",  2: "Congested"}

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(raw_dir: str) -> pd.DataFrame:
    csv_files     = glob.glob(os.path.join(raw_dir, "**", "*.csv"),     recursive=True)
    parquet_files = glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True)

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    for f in parquet_files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")

    if not frames:
        raise FileNotFoundError(f"No CSV/Parquet files found under: {raw_dir}")

    df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded {len(df):,} rows from {len(frames)} file(s).")

    # type coercion
    for col in ["speed", "acceleration", "x", "y", "z", "yaw",
                "traffic_density", "hour"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "congestion_label" in df.columns:
        df["congestion_label"] = pd.to_numeric(df["congestion_label"],
                                               errors="coerce").fillna(0).astype(int)
    return df


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def savefig(name: str):
    out = os.path.join(OUT_DIR, name)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    print(f"[SAVE] {out}")
    plt.close("all")


def cong_colors(series):
    return [CONG_MAP.get(v, "#58a6ff") for v in series]


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1 — Overview Dashboard  (2×3 grid)
# ═══════════════════════════════════════════════════════════════
def fig_overview(df: pd.DataFrame):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("CARLA Traffic Congestion — Overview Dashboard",
                 fontsize=18, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1A  Speed distribution by congestion label ──────────────
    ax = fig.add_subplot(gs[0, 0])
    for lbl, grp in df.groupby("congestion_label"):
        ax.hist(grp["speed"].dropna(), bins=50, alpha=0.7,
                color=CONG_MAP.get(lbl, "#58a6ff"),
                label=CONG_LBL.get(lbl, str(lbl)), density=True)
    ax.set_title("Speed Distribution by Congestion", fontsize=11)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── 1B  Congestion label counts ─────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    counts = df["congestion_label"].value_counts().sort_index()
    bars = ax.bar([CONG_LBL.get(i, str(i)) for i in counts.index],
                  counts.values,
                  color=[CONG_MAP.get(i, "#58a6ff") for i in counts.index],
                  edgecolor="#0e1117", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.values.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Congestion Label Distribution", fontsize=11)
    ax.set_ylabel("Row Count")
    ax.grid(True, axis="y")

    # ── 1C  Traffic density histogram ───────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df["traffic_density"].dropna(), bins=40, color=ACCENT[0],
            edgecolor="#0e1117", linewidth=0.5)
    ax.set_title("Traffic Density Distribution", fontsize=11)
    ax.set_xlabel("Traffic Density")
    ax.set_ylabel("Count")
    ax.grid(True)

    # ── 1D  Mean speed per hour ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    hourly = df.groupby("hour")["speed"].mean()
    ax.plot(hourly.index, hourly.values, color=ACCENT[0],
            linewidth=2, marker="o", markersize=5)
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color=ACCENT[0])
    ax.set_title("Mean Speed per Hour of Day", fontsize=11)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean Speed (m/s)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True)

    # ── 1E  Acceleration distribution ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(df["acceleration"].dropna().clip(-10, 10), bins=60,
            color=ACCENT[1], edgecolor="#0e1117", linewidth=0.5)
    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Acceleration Distribution", fontsize=11)
    ax.set_xlabel("Acceleration (m/s²)")
    ax.set_ylabel("Count")
    ax.grid(True)

    # ── 1F  Sessions row count ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if "session_id" in df.columns:
        sess_counts = df["session_id"].value_counts().head(15)
        ax.barh(sess_counts.index.astype(str), sess_counts.values,
                color=ACCENT[3], edgecolor="#0e1117")
        ax.set_title("Rows per Session (top 15)", fontsize=11)
        ax.set_xlabel("Row Count")
        ax.grid(True, axis="x")
    else:
        ax.text(0.5, 0.5, "session_id\nnot found",
                ha="center", va="center", transform=ax.transAxes)

    savefig("01_overview_dashboard.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 — Vehicle Kinematics
# ═══════════════════════════════════════════════════════════════
def fig_kinematics(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Vehicle Kinematics Analysis", fontsize=16, fontweight="bold")
    fig.subplots_adjust(hspace=0.4, wspace=0.38)

    # 2A  Speed vs Acceleration scatter (coloured by congestion)
    ax = axes[0, 0]
    sample = df.sample(min(8000, len(df)), random_state=42)
    for lbl in sorted(sample["congestion_label"].unique()):
        sub = sample[sample["congestion_label"] == lbl]
        ax.scatter(sub["speed"], sub["acceleration"],
                   s=4, alpha=0.4, color=CONG_MAP.get(lbl, "#58a6ff"),
                   label=CONG_LBL.get(lbl, str(lbl)), rasterized=True)
    ax.set_title("Speed vs Acceleration", fontsize=11)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True)

    # 2B  Speed box plot by congestion
    ax = axes[0, 1]
    groups = [df[df["congestion_label"] == l]["speed"].dropna()
              for l in sorted(df["congestion_label"].unique())]
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=1.5))
    for patch, lbl in zip(bp["boxes"], sorted(df["congestion_label"].unique())):
        patch.set_facecolor(CONG_MAP.get(lbl, "#58a6ff"))
        patch.set_alpha(0.8)
    ax.set_xticklabels([CONG_LBL.get(l, str(l))
                        for l in sorted(df["congestion_label"].unique())])
    ax.set_title("Speed Box Plot by Congestion", fontsize=11)
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, axis="y")

    # 2C  Yaw distribution
    ax = axes[0, 2]
    ax.hist(df["yaw"].dropna() % 360, bins=72, color=ACCENT[4],
            edgecolor="#0e1117", linewidth=0.3)
    ax.set_title("Yaw Angle Distribution (0–360°)", fontsize=11)
    ax.set_xlabel("Yaw (°)")
    ax.set_ylabel("Count")
    ax.grid(True)

    # 2D  Speed CDF
    ax = axes[1, 0]
    for lbl, grp in df.groupby("congestion_label"):
        s = grp["speed"].dropna().sort_values()
        ax.plot(s, np.linspace(0, 1, len(s)),
                color=CONG_MAP.get(lbl, "#58a6ff"),
                linewidth=2, label=CONG_LBL.get(lbl, str(lbl)))
    ax.set_title("Cumulative Speed Distribution", fontsize=11)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)
    ax.grid(True)

    # 2E  Acceleration vs Traffic Density
    ax = axes[1, 1]
    ax.scatter(df["traffic_density"].sample(min(6000, len(df)), random_state=1),
               df["acceleration"].sample(min(6000, len(df)), random_state=1),
               s=4, alpha=0.3, color=ACCENT[1], rasterized=True)
    ax.set_title("Acceleration vs Traffic Density", fontsize=11)
    ax.set_xlabel("Traffic Density")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.grid(True)

    # # 2F  Speed violin by hour bucket
    # ax = axes[1, 2]
    # df["hour_bucket"] = pd.cut(df["hour"], bins=[0, 6, 12, 18, 24],
    #                            labels=["Night\n0-6", "Morning\n6-12",
    #                                    "Afternoon\n12-18", "Evening\n18-24"])
    # vp_data = [df[df["hour_bucket"] == b]["speed"].dropna()
    #            for b in ["Night\n0-6", "Morning\n6-12",
    #                      "Afternoon\n12-18", "Evening\n18-24"]]
    # vp = ax.violinplot(vp_data, positions=range(4), showmedians=True)
    # for body, col in zip(vp["bodies"], ACCENT):
    #     body.set_facecolor(col)
    #     body.set_alpha(0.7)
    # ax.set_xticks(range(4))
    # ax.set_xticklabels(["Night\n0-6", "Morning\n6-12",
    #                     "Afternoon\n12-18", "Evening\n18-24"], fontsize=8)
    # ax.set_title("Speed Violin by Time of Day", fontsize=11)
    # ax.set_ylabel("Speed (m/s)")
    # ax.grid(True, axis="y")
    # 2F  Speed violin by hour bucket
    ax = axes[1, 2]
    # Define the buckets
    bins = [0, 6, 12, 18, 24]
    labels = ["Night\n0-6", "Morning\n6-12", "Afternoon\n12-18", "Evening\n18-24"]
    
    df["hour_bucket"] = pd.cut(df["hour"], bins=bins, labels=labels, include_lowest=True)
    
    # Prepare data and track which labels actually have data
    plot_data = []
    plot_labels = []
    
    for label in labels:
        subset = df[df["hour_bucket"] == label]["speed"].dropna()
        if not subset.empty:
            plot_data.append(subset.values)
            plot_labels.append(label)

    if plot_data:
        vp = ax.violinplot(plot_data, positions=range(len(plot_data)), showmedians=True)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(ACCENT[i % len(ACCENT)])
            body.set_alpha(0.7)
        
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_labels, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No hourly data available", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Speed Violin by Time of Day", fontsize=11)
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, axis="y")

    savefig("02_kinematics.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3 — Spatial / Map View
# ═══════════════════════════════════════════════════════════════
def fig_spatial(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Spatial Analysis (XY Plane)", fontsize=16, fontweight="bold")
    fig.subplots_adjust(wspace=0.35)

    sample = df.sample(min(15000, len(df)), random_state=7)

    # 3A  XY positions coloured by congestion
    ax = axes[0]
    for lbl in sorted(sample["congestion_label"].unique()):
        sub = sample[sample["congestion_label"] == lbl]
        ax.scatter(sub["x"], sub["y"], s=2, alpha=0.4,
                   color=CONG_MAP.get(lbl, "#58a6ff"),
                   label=CONG_LBL.get(lbl, str(lbl)), rasterized=True)
    ax.set_title("Vehicle Positions\nColoured by Congestion", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize=8, markerscale=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    # 3B  Speed heat-map on XY grid
    ax = axes[1]
    hb = ax.hexbin(sample["x"], sample["y"], C=sample["speed"],
                   gridsize=60, cmap="RdYlGn", reduce_C_function=np.mean,
                   linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("Mean Speed (m/s)", fontsize=9)
    ax.set_title("Speed Heat-map\n(Hex Bins)", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    # 3C  Density heat-map on XY grid
    ax = axes[2]
    hb2 = ax.hexbin(sample["x"], sample["y"], gridsize=60,
                    cmap="plasma", linewidths=0.2)
    cb2 = fig.colorbar(hb2, ax=ax, shrink=0.8)
    cb2.set_label("Point Count", fontsize=9)
    ax.set_title("Point Density\n(Hex Bins)", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    savefig("03_spatial.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 4 — Time-series per Session  (top N sessions)
# ═══════════════════════════════════════════════════════════════
def fig_timeseries(df: pd.DataFrame, n_sessions: int = 5):
    top_sessions = (df["session_id"].value_counts().head(n_sessions).index
                    if "session_id" in df.columns else [])
    if len(top_sessions) == 0:
        print("[SKIP] fig_timeseries — no session_id column.")
        return

    fig, axes = plt.subplots(n_sessions, 2, figsize=(18, 4 * n_sessions))
    fig.suptitle(f"Time-series: Top {n_sessions} Sessions",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    for i, sess in enumerate(top_sessions):
        sub = df[df["session_id"] == sess].sort_values("tick")
        mean_speed = sub.groupby("tick")["speed"].mean()
        mean_acc   = sub.groupby("tick")["acceleration"].mean()

        ax_s = axes[i, 0]
        ax_s.plot(mean_speed.index, mean_speed.values,
                  color=ACCENT[i % len(ACCENT)], linewidth=1.2)
        ax_s.fill_between(mean_speed.index, mean_speed.values,
                          alpha=0.12, color=ACCENT[i % len(ACCENT)])
        ax_s.set_title(f"Session {sess} — Mean Speed", fontsize=10)
        ax_s.set_xlabel("Tick")
        ax_s.set_ylabel("Speed (m/s)")
        ax_s.grid(True)

        ax_a = axes[i, 1]
        ax_a.plot(mean_acc.index, mean_acc.values,
                  color=ACCENT[(i + 2) % len(ACCENT)], linewidth=1.2)
        ax_a.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_a.fill_between(mean_acc.index, mean_acc.values, 0,
                          where=(mean_acc.values >= 0),
                          alpha=0.12, color="#3fb950")
        ax_a.fill_between(mean_acc.index, mean_acc.values, 0,
                          where=(mean_acc.values < 0),
                          alpha=0.12, color="#f78166")
        ax_a.set_title(f"Session {sess} — Mean Acceleration", fontsize=10)
        ax_a.set_xlabel("Tick")
        ax_a.set_ylabel("Acceleration (m/s²)")
        ax_a.grid(True)

    savefig("04_timeseries_sessions.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 5 — Weather Analysis
# ═══════════════════════════════════════════════════════════════
def fig_weather(df: pd.DataFrame):
    if "weather_name" not in df.columns:
        print("[SKIP] fig_weather — no weather_name column.")
        return

    weather_order = df["weather_name"].value_counts().index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Weather Impact Analysis", fontsize=16, fontweight="bold")
    fig.subplots_adjust(wspace=0.38)

    # 5A  Weather count pie
    ax = axes[0]
    w_counts = df["weather_name"].value_counts()
    wedges, texts, autotexts = ax.pie(
        w_counts.values,
        labels=w_counts.index,
        autopct="%1.1f%%",
        colors=ACCENT[:len(w_counts)],
        startangle=140,
        pctdistance=0.82,
        textprops={"fontsize": 8}
    )
    for at in autotexts:
        at.set_color("white")
    ax.set_title("Weather Condition Share", fontsize=11)

    # 5B  Mean speed per weather
    ax = axes[1]
    mean_spd = df.groupby("weather_name")["speed"].mean().reindex(weather_order)
    bars = ax.bar(mean_spd.index, mean_spd.values,
                  color=ACCENT[:len(mean_spd)], edgecolor="#0e1117")
    ax.set_title("Mean Speed by Weather", fontsize=11)
    ax.set_ylabel("Mean Speed (m/s)")
    ax.set_xticklabels(mean_spd.index, rotation=30, ha="right", fontsize=8)
    ax.grid(True, axis="y")

    # 5C  Congestion label stacked bar per weather
    ax = axes[2]
    ct = (df.groupby(["weather_name", "congestion_label"])
            .size().unstack(fill_value=0).reindex(weather_order))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(ct_pct))
    for lbl in sorted(ct_pct.columns):
        ax.bar(ct_pct.index, ct_pct[lbl], bottom=bottom,
               color=CONG_MAP.get(lbl, "#58a6ff"),
               label=CONG_LBL.get(lbl, str(lbl)),
               edgecolor="#0e1117", linewidth=0.5)
        bottom += ct_pct[lbl].values
    ax.set_title("Congestion Mix by Weather (%)", fontsize=11)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ct_pct.index, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y")

    savefig("05_weather.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 6 — Town / Map Comparison
# ═══════════════════════════════════════════════════════════════
def fig_town(df: pd.DataFrame):
    if "town" not in df.columns:
        print("[SKIP] fig_town — no town column.")
        return

    towns = df["town"].value_counts().index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Town / Map Comparison", fontsize=16, fontweight="bold")
    fig.subplots_adjust(wspace=0.38)

    # 6A  Record count per town
    ax = axes[0]
    tc = df["town"].value_counts()
    ax.barh(tc.index, tc.values, color=ACCENT[:len(tc)], edgecolor="#0e1117")
    ax.set_title("Data Volume per Town", fontsize=11)
    ax.set_xlabel("Row Count")
    ax.grid(True, axis="x")

    # 6B  Speed box per town
    ax = axes[1]
    town_groups = [df[df["town"] == t]["speed"].dropna() for t in towns]
    bp = ax.boxplot(town_groups, patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5))
    for patch, col in zip(bp["boxes"], ACCENT):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xticklabels(towns, rotation=30, ha="right", fontsize=8)
    ax.set_title("Speed Distribution per Town", fontsize=11)
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, axis="y")

    # 6C  Congestion fraction per town
    ax = axes[2]
    ct = (df.groupby(["town", "congestion_label"])
            .size().unstack(fill_value=0).reindex(towns))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(ct_pct))
    for lbl in sorted(ct_pct.columns):
        ax.bar(ct_pct.index, ct_pct[lbl], bottom=bottom,
               color=CONG_MAP.get(lbl, "#58a6ff"),
               label=CONG_LBL.get(lbl, str(lbl)),
               edgecolor="#0e1117", linewidth=0.5)
        bottom += ct_pct[lbl].values
    ax.set_xticklabels(ct_pct.index, rotation=30, ha="right", fontsize=8)
    ax.set_title("Congestion Mix per Town (%)", fontsize=11)
    ax.set_ylabel("Percentage (%)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y")

    savefig("06_town.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 7 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════
def fig_correlation(df: pd.DataFrame):
    num_cols = ["speed", "acceleration", "x", "y", "z", "yaw",
                "traffic_density", "hour", "congestion_label"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("Feature Correlation Matrix", fontsize=15, fontweight="bold")

    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    n = len(num_cols)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(num_cols, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(val) > 0.5 else "#c9d1d9")

    savefig("07_correlation.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 8 — Hourly Congestion Heatmap
# ═══════════════════════════════════════════════════════════════
def fig_hourly_heatmap(df: pd.DataFrame):
    if "weather_name" not in df.columns:
        print("[SKIP] fig_hourly_heatmap — no weather_name.")
        return

    pivot = (df.groupby(["weather_name", "hour"])["congestion_label"]
               .mean().unstack(fill_value=np.nan))

    fig, ax = plt.subplots(figsize=(16, max(5, len(pivot) * 0.7 + 2)))
    fig.suptitle("Mean Congestion Level: Weather × Hour of Day",
                 fontsize=14, fontweight="bold")

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=2, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="Mean Congestion Label (0=Free, 2=Congested)")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Hour of Day")

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        fontsize=6.5, color="black" if val > 1 else "gray")

    savefig("08_hourly_heatmap.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 9 — Per-vehicle Analysis  (top N vehicles)
# ═══════════════════════════════════════════════════════════════
def fig_per_vehicle(df: pd.DataFrame, n: int = 8):
    if "vehicle_id" not in df.columns:
        print("[SKIP] fig_per_vehicle — no vehicle_id column.")
        return

    top_veh = df["vehicle_id"].value_counts().head(n).index
    sub = df[df["vehicle_id"].isin(top_veh)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Per-Vehicle Statistics (Top {n} Vehicles)",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(wspace=0.35)

    # 9A  Mean speed per vehicle
    ax = axes[0]
    means = sub.groupby("vehicle_id")["speed"].mean().reindex(top_veh)
    stds  = sub.groupby("vehicle_id")["speed"].std().reindex(top_veh)
    y_pos = range(len(means))
    ax.barh(y_pos, means.values, xerr=stds.values, capsize=4,
            color=ACCENT[:len(means)], edgecolor="#0e1117", ecolor="#8b949e")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(means.index.astype(str), fontsize=8)
    ax.set_title("Mean Speed ± Std per Vehicle", fontsize=11)
    ax.set_xlabel("Speed (m/s)")
    ax.grid(True, axis="x")

    # 9B  Congestion time share per vehicle
    ax = axes[1]
    ct = (sub.groupby(["vehicle_id", "congestion_label"])
             .size().unstack(fill_value=0).reindex(top_veh))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(ct_pct))
    for lbl in sorted(ct_pct.columns):
        ax.barh(range(len(ct_pct)), ct_pct[lbl], left=bottom,
                color=CONG_MAP.get(lbl, "#58a6ff"),
                label=CONG_LBL.get(lbl, str(lbl)))
        bottom += ct_pct[lbl].values
    ax.set_yticks(range(len(ct_pct)))
    ax.set_yticklabels(ct_pct.index.astype(str), fontsize=8)
    ax.set_title("Congestion Time Share per Vehicle (%)", fontsize=11)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, axis="x")

    savefig("09_per_vehicle.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 10 — 3-D Scatter (Speed / Traffic Density / Congestion)
# ═══════════════════════════════════════════════════════════════
def fig_3d(df: pd.DataFrame):
    sample = df.sample(min(5000, len(df)), random_state=99)
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("3D: Speed / Acceleration / Traffic Density",
                 fontsize=14, fontweight="bold")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1d27")

    for lbl in sorted(sample["congestion_label"].unique()):
        sub = sample[sample["congestion_label"] == lbl]
        ax.scatter(sub["speed"], sub["acceleration"], sub["traffic_density"],
                   s=8, alpha=0.5, c=CONG_MAP.get(lbl, "#58a6ff"),
                   label=CONG_LBL.get(lbl, str(lbl)), depthshade=True)

    ax.set_xlabel("Speed (m/s)", labelpad=8)
    ax.set_ylabel("Acceleration (m/s²)", labelpad=8)
    ax.set_zlabel("Traffic Density", labelpad=8)
    ax.tick_params(colors="#8b949e")
    ax.legend(fontsize=9)

    savefig("10_3d_scatter.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Paths ──────────────────────────────────────────────────
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # big_data/
    ROOT_DIR   = os.path.dirname(SCRIPT_DIR)                  # CARLA_0.9.16/
    RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
    OUT_DIR    = os.path.join(SCRIPT_DIR, "visualizations")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Looking for data in : {RAW_DIR}")
    print(f"[INFO] Saving plots to     : {OUT_DIR}")

    df = load_data(RAW_DIR)

    print("[INFO] Generating figures ...")
    fig_overview(df)
    fig_kinematics(df)
    fig_spatial(df)
    fig_timeseries(df, n_sessions=5)
    fig_weather(df)
    fig_town(df)
    fig_correlation(df)
    fig_hourly_heatmap(df)
    fig_per_vehicle(df, n=8)
    fig_3d(df)

    print(f"\n[DONE] All plots saved to: {OUT_DIR}")