"""
validate_data.py — Data quality checker for CARLA traffic simulation logs.

Reads all CSV and Parquet files from data/raw/, runs quality checks,
prints a report, and saves plots to data/validation/.

USAGE:
    python validate_data.py
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CONFIG
# =============================================================================

RAW_DIR        = "data/raw"
OUTPUT_DIR     = "data/validation"
PLOT_FILE      = os.path.join(OUTPUT_DIR, "data_quality_report.png")

SPEED_MAX_KMH       = 200.0
ACCEL_MAX_MS2       = 50.0
DENSITY_MAX         = 199
MIN_ROWS_PER_SESSION = 500

# =============================================================================


def load_all(raw_dir: str) -> pd.DataFrame:
    frames = []

    parquet_files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    loaded_stems  = set()
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            df["source_file"] = os.path.basename(f)
            frames.append(df)
            loaded_stems.add(os.path.splitext(os.path.basename(f))[0])
            print(f"  [OK] {os.path.basename(f)}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  [ERR] {os.path.basename(f)}: {e}")

    for f in glob.glob(os.path.join(raw_dir, "*.csv")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in loaded_stems:
            continue
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            frames.append(df)
            print(f"  [OK] {os.path.basename(f)}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  [ERR] {os.path.basename(f)}: {e}")

    if not frames:
        raise FileNotFoundError(f"No CSV or Parquet files found in '{raw_dir}'")

    return pd.concat(frames, ignore_index=True)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_schema(df: pd.DataFrame):
    section("1. SCHEMA CHECK")
    expected = [
        "session_id", "tick", "vehicle_id", "speed", "acceleration",
        "x", "y", "z", "yaw", "traffic_density", "weather",
        "weather_name", "town", "hour", "congestion_label"
    ]
    missing = [c for c in expected if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected + ["source_file"]]
    print(f"  {'[OK]  ' if not missing else '[FAIL]'} Expected columns: "
          f"{'all present' if not missing else 'missing ' + str(missing)}")
    if extra:
        print(f"  [INFO] Extra columns: {extra}")
    print(f"  [INFO] Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")


def check_volume(df: pd.DataFrame):
    section("2. VOLUME & SESSION BREAKDOWN")
    print(f"  Total rows    : {len(df):,}")
    print(f"  Total sessions: {df['session_id'].nunique()}")
    print(f"\n  Rows per session:")
    for sid, cnt in df.groupby("session_id").size().sort_values(ascending=False).items():
        flag = "  <-- WARN: too few rows" if cnt < MIN_ROWS_PER_SESSION else ""
        print(f"    {str(sid)[-45:]:<45}  {cnt:>8,}{flag}")

    if "town" in df.columns:
        print(f"\n  Rows by town:")
        for t, c in df["town"].value_counts().items():
            print(f"    {t:<20} {c:>8,}")

    if "weather_name" in df.columns:
        print(f"\n  Rows by weather:")
        for w, c in df["weather_name"].value_counts().items():
            print(f"    {w:<25} {c:>8,}")


def check_missing(df: pd.DataFrame):
    section("3. MISSING VALUES")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  [OK]   No missing values.")
    else:
        for col, cnt in nulls.items():
            print(f"  [WARN] {col:<25} {cnt:>7,}  ({100*cnt/len(df):.2f}%)")


def check_duplicates(df: pd.DataFrame):
    section("4. DUPLICATE ROWS")
    keys = [c for c in ["session_id", "tick", "vehicle_id"] if c in df.columns]
    dupes = df.duplicated(subset=keys).sum()
    status = "[OK]  " if dupes == 0 else "[WARN]"
    print(f"  {status} Duplicate (session, tick, vehicle) rows: {dupes:,}")


def check_value_ranges(df: pd.DataFrame):
    section("5. VALUE RANGE CHECKS")
    checks = [
        ("speed",           0, SPEED_MAX_KMH,  "km/h"),
        ("acceleration",    0, ACCEL_MAX_MS2,  "m/s²"),
        ("traffic_density", 0, DENSITY_MAX,    "vehicles"),
        ("hour",            0, 23,             ""),
    ]
    for col, lo, hi, unit in checks:
        if col not in df.columns:
            continue
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        status = "[OK]  " if bad == 0 else "[WARN]"
        print(f"  {status} {col:<22} min={df[col].min():>8.2f}  "
              f"max={df[col].max():>8.2f}  mean={df[col].mean():>7.2f}  "
              f"out-of-range={bad:,}")

    numeric = df.select_dtypes(include=[np.number]).columns
    infs = np.isinf(df[numeric]).sum().sum()
    print(f"\n  {'[OK]  ' if infs == 0 else '[WARN]'} Inf values in numeric columns: {infs:,}")


def check_label_distribution(df: pd.DataFrame):
    section("6. CONGESTION LABEL DISTRIBUTION")
    if "congestion_label" not in df.columns:
        print("  [SKIP]")
        return
    counts = df["congestion_label"].value_counts()
    for label in ["Low", "Medium", "High"]:
        cnt = counts.get(label, 0)
        pct = 100 * cnt / len(df)
        bar = "█" * int(40 * pct / 100)
        print(f"  {label:<8} {cnt:>8,}  ({pct:5.1f}%)  {bar}")
    min_pct = 100 * counts.min() / len(df)
    print(f"\n  {'[OK]  ' if min_pct >= 5 else '[WARN]'} "
          f"Minority class: {min_pct:.1f}%  "
          f"{'(balance OK)' if min_pct >= 5 else '— consider rebalancing'}")


def check_spatial(df: pd.DataFrame):
    section("7. SPATIAL COVERAGE")
    if "x" not in df.columns:
        print("  [SKIP]")
        return
    grp_col = "town" if "town" in df.columns else None
    groups  = df.groupby(grp_col) if grp_col else [("all", df)]
    for name, grp in groups:
        xr = grp["x"].max() - grp["x"].min()
        yr = grp["y"].max() - grp["y"].min()
        warn = "  <-- WARN: vehicles may be stuck" if xr < 100 or yr < 100 else ""
        print(f"  {name}:  x range={xr:.1f}m   y range={yr:.1f}m{warn}")


def check_speed_density_correlation(df: pd.DataFrame):
    section("8. SPEED vs DENSITY CORRELATION")
    if "speed" not in df.columns or "traffic_density" not in df.columns:
        print("  [SKIP]")
        return
    corr = df["speed"].corr(df["traffic_density"])
    print(f"  Pearson r = {corr:.4f}")
    if corr < -0.05:
        print("  [OK]   Negative correlation — physically sensible.")
    elif corr > 0.1:
        print("  [WARN] Positive correlation — unexpected. Check density logic.")
    else:
        print("  [INFO] Near-zero — most vehicles may be free-flowing.")


def plot_report(df: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("CARLA Traffic Data — Quality Report", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    BLUE   = "#378ADD"
    GREEN  = "#1D9E75"
    PURPLE = "#7F77DD"
    AMBER  = "#EF9F27"
    RED    = "#E24B4A"
    CORAL  = "#D85A30"

    label_colors = {"Low": GREEN, "Medium": AMBER, "High": RED}

    # 1. Speed histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["speed"].clip(0, 130), bins=60, color=BLUE, edgecolor="none", alpha=0.85)
    ax.axvline(df["speed"].mean(), color=CORAL, linewidth=1.5, linestyle="--",
               label=f"mean={df['speed'].mean():.1f}")
    ax.set_title("Speed distribution (km/h)")
    ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # 2. Acceleration histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(df["acceleration"].clip(0, 15), bins=60, color=GREEN, edgecolor="none", alpha=0.85)
    ax.set_title("Acceleration (m/s²)")
    ax.set_xlabel("Acceleration"); ax.set_ylabel("Count")

    # 3. Density histogram
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df["traffic_density"], bins=40, color=PURPLE, edgecolor="none", alpha=0.85)
    ax.set_title("Traffic density")
    ax.set_xlabel("Vehicles within 50m"); ax.set_ylabel("Count")

    # 4. Label bar chart
    ax = fig.add_subplot(gs[1, 0])
    label_order = ["Low", "Medium", "High"]
    counts = df["congestion_label"].value_counts().reindex(label_order).fillna(0)
    bars = ax.bar(counts.index, counts.values,
                  color=[label_colors[l] for l in counts.index], edgecolor="none")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + max(counts.values)*0.01,
                f"{int(h):,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Congestion label distribution")
    ax.set_ylabel("Count")

    # 5. Speed vs density scatter
    ax = fig.add_subplot(gs[1, 1])
    sample = df.sample(min(4000, len(df)), random_state=42)
    for lbl, grp in sample.groupby("congestion_label"):
        ax.scatter(grp["traffic_density"], grp["speed"],
                   c=label_colors.get(lbl, "#888"), s=5, alpha=0.35, label=lbl)
    ax.set_title("Speed vs density")
    ax.set_xlabel("Density (vehicles/50m)"); ax.set_ylabel("Speed (km/h)")
    ax.legend(fontsize=9, markerscale=2)

    # 6. Rows per session
    ax = fig.add_subplot(gs[1, 2])
    scounts = df.groupby("session_id").size().sort_values()
    labels  = [str(s)[-30:] for s in scounts.index]
    ax.barh(labels, scounts.values, color=BLUE, edgecolor="none", alpha=0.85)
    ax.axvline(MIN_ROWS_PER_SESSION, color=RED, linewidth=1, linestyle="--",
               label=f"min={MIN_ROWS_PER_SESSION}")
    ax.set_title("Rows per session")
    ax.set_xlabel("Row count")
    ax.legend(fontsize=9)

    # 7. Weather breakdown
    ax = fig.add_subplot(gs[2, 0])
    if "weather_name" in df.columns:
        wc = df["weather_name"].value_counts()
        ax.barh(wc.index, wc.values, color=PURPLE, edgecolor="none", alpha=0.85)
        ax.set_title("Rows by weather preset")
        ax.set_xlabel("Count")
    else:
        ax.set_visible(False)

    # 8. Spatial heatmap
    ax = fig.add_subplot(gs[2, 1])
    sp = df.sample(min(6000, len(df)), random_state=0)
    h  = ax.hist2d(sp["x"], sp["y"], bins=60, cmap="Blues")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.set_title("Spatial coverage heatmap")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # 9. Avg speed over ticks (first session)
    ax = fig.add_subplot(gs[2, 2])
    first_sess = df["session_id"].iloc[0]
    ts = df[df["session_id"] == first_sess].groupby("tick")["speed"].mean()
    ax.plot(ts.index, ts.values, color=BLUE, linewidth=1)
    ax.set_title(f"Avg speed over ticks\n({str(first_sess)[-20:]})")
    ax.set_xlabel("Tick"); ax.set_ylabel("Avg speed (km/h)")

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\n  Saved → {out_path}")
    plt.close()


def summary_verdict(df: pd.DataFrame):
    section("FINAL VERDICT")
    issues = []

    if len(df) < 10_000:
        issues.append(f"Only {len(df):,} total rows — run more sessions.")
    if df.isnull().sum().sum() > 0:
        pct = 100 * df.isnull().sum().sum() / (len(df) * len(df.columns))
        issues.append(f"{pct:.2f}% null values overall.")
    if "speed" in df.columns and (df["speed"] > SPEED_MAX_KMH).any():
        issues.append(f"{(df['speed'] > SPEED_MAX_KMH).sum():,} rows exceed {SPEED_MAX_KMH} km/h.")
    if "congestion_label" in df.columns:
        min_pct = 100 * df["congestion_label"].value_counts(normalize=True).min()
        if min_pct < 5:
            issues.append(f"Label imbalance — minority class at {min_pct:.1f}%.")
    if "speed" in df.columns and "traffic_density" in df.columns:
        corr = df["speed"].corr(df["traffic_density"])
        if corr > 0.1:
            issues.append(f"Unexpected positive speed-density correlation ({corr:.3f}).")
    short = (df.groupby("session_id").size() < MIN_ROWS_PER_SESSION).sum()
    if short:
        issues.append(f"{short} session(s) below {MIN_ROWS_PER_SESSION} rows.")

    if issues:
        print("  [ACTION NEEDED]")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  [PASS] Data looks good — ready for PySpark preprocessing.")
    print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading files from '{RAW_DIR}'...")
    df = load_all(RAW_DIR)
    print(f"\nLoaded {len(df):,} total rows.\n")

    check_schema(df)
    check_volume(df)
    check_missing(df)
    check_duplicates(df)
    check_value_ranges(df)
    check_label_distribution(df)
    check_spatial(df)
    check_speed_density_correlation(df)

    print(f"\n{'='*60}")
    print("  Generating plots...")
    plot_report(df, PLOT_FILE)

    summary_verdict(df)


if __name__ == "__main__":
    main()