"""
preprocessing.py — PySpark preprocessing pipeline for CARLA traffic logs.

Reads all Parquet files from data/raw/, cleans and engineers features,
then writes stratified train/val/test splits to data/processed/.

USAGE:
    python preprocessing.py

REQUIRES:
    pip install pyspark pyarrow pandas
"""

import os
import math
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer
)
from pyspark.ml import Pipeline

# =============================================================================
# CONFIG
# =============================================================================

RAW_DIR         = "data/raw"
PROCESSED_DIR   = "data/processed"

# Acceleration clipping — values above this are collision artifacts
ACCEL_CLIP_MS2  = 50.0

# Train / val / test split ratios (must sum to 1.0)
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15

RANDOM_SEED     = 42

# Spark tuning — adjust based on your machine's RAM
SPARK_DRIVER_MEM    = "6g"
SPARK_EXECUTOR_MEM  = "6g"

# =============================================================================


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("CARLA_Traffic_Preprocessing")
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEM)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEM)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =============================================================================
# STEP 1 — LOAD
# =============================================================================

def load_raw(spark: SparkSession, raw_dir: str):
    section("STEP 1 — LOAD RAW DATA")
    pattern = os.path.join(raw_dir, "*.parquet")
    df = spark.read.parquet(pattern)
    count = df.count()
    print(f"  Loaded {count:,} rows from '{raw_dir}'")
    print(f"  Partitions: {df.rdd.getNumPartitions()}")
    df.printSchema()
    return df


# =============================================================================
# STEP 2 — CLEAN
# =============================================================================

def clean(df):
    section("STEP 2 — CLEANING")

    before = df.count()
    dropped = {}

    # 2a. Drop rows where any critical numeric field is null
    df = df.dropna(subset=[
        "speed", "acceleration", "traffic_density",
        "x", "y", "congestion_label", "weather", "hour", "town"
    ])
    dropped["null_critical_cols"] = before - df.count()

    # 2b. Drop exact duplicates on (session_id, tick, vehicle_id)
    before_dedup = df.count()
    df = df.dropDuplicates(["session_id", "tick", "vehicle_id"])
    dropped["duplicates"] = before_dedup - df.count()

    # 2c. Drop acceleration artifacts — values above threshold are collision
    #     physics glitches. The entire tick is unreliable so we drop the row,
    #     not just clip the value.
    #     Your data had max=263.61 m/s² (~27g) which only occurs on collision frames.
    bad_accel = df.filter(
        (F.col("acceleration") < 0) | (F.col("acceleration") > ACCEL_CLIP_MS2)
    ).count()
    df = df.filter(
        (F.col("acceleration") >= 0) & (F.col("acceleration") <= ACCEL_CLIP_MS2)
    )
    dropped["bad_acceleration"] = bad_accel

    # 2d. Drop rows with physically impossible speed
    bad_speed = df.filter(
        (F.col("speed") < 0) | (F.col("speed") > 200)
    ).count()
    df = df.filter(
        (F.col("speed") >= 0) & (F.col("speed") <= 200)
    )
    dropped["bad_speed"] = bad_speed

    # 2e. Drop rows with impossible density (negative)
    bad_density = df.filter(F.col("traffic_density") < 0).count()
    df = df.filter(F.col("traffic_density") >= 0)
    dropped["bad_density"] = bad_density

    # 2f. Drop rows with invalid label
    bad_label = df.filter(
        ~F.col("congestion_label").isin(["Low", "Medium", "High"])
    ).count()
    df = df.filter(F.col("congestion_label").isin(["Low", "Medium", "High"]))
    dropped["bad_label"] = bad_label

    # --- Summary ---
    after = df.count()
    total_dropped = before - after
    print(f"\n  {'Reason':<30} {'Rows dropped':>12}")
    print(f"  {'-'*44}")
    for reason, cnt in dropped.items():
        flag = "  <-- " if cnt > 0 else ""
        print(f"  {reason:<30} {cnt:>12,}{flag}")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL dropped':<30} {total_dropped:>12,}  ({100*total_dropped/before:.3f}%)")
    print(f"  {'Rows remaining':<30} {after:>12,}")
    return df


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    section("STEP 3 — FEATURE ENGINEERING")

    # --- Binned speed (ordinal) ---
    # 0=stationary, 1=slow, 2=moderate, 3=fast
    df = df.withColumn(
        "speed_bin",
        F.when(F.col("speed") < 5,  0)
         .when(F.col("speed") < 20, 1)
         .when(F.col("speed") < 50, 2)
         .otherwise(3)
        .cast(IntegerType())
    )

    # --- Binned density (ordinal) ---
    # 0=sparse, 1=moderate, 2=dense, 3=gridlock
    df = df.withColumn(
        "density_bin",
        F.when(F.col("traffic_density") < 3,  0)
         .when(F.col("traffic_density") < 8,  1)
         .when(F.col("traffic_density") < 15, 2)
         .otherwise(3)
        .cast(IntegerType())
    )

    # --- Is vehicle stationary? (strong congestion signal) ---
    df = df.withColumn(
        "is_stationary",
        (F.col("speed") < 1.0).cast(IntegerType())
    )

    # --- Is vehicle in high-density area? ---
    df = df.withColumn(
        "is_high_density",
        (F.col("traffic_density") >= 10).cast(IntegerType())
    )

    # --- Rush hour flag (morning: 7-9, evening: 17-19) ---
    df = df.withColumn(
        "is_rush_hour",
        (
            ((F.col("hour") >= 7)  & (F.col("hour") <= 9)) |
            ((F.col("hour") >= 17) & (F.col("hour") <= 19))
        ).cast(IntegerType())
    )

    # --- Is rainy weather? (binary, derived from weather code) ---
    # Codes 2-6 (Wet*, MidRainy*, HardRain*, SoftRain* Noon variants) are wet
    rainy_codes = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20]
    df = df.withColumn(
        "is_rainy",
        F.col("weather").isin(rainy_codes).cast(IntegerType())
    )

    # --- Speed-density interaction term ---
    # Captures the joint effect; low speed + high density = strong congestion signal
    df = df.withColumn(
        "speed_x_density",
        (F.col("speed") * F.col("traffic_density")).cast(DoubleType())
    )

    # --- Acceleration sign: braking (-) vs accelerating (+) ---
    # CARLA gives magnitude only; we use a simple proxy: low speed + high accel = braking
    df = df.withColumn(
        "is_braking",
        ((F.col("acceleration") > 1.0) & (F.col("speed") < 15.0)).cast(IntegerType())
    )

    # --- Town encoded as integer (for models that can use it) ---
    # Town01=1, Town02=2, Town03=3, Town04=4
    df = df.withColumn(
        "town_code",
        F.regexp_extract(F.col("town"), r"(\d+)$", 1).cast(IntegerType())
    )

    print("  Engineered features:")
    new_cols = [
        "speed_bin", "density_bin", "is_stationary", "is_high_density",
        "is_rush_hour", "is_rainy", "speed_x_density", "is_braking", "town_code"
    ]
    for c in new_cols:
        print(f"    + {c}")

    return df


# =============================================================================
# STEP 4 — LABEL ENCODING
# =============================================================================

def encode_labels(df):
    section("STEP 4 — LABEL ENCODING")

    # Map congestion_label -> numeric index for ML
    # Low=0, Medium=1, High=2  (natural ordinal order)
    df = df.withColumn(
        "label",
        F.when(F.col("congestion_label") == "Low",    0)
         .when(F.col("congestion_label") == "Medium", 1)
         .when(F.col("congestion_label") == "High",   2)
         .otherwise(-1)
        .cast(IntegerType())
    )

    # Sanity check — no -1 labels
    bad_labels = df.filter(F.col("label") == -1).count()
    if bad_labels > 0:
        print(f"  [WARN] {bad_labels:,} rows with unrecognised congestion_label — dropping.")
        df = df.filter(F.col("label") != -1)

    print("  Label mapping: Low=0, Medium=1, High=2")
    df.groupBy("congestion_label", "label").count().orderBy("label").show()

    return df


# =============================================================================
# STEP 5 — SUMMARY STATS (for reference / normalisation constants)
# =============================================================================

def compute_stats(df):
    section("STEP 5 — SUMMARY STATISTICS")

    numeric_cols = [
        "speed", "acceleration", "traffic_density",
        "speed_x_density", "x", "y", "yaw"
    ]
    stats = df.select(numeric_cols).summary("mean", "stddev", "min", "max")
    stats.show(truncate=False)
    return stats


# =============================================================================
# STEP 6 — TRAIN / VAL / TEST SPLIT
# =============================================================================

def split_data(df):
    section("STEP 6 — STRATIFIED TRAIN / VAL / TEST SPLIT")

    # Stratify by label to preserve class balance in each split
    splits = {"train": [], "val": [], "test": []}

    for label_val in [0, 1, 2]:
        subset = df.filter(F.col("label") == label_val)
        # First split off test, then split remainder into train/val
        test_frac  = TEST_RATIO
        val_frac   = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        rest_frac  = 1.0 - test_frac

        rest, test = subset.randomSplit([rest_frac, test_frac], seed=RANDOM_SEED)
        train, val = rest.randomSplit([1.0 - val_frac, val_frac], seed=RANDOM_SEED)

        splits["train"].append(train)
        splits["val"].append(val)
        splits["test"].append(test)

    from functools import reduce
    from pyspark.sql import DataFrame

    def union_all(dfs):
        return reduce(DataFrame.union, dfs)

    train_df = union_all(splits["train"]).orderBy(F.rand(seed=RANDOM_SEED))
    val_df   = union_all(splits["val"]).orderBy(F.rand(seed=RANDOM_SEED))
    test_df  = union_all(splits["test"]).orderBy(F.rand(seed=RANDOM_SEED))

    total = df.count()
    t_cnt = train_df.count()
    v_cnt = val_df.count()
    te_cnt = test_df.count()

    print(f"  Train : {t_cnt:>8,}  ({100*t_cnt/total:.1f}%)")
    print(f"  Val   : {v_cnt:>8,}  ({100*v_cnt/total:.1f}%)")
    print(f"  Test  : {te_cnt:>8,}  ({100*te_cnt/total:.1f}%)")

    print("\n  Label distribution per split:")
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = sdf.groupBy("label").count().orderBy("label").collect()
        dist_str = "  ".join([f"label={r['label']}: {r['count']:,}" for r in dist])
        print(f"    {name:<6}: {dist_str}")

    return train_df, val_df, test_df


# =============================================================================
# STEP 7 — SAVE
# =============================================================================

# Final feature columns that will be used for ML training
ML_FEATURE_COLS = [
    "speed", "acceleration", "traffic_density",
    "speed_bin", "density_bin",
    "is_stationary", "is_high_density",
    "is_rush_hour", "is_rainy",
    "speed_x_density", "is_braking",
    "weather", "hour", "town_code",
    "yaw",
]

# Columns to keep in the saved splits (features + label + identifiers for debugging)
KEEP_COLS = ML_FEATURE_COLS + [
    "label", "congestion_label",
    "session_id", "tick", "vehicle_id",
    "x", "y",
]


def save_splits(train_df, val_df, test_df, out_dir: str):
    section("STEP 7 — SAVING PROCESSED SPLITS")
    os.makedirs(out_dir, exist_ok=True)

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(out_dir, name)
        sdf.select(KEEP_COLS) \
           .coalesce(4) \
           .write.mode("overwrite") \
           .parquet(out_path)
        print(f"  Saved {name:<6} → {out_path}/")

    # Also save the feature column list for reference during training
    feature_list_path = os.path.join(out_dir, "feature_cols.txt")
    with open(feature_list_path, "w") as f:
        f.write("\n".join(ML_FEATURE_COLS))
    print(f"  Saved feature list → {feature_list_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\nCARLA Traffic — PySpark Preprocessing Pipeline")
    print(f"Raw dir      : {RAW_DIR}")
    print(f"Processed dir: {PROCESSED_DIR}")
    print(f"Split        : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")

    t0 = time.time()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = load_raw(spark, RAW_DIR)
    df = clean(df)
    df = engineer_features(df)
    df = encode_labels(df)

    compute_stats(df)

    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df, PROCESSED_DIR)

    elapsed = time.time() - t0
    section("DONE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Processed data → {PROCESSED_DIR}/")
    print(f"  Next step     : run train.py\n")

    spark.stop()


if __name__ == "__main__":
    main()