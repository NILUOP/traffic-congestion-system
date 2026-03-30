"""
carla_runner.py — Traffic simulation data collector for congestion analysis project.

USAGE:
    python carla_runner.py

CONFIGURE the settings block below before each run. Change town, weather,
vehicle count, and duration manually between sessions to generate diverse data.
"""

import carla
import random
import time
import math
import os
import csv
import uuid
from datetime import datetime

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("[WARN] pyarrow not found — only CSV output will be written.")

# =============================================================================
# CONFIGURATION — edit these between runs
# =============================================================================

TOWN            = "Town04"          # Town01, Town02, Town03, Town04, Town05, Town10HD
VEHICLE_COUNT   = 155                # number of NPC vehicles to spawn (50–200)
SIMULATION_FPS  = 20                # fixed simulation timestep
DURATION_SECS   = 500               # how long to run (seconds of sim time)
LOG_INTERVAL    = 5                 # log every N simulation ticks

# Weather preset — pick one or define a custom carla.WeatherParameters below
# Options: ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, MidRainyNoon,
#          HardRainNoon, SoftRainNoon, ClearSunset, CloudySunset, WetSunset,
#          WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset,
#          ClearNight, CloudyNight, WetNight, WetCloudyNight, SoftRainNight,
#          MidRainyNight, HardRainNight
WEATHER_PRESET  = "HardRainNoon"

# Simulated time-of-day context logged as a feature (0–23, does not affect lighting)
HOUR_OF_DAY     = 12

# Output paths
OUTPUT_DIR      = "data/raw"
SESSION_ID      = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{TOWN}_{WEATHER_PRESET}"

# CARLA server connection
CARLA_HOST      = "localhost"
CARLA_PORT      = 2000
TIMEOUT_SECS    = 10.0

# =============================================================================


def get_weather(preset_name: str) -> carla.WeatherParameters:
    presets = {
        attr: getattr(carla.WeatherParameters, attr)
        for attr in dir(carla.WeatherParameters)
        if not attr.startswith("_")
    }
    if preset_name not in presets:
        print(f"[WARN] Unknown weather preset '{preset_name}', using ClearNoon.")
        return carla.WeatherParameters.ClearNoon
    return presets[preset_name]


def encode_weather(preset_name: str) -> int:
    """Map weather preset name to integer code for ML features."""
    mapping = {
        "ClearNoon": 0, "CloudyNoon": 1, "WetNoon": 2, "WetCloudyNoon": 3,
        "MidRainyNoon": 4, "HardRainNoon": 5, "SoftRainNoon": 6,
        "ClearSunset": 7, "CloudySunset": 8, "WetSunset": 9,
        "WetCloudySunset": 10, "SoftRainSunset": 11, "MidRainSunset": 12,
        "HardRainSunset": 13, "ClearNight": 14, "CloudyNight": 15,
        "WetNight": 16, "WetCloudyNight": 17, "SoftRainNight": 18,
        "MidRainyNight": 19, "HardRainNight": 20,
    }
    return mapping.get(preset_name, -1)


def alive(vehicle) -> bool:
    """Return True if the actor still exists in the simulation."""
    try:
        vehicle.get_location()
        return True
    except RuntimeError:
        return False


def compute_traffic_density(vehicles, radius_m: float = 50.0) -> dict:
    """
    For each vehicle, count how many other vehicles are within `radius_m` metres.
    Returns a dict: vehicle_id -> density count.

    This is a simple O(n^2) approach — fine for ≤200 vehicles.
    For larger counts consider a spatial index.
    """
    positions = {}
    for v in vehicles:
        try:
            loc = v.get_location()
            positions[v.id] = (loc.x, loc.y)
        except RuntimeError:
            pass  # actor was destroyed between ticks, skip it

    density = {}
    for vid, (x, y) in positions.items():
        count = 0
        for other_id, (ox, oy) in positions.items():
            if other_id == vid:
                continue
            dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if dist <= radius_m:
                count += 1
        density[vid] = count
    return density


def label_congestion(density: int, speed_kmh: float) -> str:
    """
    Assign a congestion label using density + speed interaction.
    Avoids the trivial 'slow = congested' single-feature definition.

    High   — dense AND slow (genuine gridlock)
    Low    — sparse OR fast-moving
    Medium — everything in between
    """
    if density >= 10 and speed_kmh < 20.0:
        return "High"
    elif density < 4 or speed_kmh >= 50.0:
        return "Low"
    else:
        return "Medium"


def collect_vehicle_snapshot(vehicles, tick_number: int, weather_code: int) -> list:
    """
    Collect one row per vehicle per logged tick.
    Skips any actor that has been destroyed since the last tick.
    Returns a list of dicts matching the dataset schema.
    """
    live_vehicles = [v for v in vehicles if alive(v)]
    density_map   = compute_traffic_density(live_vehicles)
    rows = []

    for v in live_vehicles:
        try:
            transform = v.get_transform()
            velocity  = v.get_velocity()
            accel     = v.get_acceleration()
        except RuntimeError:
            continue  # destroyed between alive() check and here

        loc = transform.location
        speed_ms  = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6
        accel_mag = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)

        density = density_map.get(v.id, 0)

        rows.append({
            "session_id":       SESSION_ID,
            "tick":             tick_number,
            "vehicle_id":       v.id,
            "speed":            round(speed_kmh, 3),
            "acceleration":     round(accel_mag, 4),
            "x":                round(loc.x, 3),
            "y":                round(loc.y, 3),
            "z":                round(loc.z, 3),
            "yaw":              round(transform.rotation.yaw, 2),
            "traffic_density":  density,
            "weather":          weather_code,
            "weather_name":     WEATHER_PRESET,
            "town":             TOWN,
            "hour":             HOUR_OF_DAY,
            "congestion_label": label_congestion(density, speed_kmh),
        })

    return rows


def save_csv(rows: list, filepath: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_parquet(rows: list, filepath: str):
    if not rows or not PARQUET_AVAILABLE:
        return
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if os.path.exists(filepath):
        existing = pq.read_table(filepath)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, filepath, compression="snappy")
    else:
        pq.write_table(table, filepath, compression="snappy")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path     = os.path.join(OUTPUT_DIR, f"{SESSION_ID}.csv")
    parquet_path = os.path.join(OUTPUT_DIR, f"{SESSION_ID}.parquet")

    print(f"[INFO] Session   : {SESSION_ID}")
    print(f"[INFO] Town      : {TOWN}")
    print(f"[INFO] Weather   : {WEATHER_PRESET}")
    print(f"[INFO] Vehicles  : {VEHICLE_COUNT}")
    print(f"[INFO] Duration  : {DURATION_SECS}s @ {SIMULATION_FPS} FPS")
    print(f"[INFO] CSV       : {csv_path}")
    if PARQUET_AVAILABLE:
        print(f"[INFO] Parquet   : {parquet_path}")

    # --- Connect to CARLA ------------------------------------------------
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(TIMEOUT_SECS)

    print(f"\n[INFO] Loading {TOWN}...")
    world = client.load_world(TOWN)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIMULATION_FPS
    world.apply_settings(settings)

    world.set_weather(get_weather(WEATHER_PRESET))
    weather_code = encode_weather(WEATHER_PRESET)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.global_percentage_speed_difference(random.uniform(-10, 20))

    # --- Spawn vehicles --------------------------------------------------
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter("vehicle.*")

    # Exclude bikes and motorcycles for cleaner traffic data
    vehicle_bps = [
        bp for bp in vehicle_bps
        if int(bp.get_attribute("number_of_wheels")) >= 4
    ]

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    count_to_spawn = min(VEHICLE_COUNT, len(spawn_points))
    if count_to_spawn < VEHICLE_COUNT:
        print(f"[WARN] Only {len(spawn_points)} spawn points available — spawning {count_to_spawn} vehicles.")

    spawned_vehicles = []
    for i in range(count_to_spawn):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        actor = world.try_spawn_actor(bp, spawn_points[i])
        if actor:
            actor.set_autopilot(True, traffic_manager.get_port())
            spawned_vehicles.append(actor)

    print(f"[INFO] Spawned {len(spawned_vehicles)} vehicles.\n")

    # --- Warm-up ticks (let vehicles settle) -----------------------------
    WARMUP_TICKS = SIMULATION_FPS * 5  # 5 seconds
    print("[INFO] Warming up...")
    for _ in range(WARMUP_TICKS):
        world.tick()

    # --- Main data collection loop ---------------------------------------
    total_ticks  = SIMULATION_FPS * DURATION_SECS
    rows_written = 0
    tick_number  = 0
    batch        = []
    BATCH_SIZE   = 500  # flush to disk every N rows

    print("[INFO] Collecting data...")
    try:
        for tick in range(total_ticks):
            world.tick()

            if tick % LOG_INTERVAL != 0:
                continue

            spawned_vehicles = [v for v in spawned_vehicles if alive(v)]
            snapshot = collect_vehicle_snapshot(spawned_vehicles, tick_number, weather_code)
            batch.extend(snapshot)
            tick_number += 1

            if len(batch) >= BATCH_SIZE:
                save_csv(batch, csv_path)
                save_parquet(batch, parquet_path)
                rows_written += len(batch)
                batch = []

                if rows_written % 5000 == 0:
                    pct = 100 * tick / total_ticks
                    print(f"  {rows_written:>7,} rows written  ({pct:.0f}% done)")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # Flush remaining rows
        if batch:
            save_csv(batch, csv_path)
            save_parquet(batch, parquet_path)
            rows_written += len(batch)

        # Restore async mode and destroy actors
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print(f"\n[INFO] Destroying {len(spawned_vehicles)} vehicles...")
        client.apply_batch([carla.command.DestroyActor(v) for v in spawned_vehicles])

        print(f"[INFO] Done. {rows_written:,} rows written.")
        print(f"[INFO] CSV     → {csv_path}")
        if PARQUET_AVAILABLE:
            print(f"[INFO] Parquet → {parquet_path}")


if __name__ == "__main__":
    main()