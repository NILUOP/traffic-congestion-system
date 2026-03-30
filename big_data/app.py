"""
app.py — Streamlit dashboard for CARLA traffic congestion analysis.

Shows EDA, model results, live prediction, and congestion hotspot map.

USAGE:
    streamlit run dashboard/app.py

REQUIRES:
    pip install streamlit pandas pyarrow plotly scikit-learn xgboost pickle5
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="CARLA Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME / CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #c8d0e0;
}

/* Main background */
.stApp { background-color: #0a0e1a; }
section[data-testid="stSidebar"] { background-color: #0f1525; border-right: 1px solid #1e2d4a; }

/* Header strip */
.dash-header {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a0e1a 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 1.5rem 2rem 1rem;
    margin: -1rem -1rem 2rem -1rem;
}
.dash-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e8f0ff;
    letter-spacing: -0.5px;
    margin: 0;
}
.dash-header p {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #4a7fa5;
    margin: 0.3rem 0 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #0f1525;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.5rem;
}
.metric-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4a7fa5;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #e8f0ff;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #5a8aaa;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8f0ff;
    border-left: 3px solid #2e6db4;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
    letter-spacing: -0.2px;
}

/* Congestion badges */
.badge-low    { background:#0d3320; color:#3ddc84; border:1px solid #1a5c3a; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:0.78rem; }
.badge-medium { background:#2e2000; color:#ffb300; border:1px solid #5a4000; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:0.78rem; }
.badge-high   { background:#2e0a0a; color:#ff5252; border:1px solid #5a1a1a; padding:2px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:0.78rem; }

/* Prediction result box */
.pred-box {
    background: #0f1525;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    border: 1px solid #1e2d4a;
}
.pred-label {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.pred-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #4a7fa5;
    letter-spacing: 1px;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: #7a9aba !important; font-size: 0.8rem !important; font-family: 'DM Mono', monospace !important; letter-spacing: 0.5px; }

div.stButton > button {
    background: #1a3a6e;
    color: #c8d8f0;
    border: 1px solid #2e5aaa;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    width: 100%;
    padding: 0.6rem;
    transition: all 0.2s;
}
div.stButton > button:hover { background: #2e5aaa; color: #fff; }

.stTabs [data-baseweb="tab-list"] { background: #0f1525; border-bottom: 1px solid #1e2d4a; gap: 0; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #4a7fa5; letter-spacing: 0.5px; padding: 0.7rem 1.2rem; }
.stTabs [aria-selected="true"] { color: #e8f0ff !important; border-bottom: 2px solid #2e6db4 !important; }

hr { border-color: #1e2d4a; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PLOTLY DARK THEME
# =============================================================================

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,21,37,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#c8d0e0", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)

COLOR_LOW    = "#3ddc84"
COLOR_MED    = "#ffb300"
COLOR_HIGH   = "#ff5252"
COLOR_BLUE   = "#2e6db4"
LABEL_COLORS = {"Low": COLOR_LOW, "Medium": COLOR_MED, "High": COLOR_HIGH}

# =============================================================================
# DATA / MODEL LOADING
# =============================================================================

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "data", "models_clean")
COMPARISON_CSV = os.path.join(MODELS_DIR, "comparison.csv")
FEATURE_TXT    = os.path.join(PROCESSED_DIR, "feature_cols.txt")

LEAKAGE_FEATURES = [
    "traffic_density", "density_bin", "is_high_density", "speed_x_density"
]


@st.cache_data(show_spinner=False)
def load_raw_data():
    import glob
    files = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_test_data():
    path = os.path.join(PROCESSED_DIR, "test")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    safe = name.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{safe}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_feature_cols():
    if not os.path.exists(FEATURE_TXT):
        return []
    with open(FEATURE_TXT) as f:
        cols = [l.strip() for l in f if l.strip()]
    return [c for c in cols if c not in LEAKAGE_FEATURES]


def load_comparison():
    if not os.path.exists(COMPARISON_CSV):
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_CSV)


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="dash-header">
    <h1>🚦 CARLA Traffic Intelligence</h1>
    <p>Scalable Congestion Analysis · CARLA 0.9.16 · 1M+ Records · PySpark + ML</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)

    raw_df  = load_raw_data()
    test_df = load_test_data()

    if not raw_df.empty:
        towns    = ["All"] + sorted(raw_df["town"].unique().tolist())
        weathers = ["All"] + sorted(raw_df["weather_name"].unique().tolist())
        sel_town    = st.selectbox("Town", towns)
        sel_weather = st.selectbox("Weather", weathers)
    else:
        sel_town = sel_weather = "All"

    st.markdown("---")
    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
    sel_model = st.selectbox(
        "Active model",
        ["Random Forest", "XGBoost", "Logistic Regression"]
    )
    model = load_model(sel_model)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#2e4a6a;line-height:1.8;">
    CARLA 0.9.16<br>
    PySpark 3.x<br>
    scikit-learn · XGBoost<br>
    4 Towns · 4 Weathers<br>
    1,007,762 records
    </div>
    """, unsafe_allow_html=True)

# Apply sidebar filters
def filter_df(df):
    if df.empty:
        return df
    if sel_town != "All":
        df = df[df["town"] == sel_town]
    if sel_weather != "All" and "weather_name" in df.columns:
        df = df[df["weather_name"] == sel_weather]
    return df

fdf = filter_df(raw_df)

# =============================================================================
# TOP KPI CARDS
# =============================================================================

c1, c2, c3, c4, c5 = st.columns(5)
kpi_data = [
    (c1, "Total records",   f"{len(fdf):,}",                  "after filtering"),
    (c2, "Sessions",        f"{fdf['session_id'].nunique() if not fdf.empty else 0}", "unique runs"),
    (c3, "Avg speed",       f"{fdf['speed'].mean():.1f} km/h" if not fdf.empty else "—", "fleet average"),
    (c4, "High congestion", f"{(fdf['congestion_label']=='High').mean()*100:.1f}%" if not fdf.empty else "—", "of records"),
    (c5, "Best model F1",   "0.810",                          "Random Forest"),
]
for col, label, value, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Data Explorer",
    "🗺️  Congestion Map",
    "🤖  Model Results",
    "⚡  Live Predictor",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    if fdf.empty:
        st.warning("No data found in data/raw/. Run carla_runner.py first.")
    else:
        st.markdown('<div class="section-title">Speed distribution</div>', unsafe_allow_html=True)

        c_l, c_r = st.columns(2)

        with c_l:
            # Speed histogram by label
            fig = px.histogram(
                fdf.sample(min(50_000, len(fdf)), random_state=42),
                x="speed", color="congestion_label",
                nbins=80, barmode="overlay",
                color_discrete_map=LABEL_COLORS,
                labels={"speed": "Speed (km/h)", "congestion_label": "Congestion"},
                title="Speed distribution by congestion level",
            )
            fig.update_traces(opacity=0.72)
            fig.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with c_r:
            # Label distribution donut
            lc = fdf["congestion_label"].value_counts().reset_index()
            lc.columns = ["label", "count"]
            fig2 = px.pie(
                lc, names="label", values="count",
                hole=0.55,
                color="label",
                color_discrete_map=LABEL_COLORS,
                title="Congestion label distribution",
            )
            fig2.update_traces(textfont_size=13)
            fig2.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Speed vs traffic density</div>', unsafe_allow_html=True)

        sample = fdf.sample(min(8_000, len(fdf)), random_state=1)
        fig3 = px.scatter(
            sample,
            x="traffic_density", y="speed",
            color="congestion_label",
            color_discrete_map=LABEL_COLORS,
            opacity=0.45,
            size_max=4,
            labels={
                "traffic_density": "Traffic density (vehicles/50m)",
                "speed": "Speed (km/h)",
                "congestion_label": "Congestion",
            },
            title="Speed vs density  —  coloured by congestion label",
        )
        fig3.update_traces(marker=dict(size=3))
        fig3.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-title">Per-session summary</div>', unsafe_allow_html=True)

        sess = fdf.groupby("session_id").agg(
            rows=("speed", "count"),
            avg_speed=("speed", "mean"),
            avg_density=("traffic_density", "mean"),
            pct_high=("congestion_label", lambda x: (x == "High").mean() * 100),
        ).reset_index().round(2)
        sess.columns = ["Session", "Rows", "Avg Speed (km/h)", "Avg Density", "% High Congestion"]
        st.dataframe(sess, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CONGESTION MAP
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    if fdf.empty:
        st.warning("No data available.")
    else:
        st.markdown('<div class="section-title">Spatial congestion heatmap</div>', unsafe_allow_html=True)

        map_sample = fdf.sample(min(30_000, len(fdf)), random_state=7)

        view_mode = st.radio(
            "View mode",
            ["Congestion labels", "Speed heatmap", "Density heatmap"],
            horizontal=True,
        )

        if view_mode == "Congestion labels":
            fig_map = px.scatter(
                map_sample,
                x="x", y="y",
                color="congestion_label",
                color_discrete_map=LABEL_COLORS,
                opacity=0.5,
                size_max=3,
                labels={"x": "X (m)", "y": "Y (m)", "congestion_label": "Congestion"},
                title="Vehicle positions coloured by congestion level",
            )
            fig_map.update_traces(marker=dict(size=3))

        elif view_mode == "Speed heatmap":
            fig_map = px.density_heatmap(
                map_sample, x="x", y="y", z="speed",
                histfunc="avg", nbinsx=60, nbinsy=60,
                color_continuous_scale="RdYlGn",
                labels={"x": "X (m)", "y": "Y (m)", "speed": "Avg speed"},
                title="Average speed heatmap",
            )

        else:
            fig_map = px.density_heatmap(
                map_sample, x="x", y="y", z="traffic_density",
                histfunc="avg", nbinsx=60, nbinsy=60,
                color_continuous_scale="YlOrRd",
                labels={"x": "X (m)", "y": "Y (m)", "traffic_density": "Avg density"},
                title="Traffic density heatmap",
            )

        fig_map.update_layout(**PLOTLY_THEME, height=520)
        st.plotly_chart(fig_map, use_container_width=True)

        # Hotspot table — grid cells with highest density
        st.markdown('<div class="section-title">Top congestion hotspots</div>', unsafe_allow_html=True)
        fdf_copy = fdf.copy()
        fdf_copy["x_bin"] = (fdf_copy["x"] // 20 * 20).astype(int)
        fdf_copy["y_bin"] = (fdf_copy["y"] // 20 * 20).astype(int)
        hotspots = (
            fdf_copy.groupby(["x_bin", "y_bin"])
            .agg(
                records=("speed", "count"),
                avg_speed=("speed", "mean"),
                avg_density=("traffic_density", "mean"),
                pct_high=("congestion_label", lambda x: (x == "High").mean() * 100),
            )
            .reset_index()
            .sort_values("pct_high", ascending=False)
            .head(10)
            .round(1)
        )
        hotspots.columns = ["X grid", "Y grid", "Records", "Avg Speed (km/h)", "Avg Density", "% High"]
        st.dataframe(hotspots, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    comp_df = load_comparison()

    if comp_df.empty:
        st.warning("No comparison.csv found. Run train.py first.")
    else:
        st.markdown('<div class="section-title">Model comparison — test set</div>', unsafe_allow_html=True)

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        fig_comp = go.Figure()
        bar_colors = {"Logistic Regression": "#7566cc", "Random Forest": "#2ecc9a", "XGBoost": "#e86a3a"}

        for _, row in comp_df.iterrows():
            fig_comp.add_trace(go.Bar(
                name=row["model"],
                x=[m.upper() for m in metrics],
                y=[row[m] for m in metrics],
                marker_color=bar_colors.get(row["model"], COLOR_BLUE),
                text=[f"{row[m]:.3f}" for m in metrics],
                textposition="outside",
                textfont=dict(size=11),
            ))

        fig_comp.add_hline(y=0.8, line_dash="dot", line_color="#444", annotation_text="0.8 target")
        fig_comp.update_layout(
            **PLOTLY_THEME,
            barmode="group",
            yaxis=dict(range=[0, 1.12]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Metrics table
        display = comp_df.copy()
        display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            display[col] = display[col].map("{:.4f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Feature importance — leakage-free features</div>', unsafe_allow_html=True)

        rf  = load_model("Random Forest")
        xgb = load_model("XGBoost")
        feature_cols = load_feature_cols()

        if rf and feature_cols:
            c_imp_l, c_imp_r = st.columns(2)
            for col, mdl, title in [
                (c_imp_l, rf,  "Random Forest"),
                (c_imp_r, xgb, "XGBoost"),
            ]:
                if mdl is None:
                    continue
                imp = mdl.feature_importances_
                idx = np.argsort(imp)
                fig_imp = go.Figure(go.Bar(
                    x=imp[idx],
                    y=[feature_cols[i] for i in idx],
                    orientation="h",
                    marker=dict(
                        color=imp[idx],
                        colorscale=[[0, "#1a3a6e"], [1, "#2e8fd4"]],
                        showscale=False,
                    ),
                ))
                fig_imp.update_layout(
                    **PLOTLY_THEME,
                    title=title,
                    height=360,
                    xaxis_title="Importance",
                )
                with col:
                    st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — LIVE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown('<div class="section-title">Predict congestion from vehicle state</div>', unsafe_allow_html=True)

    feature_cols = load_feature_cols()

    if not feature_cols:
        st.warning("feature_cols.txt not found. Run preprocessing.py first.")
    elif model is None:
        st.warning(f"Model '{sel_model}' not found in {MODELS_DIR}. Run train.py first.")
    else:
        st.markdown(f"Using **{sel_model}** · {len(feature_cols)} features (leakage-free)")

        WEATHER_MAP = {
            "ClearNoon": 0, "CloudyNoon": 1, "WetNoon": 2,
            "WetCloudyNoon": 3, "MidRainyNoon": 4, "HardRainNoon": 5,
            "SoftRainNoon": 6, "ClearSunset": 7, "HardRainSunset": 13,
            "ClearNight": 14, "HardRainNight": 20,
        }
        TOWN_MAP = {"Town01": 1, "Town02": 2, "Town03": 3, "Town04": 4}

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Vehicle dynamics**")
            speed       = st.slider("Speed (km/h)",       0.0, 120.0, 25.0, 0.5)
            acceleration = st.slider("Acceleration (m/s²)", 0.0, 15.0, 1.2, 0.1)
            yaw         = st.slider("Yaw (°)",            -180.0, 180.0, 0.0, 1.0)

        with col_b:
            st.markdown("**Environment**")
            weather_name = st.selectbox("Weather", list(WEATHER_MAP.keys()))
            town_name    = st.selectbox("Town", list(TOWN_MAP.keys()))
            hour         = st.slider("Hour of day", 0, 23, 12)

        with col_c:
            st.markdown("**Derived flags**")
            st.info(
                "These are computed automatically from your inputs.",
                icon="ℹ️"
            )
            speed_bin      = int(speed < 5) * 0 + int(5 <= speed < 20) * 1 + int(20 <= speed < 50) * 2 + int(speed >= 50) * 3
            is_stationary  = int(speed < 1.0)
            is_rush_hour   = int((7 <= hour <= 9) or (17 <= hour <= 19))
            weather_code   = WEATHER_MAP[weather_name]
            is_rainy       = int(weather_code in [2,3,4,5,6,9,10,11,12,13,16,17,18,19,20])
            is_braking     = int(acceleration > 1.0 and speed < 15.0)
            town_code      = TOWN_MAP[town_name]

            st.markdown(f"""
            | Flag | Value |
            |---|---|
            | speed_bin | `{speed_bin}` |
            | is_stationary | `{is_stationary}` |
            | is_rush_hour | `{is_rush_hour}` |
            | is_rainy | `{is_rainy}` |
            | is_braking | `{is_braking}` |
            """)

        st.markdown("---")

        if st.button("▶  PREDICT CONGESTION LEVEL"):
            feature_values = {
                "speed":         speed,
                "acceleration":  acceleration,
                "speed_bin":     speed_bin,
                "is_stationary": is_stationary,
                "is_rush_hour":  is_rush_hour,
                "is_rainy":      is_rainy,
                "is_braking":    is_braking,
                "weather":       weather_code,
                "hour":          hour,
                "town_code":     town_code,
                "yaw":           yaw,
            }
            X_input = np.array([[feature_values[f] for f in feature_cols]])

            proba  = model.predict_proba(X_input)[0]
            pred   = int(np.argmax(proba))
            labels = ["Low", "Medium", "High"]
            colors = [COLOR_LOW, COLOR_MED, COLOR_HIGH]
            icons  = ["🟢", "🟡", "🔴"]

            pred_label = labels[pred]
            pred_color = colors[pred]
            pred_conf  = proba[pred] * 100

            res_l, res_r = st.columns([1, 2])

            with res_l:
                st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-conf">PREDICTED CONGESTION</div>
                    <div class="pred-label" style="color:{pred_color}">{icons[pred]} {pred_label}</div>
                    <div class="pred-conf">CONFIDENCE: {pred_conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with res_r:
                fig_proba = go.Figure(go.Bar(
                    x=labels,
                    y=proba * 100,
                    marker_color=[COLOR_LOW, COLOR_MED, COLOR_HIGH],
                    text=[f"{p*100:.1f}%" for p in proba],
                    textposition="outside",
                ))
                fig_proba.update_layout(
                    **PLOTLY_THEME,
                    title="Class probabilities",
                    yaxis=dict(range=[0, 115], title="Probability (%)"),
                    height=260,
                )
                st.plotly_chart(fig_proba, use_container_width=True)

            # Interpretation
            st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
            reasons = []
            if is_stationary:
                reasons.append("🔴 Vehicle is stationary — strong congestion signal")
            if speed < 15:
                reasons.append(f"🟡 Speed is low ({speed:.0f} km/h)")
            if is_braking:
                reasons.append("🟡 High deceleration at low speed suggests stop-and-go traffic")
            if is_rush_hour:
                reasons.append("🟡 Rush hour window (7–9am or 5–7pm)")
            if is_rainy:
                reasons.append("🔵 Wet weather reduces vehicle speeds")
            if speed >= 50:
                reasons.append("🟢 High speed indicates free-flow conditions")
            if not reasons:
                reasons.append("🔵 No strong signals — prediction based on combined feature pattern")

            for r in reasons:
                st.markdown(f"- {r}")