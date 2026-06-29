import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import hashlib
import datetime
import requests
import copy

from data.constants import col_map, build_base_nodes_and_edges
from data.calibration import run_calibration
from data.pagasa import fetch_pagasa_advisory
from analysis.sensitivity import run_sensitivity
from engine.twin import DigitalTwin
from ui.gauges import render_pagasa_gauge, render_sim_gauge, render_waterlevel_gauge
from ui.charts import render_network_graph, render_cluster_breakdown, render_cac_bubble
from ui.insights import render_policy_insights

st.set_page_config(page_title="Tagoloan Flood-Prone Communities Digital Twin", layout="wide")

# ---------- Dark mode ----------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def apply_dark_mode():
    if st.session_state.dark_mode:
        dark_css = """<style> … </style>"""  # same as before
        st.markdown(dark_css, unsafe_allow_html=True)

apply_dark_mode()

# ---------- Session state ----------
if 'twin' not in st.session_state:
    nodes, edges = build_base_nodes_and_edges()
    st.session_state.twin = DigitalTwin(
        nodes=nodes, edges=edges, cluster_profiles={},
        total_population=0, flood_severity=0.0, lgu_threat=False,
        col_map=col_map
    )

defaults = {
    'data_calibrated': False, 'respondent_clusters': None,
    'current_barangay': "All Barangays", 'raw_data': None,
    'disable_flashing': False, 'use_pagasa_auto': True,
    'pagasa_severity': None, 'prev_pagasa_severity': None,
    'baseline_params': None, 'log_entries': [], 'auto_log': True,
    'prev_k_mode': "Auto (silhouette)", 'dark_mode': False,
    'sensitivity_results': None, 'sensitivity_param': ""
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- Sidebar ----------
with st.sidebar:
    st.toggle("🌙 Dark Mode", key="dark_mode")
    st.header("📊 Data Upload & Calibration")
    # … (same upload and recalibrate logic, using run_calibration and building a twin)
    # …
    # PAGASA gauge
    st.subheader("🌧️ PAGASA Advisory")
    advisory = fetch_pagasa_advisory()
    pagasa_severity, pagasa_label = render_pagasa_gauge(advisory, st.session_state.disable_flashing, st.session_state.disable_flashing)

    # Auto‑mode
    if st.session_state.use_pagasa_auto and pagasa_severity is not None:
        st.session_state.twin.flood_severity = pagasa_severity

    # Sim severity slider
    st.subheader("🧪 Simulated Flood Severity")
    # … manual / auto logic …
    flood_sev = st.session_state.twin.flood_severity   # simplified
    sim_label = render_sim_gauge(flood_sev)

    # Water level
    render_waterlevel_gauge()

    # LGU sliders … same as before

    # Sensitivity analysis
    st.subheader("📊 Sensitivity Analysis")
    # … same controls, calls run_sensitivity()

# ---------- Main Dashboard ----------
barangay_title = st.session_state.current_barangay if st.session_state.current_barangay != "All Barangays" else "Municipal"
st.title(f"Tagoloan Flood-Prone Communities Digital Twin ({barangay_title})")

twin = st.session_state.twin
metrics = twin.get_metrics()
advanced = twin.get_advanced_metrics()

# Official Map
st.subheader("🗺️ Tagoloan River Basin")
# … same map with download button …

# Behavioral Outcomes
st.subheader("Community Behavioral Outcomes")
# … metrics display …

# Advanced Indicators
# … display …

# Network graph, cluster breakdown, CAC bubble
st.subheader("Socio-Psychological Network Graph")
render_network_graph(twin, barangay_title)
st.subheader("Behavioral Distribution by Cluster")
cluster_df = twin.analytics.get_cluster_breakdown()
render_cluster_breakdown(cluster_df, barangay_title)
st.subheader("CAC Breakdown")
cac_avgs = twin.analytics.get_cac_averages(col_map)
render_cac_bubble(cac_avgs, col_map, barangay_title)

# Policy Insights
st.subheader("Policy Insights & Actionable Recommendations")
render_policy_insights(twin, metrics, advanced, flood_sev, pagasa_label, barangay_title)

# Simulation Timeline and logging … (same as before)
