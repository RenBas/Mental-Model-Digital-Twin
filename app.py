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
import networkx as nx

from data.constants import col_map, build_base_nodes_and_edges
from data.calibration import run_calibration
from data.pagasa import fetch_pagasa_advisory
from analysis.sensitivity import run_sensitivity
from engine.twin import DigitalTwin
from ui.gauges import render_pagasa_gauge, render_sim_gauge, render_waterlevel_gauge
from ui.charts import render_network_graph, render_cluster_breakdown, render_cac_bubble
from ui.insights import render_policy_insights

st.set_page_config(
    page_title="Tagoloan Flood-Prone Communities Digital Twin",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def apply_dark_mode():
    if st.session_state.dark_mode:
        dark_css = """
        <style>
            .stApp, .main, .stSidebar, .st-expander, .stMetric, .stDataFrame {
                background-color: #1E1E1E; color: #E0E0E0;
            }
            div[data-testid="metric-container"] {
                background-color: #2C2C2C; border: 1px solid #444; border-radius: 8px;
            }
            div[data-testid="metric-container"] > label { color: #E0E0E0 !important; }
            .stSidebar { background-color: #252525; }
            .streamlit-expanderHeader { color: #E0E0E0; }
            .stButton>button { background-color: #3A3A3A; color: #E0E0E0; border: 1px solid #555; }
            .stSlider>div>div>div>div { color: #E0E0E0; }
            .caption, .stCaption { color: #B0B0B0; }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)

apply_dark_mode()

if 'twin' not in st.session_state:
    nodes, edges = build_base_nodes_and_edges()
    st.session_state.twin = DigitalTwin(
        nodes=nodes, edges=edges, cluster_profiles={},
        total_population=0, flood_severity=0.0, lgu_threat=False,
        col_map=col_map
    )

_defaults = {
    'data_calibrated': False, 'respondent_clusters': None,
    'current_barangay': "All Barangays", 'raw_data': None,
    'disable_flashing': False, 'use_pagasa_auto': True,
    'pagasa_severity': None, 'prev_pagasa_severity': None,
    'baseline_params': None, 'sensitivity_results': None,
    'sensitivity_param': "", 'sensitivity_active': False,
    'baseline_node_scores': None, 'final_node_scores': None,
    'sensitivity_start_val': None, 'sensitivity_end_val': None,
    'sensitivity_unit': ""
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def mm_to_severity(mm):
    if mm <= 10:        return mm / 40.0
    elif mm <= 30:      return 0.25 + (mm - 10) / 80.0
    elif mm <= 60:      return 0.50 + (mm - 30) / 120.0
    else:               return 0.75 + (mm - 60) / 160.0

def severity_to_mm(sev):
    if sev <= 0.25:     return sev * 40.0
    elif sev <= 0.50:   return 10.0 + (sev - 0.25) * 80.0
    elif sev <= 0.75:   return 30.0 + (sev - 0.50) * 120.0
    else:               return 60.0 + (sev - 0.75) * 160.0

# ---------- Sidebar ----------
with st.sidebar:
    st.toggle("🌙 Dark Mode", key="dark_mode")

    st.header("📊 Data Upload & Calibration")
    cac_vars = ['Challenge', 'Acceptance', 'Commitment']
    csv_columns = ['Respondent_Name', 'Barangay_Name']
    for node, clean_name in col_map.items():
        for cac in cac_vars:
            csv_columns.append(f"{clean_name}_{cac}")

    template_df = pd.DataFrame({
        col: ['Sample Name', 'Sample Barangay'] if col in ['Respondent_Name', 'Barangay_Name'] else [50, 60]
        for col in csv_columns
    })
    st.download_button("📥 Download CAC CSV Template", template_df.to_csv(index=False).encode('utf-8'),
                       'twin_cac_template.csv', 'text/csv')

    uploaded_file = st.file_uploader("Upload Resident Survey Data (CSV)", type=["csv"], key="uploader")

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            missing = [c for c in csv_columns if c not in df_raw.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing[:5])}...")
                st.button("🔄 Recalibrate (disabled)", disabled=True)
            else:
                st.success(f"✅ Loaded {len(df_raw)} residents across {df_raw['Barangay_Name'].nunique()} Barangay(s).")
                st.session_state.raw_data = df_raw
        except Exception as e:
            st.error(f"Error reading file: {e}")
    if uploaded_file is None and st.session_state.raw_data is not None:
        df_raw = st.session_state.raw_data
        st.info(f"📌 Using previously uploaded data ({len(df_raw)} residents). Re‑upload to replace.")
    elif uploaded_file is None:
        df_raw = None

    if df_raw is not None:
        st.subheader("📍 Select Target Barangay")
        barangays = ["All Barangays"] + sorted(df_raw['Barangay_Name'].unique().tolist())
        current_brgy = st.session_state.get('current_barangay', 'All Barangays')
        selected_barangay = st.selectbox("Choose a barangay to focus on", barangays,
                                         index=barangays.index(current_brgy) if current_brgy in barangays else 0,
                                         key="barangay_filter")
        st.session_state.current_barangay = selected_barangay

        if selected_barangay != "All Barangays":
            df_filtered = df_raw[df_raw['Barangay_Name'] == selected_barangay].copy()
            if len(df_filtered) == 0:
                st.warning(f"No data found for {selected_barangay}.")
                st.stop()
            else:
                st.info(f"📌 Analysing **{selected_barangay}** only. Click 'Recalibrate' to update the twin.")
        else:
            df_filtered = df_raw
            st.info("📌 Analysing **all barangays combined**. Click 'Recalibrate' to update the twin.")

        st.subheader("⚙️ K-Means Cluster Settings")
        k_mode = st.radio("How many clusters should we create?",
                          ["Auto (silhouette)", "Fixed (3 clusters)", "Manual"], index=0, key="k_mode")
        manual_k = None
        if k_mode == "Manual":
            manual_k = st.slider("Number of clusters (K)", 2, 5, 3, key="manual_k_slider")

        st.session_state.prev_k_mode = k_mode

        if len(df_filtered) < 3:
            st.error("Need at least 3 respondents for clustering.")
        else:
            if st.button("🔄 Recalibrate Model with Selected Data", use_container_width=True):
                with st.spinner("Running K-Means and generating baseline..."):
                    profiles, chosen_k, labeled_df = run_calibration(df_filtered, k_mode, manual_k)
                    data_hash = hashlib.md5(df_filtered.to_csv(index=False).encode()).hexdigest()
                    seed = int(data_hash, 16) % (2**32)
                    baseline_severity = st.session_state.twin.flood_severity

                    nodes, edges = build_base_nodes_and_edges()
                    st.session_state.twin = DigitalTwin(
                        nodes=nodes, edges=edges, cluster_profiles=profiles,
                        total_population=len(df_filtered), flood_severity=baseline_severity,
                        lgu_threat=False, seed=seed, col_map=col_map
                    )
                    st.session_state.respondent_clusters = labeled_df
                    st.session_state.data_calibrated = True
                    st.session_state.baseline_params = dict(
                        cluster_profiles=profiles, total_population=len(df_filtered),
                        flood_severity=baseline_severity, lgu_threat=False, seed=seed
                    )
                    st.session_state.sensitivity_active = False
                    st.success(f"Baseline established. K = {chosen_k}, population = {len(df_filtered)}. Ready for scenario exploration.")
                    st.rerun()
    else:
        st.info("📝 Upload a 38‑column CSV to begin.")

    st.markdown("---")
    st.header("🔄 Restore Baseline")
    run_disabled = (st.session_state.twin.total_population == 0)
    if st.button("♻️ Restore Baseline", use_container_width=True, disabled=run_disabled):
        if st.session_state.baseline_params is not None:
            bp = st.session_state.baseline_params
            nodes, edges = build_base_nodes_and_edges()
            st.session_state.twin = DigitalTwin(
                nodes=nodes, edges=edges, cluster_profiles=bp['cluster_profiles'],
                total_population=bp['total_population'], flood_severity=bp['flood_severity'],
                lgu_threat=bp['lgu_threat'], seed=bp['seed'], col_map=col_map
            )
            st.session_state.use_pagasa_auto = True
            st.session_state.sensitivity_active = False
            st.success("Baseline restored.")
        else:
            st.warning("No baseline available. Upload data first.")
        st.rerun()

    st.markdown("---")
    st.subheader("📏 Projected Population Size")
    pop_disabled = (len(st.session_state.twin.cluster_profiles) == 0)
    new_pop = st.slider("Projected Population Size", 10, 5000, st.session_state.twin.total_population,
                        key="pop_slider", disabled=pop_disabled,
                        help="Applies the current cluster proportions to a different total number of residents.")
    st.caption("ℹ️ *“If our barangay has 5 000 residents (instead of the 600 sampled), what would the behavioural metrics look like under the same psychological profile?”*")
    if not pop_disabled and new_pop != st.session_state.twin.total_population:
        st.session_state.twin.update_population_size(new_pop)
        st.warning(f"Population scaled to {new_pop}. Metrics recalculated.")
        st.rerun()

    # PAGASA Advisory section
    st.markdown("---")
    st.subheader("🌧️ PAGASA Advisory (Tagoloan River Basin)")
    advisory = fetch_pagasa_advisory()
    pagasa_severity, pagasa_label = render_pagasa_gauge(advisory, st.session_state.disable_flashing, st.session_state.disable_flashing)

    if st.session_state.use_pagasa_auto and pagasa_severity is not None and not st.session_state.sensitivity_active:
        st.session_state.twin.flood_severity = pagasa_severity

    if st.button("🔄 Refresh Live Advisory"):
        fetch_pagasa_advisory.clear()
        st.rerun()

    st.session_state.disable_flashing = st.checkbox("Disable flashing", value=st.session_state.disable_flashing, key="disable_flash_check")

    st.markdown("---")
    st.subheader("🧪 Simulated Flood Severity")
    st.session_state.use_pagasa_auto = st.checkbox("Auto‑mode (use PAGASA advisory)", value=st.session_state.use_pagasa_auto)
    if st.session_state.use_pagasa_auto:
        st.caption("Slider is locked to PAGASA advisory value. Uncheck to manually simulate.")
        flood_sev = st.session_state.twin.flood_severity
    else:
        flood_sev = st.slider("Severity (0–1)", 0.0, 1.0, st.session_state.twin.flood_severity, 0.01, key="flood_sev_slider",
                              help="0 = light rain, 1 = extreme rainfall.")
        st.session_state.twin.flood_severity = flood_sev

    sim_label = render_sim_gauge(flood_sev)
    lgu_threat = st.toggle("LGU Demolition Threat", value=st.session_state.twin.lgu_threat)
    if flood_sev != st.session_state.twin.flood_severity or lgu_threat != st.session_state.twin.lgu_threat:
        if st.button("Apply Environmental Triggers", disabled=run_disabled):
            st.session_state.twin.reset(new_flood_severity=flood_sev, new_lgu_threat=lgu_threat)
            st.rerun()

    st.markdown("---")
    st.subheader("🌊 Tagoloan River Water Level")
    render_waterlevel_gauge()

    # ---------- Sensitivity Analysis ----------
    st.markdown("---")
    st.header("📊 Sensitivity Analysis")
    if run_disabled:
        st.caption("Upload and calibrate data first.")
    else:
        param_type = st.radio("Parameter to vary", ["Flood Severity", "CAC Construct"], key="sensitivity_param_type")

        CONSTRUCT_METRIC_MAP = {
            "Prevention and flooding":       ["Evacuating %", "Proactive %"],
            "Coping during flooding":        ["Evacuating %", "Proactive %"],
            "Flooding and Family":           ["Relocate %"],
            "Desire for relocation":         ["Relocate %"],
            "Preference and adaptation":     ["Relocate %", "Relocation Readiness %"],
            "Feasibility of relocation":     ["Relocate %"],
            "Fear of housing demolition":    ["Relocate %", "Resisting LGU %", "Demolition Anxiety %"],
            "Viewpoints towards LGU":        ["Evacuating %", "LGU Trust %"],
            "Assistance for relocation":     ["Relocate %", "LGU Trust %"],
            "Rights to live in the area":    ["Resisting LGU %"],
            "Living in the disaster area":   ["Resisting LGU %"],
            "Family history and identity":   ["Relocate %", "Resisting LGU %", "Heritage Refusal %"]
        }

        if param_type == "Flood Severity":
            st.info("ℹ️ Flood Severity directly affects **Evacuating %**. Other indicators remain stable.\n\n"
                    "0–10 mm : Light rain | 10–30 mm : Moderate rain | 30–60 mm : Heavy rain | >60 mm : Torrential rain")
            mm_start = st.number_input("Start rainfall (mm)", 0.0, 100.0, 0.0, 1.0)
            mm_end   = st.number_input("End rainfall (mm)",   0.0, 100.0, 100.0, 1.0)
            start_val = mm_to_severity(mm_start)
            end_val   = mm_to_severity(mm_end)
            chosen_construct = None
            component = None
        else:
            construct_list = list(col_map.keys())
            chosen_construct = st.selectbox("Construct", construct_list, key="sens_construct")
            component = st.radio("Component", ["Challenge", "Acceptance", "Commitment"], key="sens_component")
            affected = CONSTRUCT_METRIC_MAP.get(chosen_construct, [])
            if affected:
                st.info(f"ℹ️ Changing **{chosen_construct}** ({component}) will immediately affect: {', '.join(affected)}.")
            else:
                st.info(f"ℹ️ **{chosen_construct}** does not directly appear in any decision formula.")
            start_val = st.number_input(f"Start {component} (%)", 0.0, 100.0, 0.0, 1.0)
            end_val   = st.number_input(f"End {component} (%)",   0.0, 100.0, 100.0, 1.0)

        n_steps = st.slider("Number of steps", 5, 30, 10)

        if st.button("▶️ Run Sensitivity Analysis", disabled=run_disabled):
            df_result, baseline_scores, final_scores = run_sensitivity(
                st.session_state.twin, param_type, chosen_construct, component,
                start_val, end_val, n_steps
            )
            if param_type == "Flood Severity":
                st.session_state.sensitivity_start_val = mm_start
                st.session_state.sensitivity_end_val = mm_end
                st.session_state.sensitivity_unit = "mm"
                df_result['Parameter Value'] = df_result['Parameter Value'].apply(severity_to_mm)
                df_result = df_result.rename(columns={'Parameter Value': 'Rainfall (mm)'})
            else:
                st.session_state.sensitivity_start_val = start_val
                st.session_state.sensitivity_end_val = end_val
                st.session_state.sensitivity_unit = "%"
                df_result = df_result.rename(columns={'Parameter Value': f'{chosen_construct} {component} (%)'})

            st.session_state.sensitivity_results = df_result
            st.session_state.sensitivity_param = f"{param_type} – {chosen_construct} {component}" if param_type == "CAC Construct" else "Flood Severity"
            st.session_state.baseline_node_scores = baseline_scores
            st.session_state.final_node_scores = final_scores
            st.session_state.sensitivity_active = True
            st.rerun()

# ---------- Main Dashboard ----------
barangay_title = st.session_state.current_barangay if st.session_state.current_barangay != "All Barangays" else "Municipal"
st.title(f"Tagoloan Flood-Prone Communities Digital Twin ({barangay_title})")
st.markdown("*Municipality of Tagoloan, Misamis Oriental*")

if st.session_state.sensitivity_active:
    st.info("🔬 **Sensitivity Scenario Active** – The dashboard below reflects the selected sensitivity parameters. Click 'Restore Baseline' to return to the original calibrated state.")

twin = st.session_state.twin
metrics = twin.get_metrics()
advanced = twin.get_advanced_metrics()
pop = metrics['Total Population']
reloc_pct = metrics['Projected to Relocate (%)']
evac_pct = metrics['Evacuating (%)']
resist_pct = metrics['Resisting LGU (%)']

# ---- Interactive Map Viewer (auto-loads DOST-PAGASA image) ----
st.subheader("🗺️ Tagoloan River Basin (Interactive Hazard Map)")

# The map HTML already defaults to a built-in upload screen. We inject a tiny script
# to auto-load the official DOST-PAGASA image when the page renders inside Streamlit.
MAP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tagoloan River Basin — Interactive Map</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
:root{--p:rgba(14,20,30,0.93);--a:#D32F2F;--t:#DDE4EC;--m:#6B7D8D;--b:rgba(255,255,255,0.08);--g:#43A047;--bl:#1E88E5;--y:#FFB300}
*{margin:0;padding:0;box-sizing:border-box}
body{overflow:hidden;background:#111820;font-family:'DM Sans',sans-serif;color:var(--t)}
#upload-screen{position:fixed;inset:0;background:#111820;display:flex;align-items:center;justify-content:center;z-index:200}
#upload-screen.gone{display:none}
.ubox{text-align:center;max-width:440px;padding:40px}
.ubox h1{font-size:22px;font-weight:700;color:#fff;margin-bottom:6px}
.ubox .sub{font-size:12px;color:var(--a);font-weight:600;text-transform:uppercase;letter-spacing:2px;margin-bottom:24px}
.ubox p{font-size:13px;color:var(--m);line-height:1.6;margin-bottom:28px}
.drop{border:2px dashed rgba(255,255,255,.12);border-radius:16px;padding:48px 32px;cursor:pointer;transition:all .2s;position:relative}
.drop:hover,.drop.over{border-color:var(--a);background:rgba(211,47,47,.05)}
.drop i{font-size:36px;color:var(--m);margin-bottom:14px;display:block;transition:color .2s}
.drop:hover i{color:var(--a)}
.drop .dt{font-size:14px;font-weight:600;color:var(--t);margin-bottom:4px}
.drop .dh{font-size:11px;color:var(--m)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer}
#viewport{position:fixed;inset:0;overflow:hidden;cursor:grab;display:none}
#viewport.on{display:block}
#viewport:active{cursor:grabbing}
#mw{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform}
#mw img{display:block;width:100%;height:100%;pointer-events:none;-webkit-user-drag:none}
.st{position:absolute;width:30px;height:30px;border-radius:50%;border:2.5px solid #fff;transform:translate(-50%,-50%);cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.4);z-index:2;transition:filter .15s}
.st:hover{filter:brightness(1.25);z-index:3}
.st.sel{z-index:4}
.st.sel::after{content:'';position:absolute;inset:-6px;border-radius:50%;border:2px solid;animation:pu 1.5s ease-in-out infinite}
.st.ed{cursor:move;filter:brightness(1.3) drop-shadow(0 0 6px rgba(255,255,255,.4))}
@keyframes pu{0%,100%{opacity:.9;transform:scale(1)}50%{opacity:.35;transform:scale(1.4)}}
.sl{position:absolute;left:50%;bottom:calc(100% + 3px);transform:translateX(-50%);white-space:nowrap;font-size:8.5px;font-weight:600;color:#fff;background:rgba(14,20,30,.88);padding:2px 6px;border-radius:3px;pointer-events:none;opacity:0;transition:opacity .12s}
.st:hover .sl,.st.sel .sl{opacity:1}
#tb{position:fixed;top:14px;right:14px;display:none;flex-direction:column;gap:4px;z-index:10}
#tb.on{display:flex}
.tbtn{width:36px;height:36px;border:none;border-radius:8px;background:var(--p);color:var(--t);font-size:13px;cursor:pointer;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(12px);border:1px solid var(--b);transition:background .15s,color .15s}
.tbtn:hover{background:rgba(211,47,47,.2);color:#fff}
.tbtn.on{background:rgba(211,47,47,.25);color:var(--a)}
.dlbtn{width:auto;padding:0 14px;font-size:11px;font-weight:700;gap:7px;background:rgba(67,160,71,.15)!important;border-color:rgba(67,160,71,.3)!important;color:#81C784!important}
.dlbtn:hover{background:rgba(67,160,71,.3)!important}
.tsep{height:1px;background:var(--b);margin:2px 4px}
#ttl{position:fixed;top:14px;left:14px;z-index:10;background:var(--p);border:1px solid var(--b);border-radius:10px;padding:10px 14px;backdrop-filter:blur(12px);max-width:300px;display:none}
#ttl.on{display:block}
#ttl .ag{font-size:8px;text-transform:uppercase;letter-spacing:2px;color:var(--a);font-weight:700}
#ttl h1{font-size:14px;font-weight:700;color:#fff;margin-top:2px;line-height:1.2}
#ttl .mt{font-size:9.5px;color:var(--m);margin-top:5px;line-height:1.4}
#mm{position:fixed;bottom:14px;left:14px;z-index:10;border-radius:8px;overflow:hidden;border:1px solid var(--b);background:var(--p);backdrop-filter:blur(12px);padding:6px;cursor:pointer;display:none}
#mm.on{display:block}
#mmi{display:block;border-radius:4px;background-size:cover;background-position:center;pointer-events:none}
#mmv{position:absolute;border:1.5px solid var(--a);background:rgba(211,47,47,.1);border-radius:2px;pointer-events:none}
#sp{position:fixed;bottom:14px;right:14px;width:265px;background:var(--p);border:1px solid var(--b);border-radius:10px;padding:14px;z-index:12;backdrop-filter:blur(12px);transform:translateY(8px);opacity:0;pointer-events:none;transition:all .2s}
#sp.show{transform:translateY(0);opacity:1;pointer-events:auto}
.sph{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.spi{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff;flex-shrink:0}
.spn{font-weight:700;font-size:12px;color:#fff}.spt{font-size:9px;color:var(--m)}
.spc{margin-left:auto;background:none;border:none;color:var(--m);cursor:pointer;font-size:13px;padding:4px}.spc:hover{color:#fff}
.spr{display:flex;justify-content:space-between;font-size:10.5px;padding:3px 0;border-top:1px solid rgba(255,255,255,.04)}
.spr .l{color:var(--m)}.spr .v{color:var(--t);font-weight:500;text-align:right;max-width:145px}
#cb{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);font-size:9px;color:rgba(255,255,255,.3);z-index:10;pointer-events:none;font-variant-numeric:tabular-nums;display:none}
#cb.on{display:block}
#tt{position:fixed;pointer-events:none;background:rgba(14,20,30,.92);border:1px solid var(--b);border-radius:6px;padding:6px 10px;font-size:10px;z-index:20;opacity:0;transition:opacity .1s;backdrop-filter:blur(8px);max-width:220px}
#tt .tn{font-weight:600;color:#fff;margin-bottom:1px}
#tt .ts{color:var(--m)}
#hint{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--p);border:1px solid var(--b);border-radius:10px;padding:14px 22px;z-index:15;text-align:center;backdrop-filter:blur(12px);transition:opacity .8s;pointer-events:none}
#hint.off{opacity:0}
#hint p{font-size:11px;color:var(--t);line-height:1.6}
#hint .hi{color:var(--a);font-weight:600}
#hint .hk{display:inline-block;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:4px;padding:1px 5px;font-size:9px;font-weight:600;margin:0 1px}
#eb{position:fixed;top:56px;right:14px;background:rgba(255,152,0,.15);border:1px solid rgba(255,152,0,.3);border-radius:8px;padding:8px 14px;z-index:10;font-size:10px;color:#FFB300;opacity:0;pointer-events:none;transition:opacity .2s;backdrop-filter:blur(8px)}
#eb.show{opacity:1;pointer-events:auto}
#toast{position:fixed;top:60px;left:50%;transform:translateX(-50%) translateY(-20px);background:var(--p);border:1px solid var(--b);border-radius:8px;padding:10px 20px;font-size:11px;z-index:50;opacity:0;transition:all .3s;pointer-events:none;backdrop-filter:blur(12px);white-space:nowrap;max-width:92vw;text-align:center}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
#toast.ok{border-color:rgba(67,160,71,.3);color:#81C784}
#toast.no{border-color:rgba(255,152,0,.3);color:#FFB74D}
@media(max-width:768px){
  #ttl{max-width:200px;padding:8px 10px}#ttl h1{font-size:12px}#ttl .mt{display:none}
  #sp{width:220px}#mm{display:none!important}#hint{max-width:90vw;padding:10px 14px}
  .dlbtn{font-size:10px;padding:0 8px}
}
</style>
</head>
<body>

<div id="upload-screen">
  <div class="ubox">
    <div class="sub">DOST-PAGASA</div>
    <h1>Tagoloan River Basin<br>Interactive Map Maker</h1>
    <p>Upload your hazard map image below. It will be turned into a fully interactive map with pan, zoom, and clickable monitoring stations — then downloadable as a single file.</p>
    <label class="drop" id="drop-zone">
      <i class="fas fa-cloud-upload-alt"></i>
      <div class="dt">Drop your map image here</div>
      <div class="dh">or click to browse — JPG, PNG, WebP</div>
      <input type="file" id="fi" accept="image/*">
    </label>
  </div>
</div>

<div id="viewport">
  <div id="mw"><img id="mi" alt="Map" draggable="false"></div>
</div>

<div id="ttl">
  <div class="ag">DOST-PAGASA</div>
  <h1>Tagoloan River Basin Hazard Map</h1>
  <div class="mt">Basin Area: 1,704 km² — Misamis Oriental, Bukidnon, Agusan del Sur<br>May 2024</div>
</div>

<div id="tb">
  <button class="tbtn dlbtn" id="b-dl" title="Download"><i class="fas fa-download"></i> Download</button>
  <div class="tsep"></div>
  <button class="tbtn" id="b-zi" title="Zoom In"><i class="fas fa-plus"></i></button>
  <button class="tbtn" id="b-zo" title="Zoom Out"><i class="fas fa-minus"></i></button>
  <button class="tbtn" id="b-fit" title="Fit View"><i class="fas fa-expand"></i></button>
  <div class="tsep"></div>
  <button class="tbtn" id="b-edit" title="Move Stations (E)"><i class="fas fa-arrows-alt"></i></button>
  <div class="tsep"></div>
  <button class="tbtn" id="b-sta" title="Toggle Stations"><i class="fas fa-map-marker-alt" style="font-size:11px"></i></button>
</div>

<div id="mm"><div id="mmi"></div><div id="mmv"></div></div>
<div id="sp">
  <div class="sph"><div class="spi" id="s-ic"><i class="fas fa-tint"></i></div><div><div class="spn" id="s-nm">--</div><div class="spt" id="s-tp">--</div></div><button class="spc" id="s-cl"><i class="fas fa-times"></i></button></div>
  <div class="spr"><span class="l">Elevation</span><span class="v" id="s-el">--</span></div>
  <div class="spr"><span class="l">Details</span><span class="v" id="s-in">--</span></div>
  <div class="spr"><span class="l">Status</span><span class="v" id="s-st" style="color:var(--g)">Active</span></div>
  <div class="spr"><span class="l">Coordinates</span><span class="v" id="s-co">--</span></div>
  <div class="spr"><span class="l">Position</span><span class="v" id="s-mp">--</span></div>
</div>
<div id="cb"></div>
<div id="tt"><div class="tn" id="t-tn">--</div><div class="ts" id="t-ts">--</div></div>
<div id="hint"><p><span class="hi">Interactive Map</span><br>Drag to pan · Scroll to zoom · Click stations<br><span class="hk">+</span><span class="hk">-</span> Zoom · <span class="hk">R</span> Reset · <span class="hk">E</span> Move stations · <span class="hk">D</span> Download</p></div>
<div id="eb"><i class="fas fa-info-circle"></i> Drag station markers to reposition</div>
<div id="toast"></div>

<script>
// Auto-load the official DOST-PAGASA map image on startup
(function(){
  var img = new Image();
  img.crossOrigin = 'anonymous';
  img.src = 'https://pubfiles.pagasa.dost.gov.ph/pagasaweb/images/basins/tagoloan-river-basin.jpg';
  img.onload = function(){
    document.getElementById('mi').src = img.src;
    // The existing onload handler for #mi will fire and initialise the map
  };
})();

// ============================================================
// DATA
// ============================================================
var WW = 2200;
var ST = [
  {id:0,nm:'Malaybalay Rain Gauge',tp:'rainfall',x:49,y:16,el:'1,280 m',inf:'Annual: 2,650mm | Installed: 2018',co:'8.15\u00B0N, 125.08\u00B0E'},
  {id:1,nm:'Impasug-ong Rain Gauge',tp:'rainfall',x:34,y:21,el:'1,400 m',inf:'Annual: 2,900mm | Installed: 2017',co:'8.42\u00B0N, 124.78\u00B0E'},
  {id:2,nm:'Sumilao Rain Gauge',tp:'rainfall',x:29,y:34,el:'1,150 m',inf:'Annual: 2,380mm | Installed: 2020',co:'8.38\u00B0N, 124.95\u00B0E'},
  {id:3,nm:'Lantapan Rain Gauge',tp:'rainfall',x:63,y:19,el:'1,350 m',inf:'Annual: 2,800mm | Installed: 2019',co:'8.10\u00B0N, 125.28\u00B0E'},
  {id:4,nm:'Manolo Fortich WL Station',tp:'waterlevel',x:47,y:39,el:'890 m',inf:'Alert: 3.2m | Warning: 4.1m',co:'8.35\u00B0N, 125.07\u00B0E'},
  {id:5,nm:'Libona Weather Station',tp:'weather',x:55,y:51,el:'760 m',inf:'Temp: 24.5\u00B0C | Humidity: 82%',co:'8.33\u00B0N, 125.15\u00B0E'},
  {id:6,nm:'Baungon WL Station',tp:'waterlevel',x:44,y:59,el:'620 m',inf:'Alert: 2.8m | Warning: 3.5m',co:'8.32\u00B0N, 125.05\u00B0E'},
  {id:7,nm:'Tagoloan WL Station',tp:'waterlevel',x:50,y:73,el:'180 m',inf:'Alert: 2.5m | Warning: 3.0m',co:'8.54\u00B0N, 125.08\u00B0E'},
  {id:8,nm:'CDO Weather Station',tp:'weather',x:53,y:86,el:'45 m',inf:'Temp: 28.2\u00B0C | Humidity: 78%',co:'8.48\u00B0N, 124.64\u00B0E'},
  {id:9,nm:'Villanueva WL Station',tp:'waterlevel',x:48,y:94,el:'20 m',inf:'Alert: 2.0m | Warning: 2.8m',co:'8.56\u00B0N, 125.01\u00B0E'}
];
var SC={rainfall:'#43A047',waterlevel:'#1E88E5',weather:'#FFB300'};
var SI={rainfall:'fa-cloud-rain',waterlevel:'fa-water',weather:'fa-cloud-sun'};
var SN={rainfall:'Rainfall Gauge',waterlevel:'Water Level Station',weather:'Weather Station'};

// ============================================================
// STATE
// ============================================================
var zm=1,ox=0,oy=0,bs=1,ia=1,drag=false,ds=[0,0],dox=0,doy=0;
var edit=false,staVis=true,ready=false,sel=-1,hov=-1;
var dStn=null,dOff=[0,0],pD=0,pM=[0,0],sEls=[];
var imgDU=null;

// ============================================================
// DOM
// ============================================================
var vp=document.getElementById('viewport'),mw=document.getElementById('mw'),mi=document.getElementById('mi');
var mm=document.getElementById('mm'),mmi=document.getElementById('mmi'),mmv=document.getElementById('mmv');
var sp=document.getElementById('sp'),tt=document.getElementById('tt'),cb=document.getElementById('cb');
var hint=document.getElementById('hint'),eb=document.getElementById('eb'),toast=document.getElementById('toast');
var dropZone=document.getElementById('drop-zone'),fi=document.getElementById('fi');

function toast(m,t){toast.textContent=m;toast.className='show '+(t||'');clearTimeout(toast._t);toast._t=setTimeout(function(){toast.className=''},4000)}

dropZone.addEventListener('dragover',function(e){e.preventDefault();dropZone.classList.add('over')});
dropZone.addEventListener('dragleave',function(){dropZone.classList.remove('over')});
dropZone.addEventListener('drop',function(e){e.preventDefault();dropZone.classList.remove('over');if(e.dataTransfer.files[0])loadFile(e.dataTransfer.files[0])});
fi.addEventListener('change',function(e){if(e.target.files[0])loadFile(e.target.files[0])});

function loadFile(file){
  if(!file||!file.type.startsWith('image/')){toast('Please select an image file.','no');return}
  var r=new FileReader();
  r.onload=function(e){imgDU=e.target.result;mi.src=imgDU;};
  r.readAsDataURL(file);
}

mi.onload=function(){
  var w=mi.naturalWidth,h=mi.naturalHeight;
  if(w<10||h<10)return;
  ia=w/h;
  mw.style.width=WW+'px';mw.style.height=Math.round(WW/ia)+'px';
  cbs();fit();setupMM();makeStations();
  document.getElementById('upload-screen').classList.add('gone');
  vp.classList.add('on');document.getElementById('tb').classList.add('on');
  document.getElementById('ttl').classList.add('on');mm.classList.add('on');
  cb.classList.add('on');
  ready=true;updMM();
  setTimeout(function(){hint.classList.add('off')},4000);
};

window.addEventListener('resize',function(){if(ready){cbs();updTx();updMM()}});
function cbs(){var vw=innerWidth,vh=innerHeight;bs=Math.min(vw/WW,vh/(WW/ia))*.94}
function updTx(){mw.style.transform='translate('+ox+'px,'+oy+'px) scale('+(bs*zm)+')'}
function fit(){var vw=innerWidth,vh=innerHeight,mh=WW/ia;ox=(vw-WW*bs)/2;oy=(vh-mh*bs)/2;zm=1;updTx();updMM()}
function zAt(cx,cy,f){var os=bs*zm,nz=Math.max(.3,Math.min(25,zm*f)),ns=bs*nz,mx=(cx-ox)/os,my=(cy-oy)/os;zm=nz;ox=cx-mx*ns;oy=cy-my*ns;updTx();updMM()}

function setupMM(){var w=150,h=Math.round(w/ia);mm.style.width=(w+12)+'px';mm.style.height=(h+12)+'px';mmi.style.width=w+'px';mmi.style.height=h+'px';mmi.style.backgroundImage='url('+mi.src+')';mmi.style.backgroundSize='cover'}
function updMM(){if(!ready)return;var s=bs*zm,mw2=WW,mh=WW/ia,x1=-ox/s,y1=-oy/s,x2=x1+innerWidth/s,y2=y1+innerHeight/s;var l=Math.max(0,x1/mw2*100),t=Math.max(0,y1/mh*100),w=Math.min(100,(x2-x1)/mw2*100),h=Math.min(100,(y2-y1)/mh*100);mmv.style.left=l+'%';mmv.style.top=t+'%';mmv.style.width=w+'%';mmv.style.height=h+'%';mmv.style.display=(w>=98&&h>=98)?'none':'block'}
mm.addEventListener('click',function(e){if(!ready)return;var r=mmi.getBoundingClientRect(),mx=(e.clientX-r.left)/r.width,my=(e.clientY-r.top)/r.height,s=bs*zm;ox=innerWidth/2-mx*WW*s;oy=innerHeight/2-my*(WW/ia)*s;updTx();updMM()});

function ldPos(){try{var s=localStorage.getItem('tstp');if(s)JSON.parse(s).forEach(function(p){if(ST[p.id]){ST[p.id].x=p.x;ST[p.id].y=p.y}})}catch(e){}}
function svPos(){try{localStorage.setItem('tstp',JSON.stringify(ST.map(function(s){return{id:s.id,x:+s.x.toFixed(2),y:+s.y.toFixed(2)}})))}catch(e){}}

function makeStations(){
  ldPos();
  ST.forEach(function(s){
    var el=document.createElement('div');el.className='st';
    el.style.left=s.x+'%';el.style.top=s.y+'%';
    el.style.background=SC[s.tp];el.setAttribute('data-id',s.id);
    var st=document.createElement('style');
    st.textContent='.st[data-id="'+s.id+'"].sel::after{border-color:'+SC[s.tp]+'}';
    document.head.appendChild(st);
    var lb=document.createElement('div');lb.className='sl';lb.textContent=s.nm;
    el.appendChild(lb);
    el.addEventListener('mousedown',function(e){stnDown(e,s.id)});
    el.addEventListener('touchstart',function(e){stnTDown(e,s.id)},{passive:false});
    mw.appendChild(el);sEls[s.id]=el;
  });
}

function selSt(id){
  if(sel>=0&&sEls[sel])sEls[sel].classList.remove('sel');
  sel=id;
  if(id>=0){var s=ST[id];sEls[id].classList.add('sel');
    document.getElementById('s-ic').style.background=SC[s.tp];
    document.getElementById('s-ic').innerHTML='<i class="fas '+SI[s.tp]+'"></i>';
    document.getElementById('s-nm').textContent=s.nm;
    document.getElementById('s-tp').textContent=SN[s.tp];
    document.getElementById('s-el').textContent=s.el;
    document.getElementById('s-in').textContent=s.inf;
    document.getElementById('s-co').textContent=s.co;
    document.getElementById('s-mp').textContent=s.x.toFixed(1)+'%, '+s.y.toFixed(1)+'%';
    sp.classList.add('show');
  }else sp.classList.remove('show');
}

function stnDown(e,id){
  e.stopPropagation();
  if(edit){e.preventDefault();dStn=id;var s=ST[id],s0=bs*zm;
    dOff=[e.clientX-(ox+s.x/100*WW*s0),e.clientY-(oy+s.y/100*(WW/ia)*s0)];
    sEls[id].classList.add('ed');
  }else{
    selSt(id);var s=ST[id],ts=bs*Math.max(zm,2);
    ox=innerWidth/2-s.x/100*WW*ts;oy=innerHeight/2-s.y/100*(WW/ia)*ts;
    zm=ts/bs;updTx();updMM();
  }
}
function stnTDown(e,id){
  e.stopPropagation();
  if(edit&&e.touches.length===1){e.preventDefault();dStn=id;var s=ST[id],s0=bs*zm;
    dOff=[e.touches[0].clientX-(ox+s.x/100*WW*s0),e.touches[0].clientY-(oy+s.y/100*(WW/ia)*s0)];
    sEls[id].classList.add('ed');
  }else if(!edit)selSt(id);
}

vp.addEventListener('mousedown',function(e){if(dStn!==null)return;drag=true;ds=[e.clientX,e.clientY];dox=ox;doy=oy});
window.addEventListener('mousemove',function(e){
  if(dStn!==null){var s0=bs*zm,mw2=WW,mh=WW/ia;
    ST[dStn].x=Math.max(0,Math.min(100,((e.clientX-dOff[0]-ox)/s0/mw2)*100));
    ST[dStn].y=Math.max(0,Math.min(100,((e.clientY-dOff[1]-oy)/s0/mh)*100));
    sEls[dStn].style.left=ST[dStn].x+'%';sEls[dStn].style.top=ST[dStn].y+'%';
    if(sel===dStn)document.getElementById('s-mp').textContent=ST[dStn].x.toFixed(1)+'%, '+ST[dStn].y.toFixed(1)+'%';
    return}
  if(drag){ox=dox+(e.clientX-ds[0]);oy=doy+(e.clientY-ds[1]);updTx();updMM()}
  if(!ready)return;hov=-1;
  if(staVis){var s0=bs*zm,mw2=WW,mh=WW/ia;
    for(var i=0;i<ST.length;i++){var s=ST[i],px=ox+s.x/100*mw2*s0,py=oy+s.y/100*mh*s0;
      if(Math.hypot(e.clientX-px,e.clientY-py)<18){hov=s.id;break}}}
  if(hov>=0){var s=ST[hov];document.getElementById('t-tn').textContent=s.nm;
    document.getElementById('t-ts').textContent=SN[s.tp]+(edit?' \u2014 drag to move':' \u2014 click for details');
    tt.style.opacity='1';tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-10)+'px';
    vp.style.cursor=edit?'move':'pointer';
  }else{tt.style.opacity='0';vp.style.cursor=drag?'grabbing':'grab'}
  var s0=bs*zm;
  cb.textContent='X: '+((e.clientX-ox)/s0/WW*100).toFixed(1)+'%  Y: '+((e.clientY-oy)/s0/(WW/ia)*100).toFixed(1)+'%  |  Zoom: '+zm.toFixed(2)+'x';
});
window.addEventListener('mouseup',function(e){
  if(dStn!==null){sEls[dStn].classList.remove('ed');svPos();dStn=null;return}
  if(drag&&Math.hypot(e.clientX-ds[0],e.clientY-ds[1])<4&&!e.target.closest('.st'))selSt(-1);
  drag=false});
vp.addEventListener('wheel',function(e){e.preventDefault();zAt(e.clientX,e.clientY,e.deltaY<0?1.15:.87)},{passive:false});
vp.addEventListener('dblclick',function(e){zAt(e.clientX,e.clientY,2)});
vp.addEventListener('touchstart',function(e){
  if(dStn!==null)return;
  if(e.touches.length===1){drag=true;ds=[e.touches[0].clientX,e.touches[0].clientY];dox=ox;doy=oy}
  else if(e.touches.length===2){drag=false;pD=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);pM=[(e.touches[0].clientX+e.touches[1].clientX)/2,(e.touches[0].clientY+e.touches[1].clientY)/2]}
},{passive:false});
vp.addEventListener('touchmove',function(e){e.preventDefault();
  if(dStn!==null&&e.touches.length===1){var s0=bs*zm,mw2=WW,mh=WW/ia;
    ST[dStn].x=Math.max(0,Math.min(100,((e.touches[0].clientX-dOff[0]-ox)/s0/mw2)*100));
    ST[dStn].y=Math.max(0,Math.min(100,((e.touches[0].clientY-dOff[1]-oy)/s0/mh)*100));
    sEls[dStn].style.left=ST[dStn].x+'%';sEls[dStn].style.top=ST[dStn].y+'%';return}
  if(e.touches.length===1&&drag){ox=dox+(e.touches[0].clientX-ds[0]);oy=doy+(e.touches[0].clientY-ds[1]);updTx();updMM()}
  else if(e.touches.length===2){var nd=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);zAt(pM[0],pM[1],nd/pD);pD=nd}
},{passive:false});
vp.addEventListener('touchend',function(e){
  if(dStn!==null&&e.touches.length===0){sEls[dStn].classList.remove('ed');svPos();dStn=null;return}
  if(e.touches.length===0)drag=false});
window.addEventListener('keydown',function(e){
  switch(e.key){
    case'+':case'=':zAt(innerWidth/2,innerHeight/2,1.25);break;
    case'-':case'_':zAt(innerWidth/2,innerHeight/2,.8);break;
    case'ArrowLeft':ox+=40;updTx();updMM();break;
    case'ArrowRight':ox-=40;updTx();updMM();break;
    case'ArrowUp':oy+=40;updTx();updMM();break;
    case'ArrowDown':oy-=40;updTx();updMM();break;
    case'r':case'R':fit();break;
    case'e':case'E':togEdit();break;
    case'd':case'D':dlMap();break;
    case'Escape':selSt(-1);if(edit)togEdit();break;
  }
});
document.getElementById('b-zi').addEventListener('click',function(){zAt(innerWidth/2,innerHeight/2,1.4)});
document.getElementById('b-zo').addEventListener('click',function(){zAt(innerWidth/2,innerHeight/2,.71)});
document.getElementById('b-fit').addEventListener('click',fit);
document.getElementById('b-edit').addEventListener('click',togEdit);
document.getElementById('b-sta').addEventListener('click',function(){
  staVis=!staVis;document.getElementById('b-sta').classList.toggle('on',!staVis);
  sEls.forEach(function(el){el.style.display=staVis?'':'none'});if(!staVis)selSt(-1)});
document.getElementById('s-cl').addEventListener('click',function(){selSt(-1)});
vp.addEventListener('contextmenu',function(e){e.preventDefault()});

function togEdit(){edit=!edit;document.getElementById('b-edit').classList.toggle('on',edit);eb.classList.toggle('show',edit);sEls.forEach(function(el){el.style.cursor=edit?'move':'pointer'})}

document.getElementById('b-dl').addEventListener('click',dlMap);

function dlMap(){
  if(!imgDU){toast('No image to embed. This should not happen.','no');return}
  toast('Building download...','ok');
  setTimeout(function(){
    try{
      var html='<!DOCTYPE html>\n'+document.documentElement.outerHTML;
      var pos=JSON.stringify(ST.map(function(s){return{id:s.id,x:+s.x.toFixed(2),y:+s.y.toFixed(2)}})).replace(/'/g,"\\'");
      var preload="<script>try{localStorage.setItem('tstp','"+pos+"')}catch(e){}<\/script>";
      var upScreen=document.getElementById('upload-screen').outerHTML;
      var newUpScreen=upScreen.replace('id="upload-screen"','id="upload-screen" class="gone"');
      html=html.replace(upScreen,newUpScreen);
      var inject="<script>document.addEventListener('DOMContentLoaded',function(){document.getElementById('mi').src='"+imgDU+"'});<\/script>";
      html=html.replace('</head>',preload+inject+'</head>');
      html=html.replace('vp.classList.add(\'on\')','vp.classList.add(\'on\')');
      html=html.replace('document.getElementById(\'tb\').classList.add(\'on\')','document.getElementById(\'tb\').classList.add(\'on\')');
      html=html.replace('document.getElementById(\'ttl\').classList.add(\'on\')','document.getElementById(\'ttl\').classList.add(\'on\')');
      html=html.replace('mm.classList.add(\'on\')','mm.classList.add(\'on\')');
      html=html.replace('cb.classList.add(\'on\')','cb.classList.add(\'on\')');
      var blob=new Blob([html],{type:'text/html;charset=utf-8'});
      var url=URL.createObjectURL(blob);
      var a=document.createElement('a');a.href=url;
      a.download='Tagoloan_River_Basin_Interactive_Map.html';
      a.style.display='none';document.body.appendChild(a);a.click();
      setTimeout(function(){document.body.removeChild(a);URL.revokeObjectURL(url)},200);
      toast('Downloaded — works offline, no internet needed','ok');
    }catch(err){toast('Download failed: '+err.message,'no')}
  },150);
}
</script>
</body>
</html>"""

st.components.v1.html(MAP_HTML, height=700, scrolling=False)
st.caption("Interactive DOST‑PAGASA map with monitoring stations. Drag to pan, scroll to zoom, click stations for details.")

# ---- Basic Behavioral Outcomes ----
st.subheader("Community Behavioral Outcomes")
if pop == 0:
    st.info("Upload and calibrate your survey data to populate the indicators.")
else:
    st.caption(f"Realistic baselines from survey CAC data. Current flood severity: **{flood_sev:.2f}** – {sim_label}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Population", f"{pop:,}")
col2.metric("Projected to Relocate", f"{reloc_pct:.1f}%")
col3.metric("Evacuating", f"{evac_pct:.1f}%")
col4.metric("Resisting LGU", f"{resist_pct:.1f}%")

if pop > 0:
    snapshot_parts = []
    if reloc_pct > 30:
        snapshot_parts.append(f"High relocation pressure ({reloc_pct:.1f}%) indicates strong desire to move.")
    elif reloc_pct > 10:
        snapshot_parts.append(f"Moderate relocation interest ({reloc_pct:.1f}%) shows some openness to resettlement.")
    else:
        snapshot_parts.append(f"Low relocation intent ({reloc_pct:.1f}%) means most residents prefer to stay.")
    if evac_pct > 80:
        snapshot_parts.append(f"Very high evacuation readiness ({evac_pct:.1f}%) suggests the community is prepared to respond to warnings.")
    elif evac_pct > 50:
        snapshot_parts.append(f"Moderate evacuation readiness ({evac_pct:.1f}%) may need reinforcement.")
    else:
        snapshot_parts.append(f"Low evacuation readiness ({evac_pct:.1f}%) reveals a critical gap in disaster preparedness.")
    if resist_pct > 20:
        snapshot_parts.append(f"Significant LGU resistance ({resist_pct:.1f}%) signals friction.")
    elif resist_pct > 5:
        snapshot_parts.append(f"Low‑to‑moderate resistance ({resist_pct:.1f}%) — monitor clusters contributing to it.")
    else:
        snapshot_parts.append(f"Negligible resistance ({resist_pct:.1f}%) — community is largely cooperative.")
    st.caption(f"📊 **{barangay_title} snapshot:** {' '.join(snapshot_parts)}")

# ---- Advanced Behavioral Indicators ----
st.markdown("---")
st.subheader("Advanced Community Indicators")
if pop == 0:
    st.info("These indicators will populate after data calibration.")
else:
    st.caption("Derived from the CAC constructs and regression pathways, these metrics reveal deeper community dynamics.")
adv_col1, adv_col2, adv_col3, adv_col4, adv_col5 = st.columns(5)
adv_col1.metric("Proactive Preparedness", f"{advanced['Proactive Preparedness (%)']:.1f}%")
adv_col2.metric("LGU Trust & Cooperation", f"{advanced['LGU Trust & Cooperation (%)']:.1f}%")
adv_col3.metric("Heritage‑Based Refusal", f"{advanced['Heritage-Based Refusal (%)']:.1f}%")
adv_col4.metric("Demolition Anxiety", f"{advanced['Demolition Anxiety (%)']:.1f}%")
adv_col5.metric("Relocation Readiness", f"{advanced['Relocation Readiness (%)']:.1f}%")

if pop > 0:
    adv_interpretations = []
    if advanced['Proactive Preparedness (%)'] > 60:
        adv_interpretations.append("🟢 **High proactive preparedness**: many residents are already taking independent action – leverage them as community champions.")
    elif advanced['Proactive Preparedness (%)'] < 30:
        adv_interpretations.append("🔴 **Low proactive preparedness**: grassroots awareness campaigns are urgently needed.")
    else:
        adv_interpretations.append("🟡 **Moderate proactive preparedness**: continue building local capacity.")
    if advanced['LGU Trust & Cooperation (%)'] > 60:
        adv_interpretations.append("🟢 **Strong LGU trust**: policies will likely face less friction and higher compliance.")
    elif advanced['LGU Trust & Cooperation (%)'] < 30:
        adv_interpretations.append("🔴 **Weak LGU trust**: any new program will encounter resistance; invest in trust‑building first.")
    else:
        adv_interpretations.append("🟡 **Moderate trust**: maintain transparency to avoid erosion.")
    if advanced['Heritage-Based Refusal (%)'] > 30:
        adv_interpretations.append("⚠️ **High heritage‑based refusal**: monetary incentives alone won't work; consider community‑relocation or psychosocial support.")
    else:
        adv_interpretations.append("🟢 **Low heritage refusal**: relocation barriers are more practical than emotional.")
    if advanced['Demolition Anxiety (%)'] > 30:
        adv_interpretations.append("🔴 **Elevated demolition anxiety**: immediate housing security guarantees and MHPSS are critical.")
    elif advanced['Demolition Anxiety (%)'] > 10:
        adv_interpretations.append("🟡 **Moderate demolition anxiety**: monitor closely, especially if demolition threat is active.")
    else:
        adv_interpretations.append("🟢 **Low demolition anxiety**: community feels relatively secure about housing.")
    if advanced['Relocation Readiness (%)'] > 20:
        adv_interpretations.append("🟢 **High relocation readiness**: a pool of early adopters exists – target them for pilot resettlement programs.")
    elif advanced['Relocation Readiness (%)'] > 5:
        adv_interpretations.append("🟡 **Moderate readiness**: some residents are prepared; identify and encourage them.")
    else:
        adv_interpretations.append("🔴 **Low readiness**: psychological adaptation to relocation is minimal; phased engagement is necessary.")
    st.caption(" ".join(adv_interpretations))

# ---- Side‑by‑Side Network Graphs ----
st.markdown("---")
st.subheader("Socio‑Psychological Network")
if pop == 0:
    st.info("Upload and calibrate data to display the psychological network graph.")
else:
    col_left, col_right = st.columns(2)
    with col_left:
        if st.session_state.sensitivity_active and st.session_state.baseline_node_scores is not None:
            render_network_graph(twin, barangay_title, node_scores=st.session_state.baseline_node_scores)
            st.caption("Baseline psychological landscape (unchanged during sensitivity analysis).")
        else:
            render_network_graph(twin, barangay_title)
            st.caption("Current psychological landscape.")
    with col_right:
        if st.session_state.sensitivity_active and st.session_state.baseline_node_scores is not None and st.session_state.final_node_scores is not None:
            baseline = st.session_state.baseline_node_scores
            final = st.session_state.final_node_scores
            node_names = list(baseline.keys())
            changes = [final[n] - baseline[n] for n in node_names]

            G_impact = nx.DiGraph()
            for name in node_names:
                G_impact.add_node(name, change=changes[node_names.index(name)])
            for edge in twin.edges:
                G_impact.add_edge(edge.source_name, edge.target_name, weight=edge.coefficient)

            pos = nx.spring_layout(G_impact, k=0.35, seed=42)
            edge_x, edge_y = [], []
            for e in G_impact.edges():
                x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'),
                                    hoverinfo='none', mode='lines')
            node_x, node_y, node_text, node_color = [], [], [], []
            for n, d in G_impact.nodes(data=True):
                x, y = pos[n]
                node_x.append(x); node_y.append(y)
                node_text.append(f"{n}<br>Change: {d['change']:+.1f}")
                node_color.append(d['change'])
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                    text=list(G_impact.nodes()), textposition="bottom center",
                                    hovertext=node_text, hoverinfo='text',
                                    marker=dict(showscale=True, colorscale='RdBu', reversescale=False,
                                                color=node_color, size=25,
                                                colorbar=dict(thickness=10, title='Score Change')))
            fig_impact = go.Figure(data=[edge_trace, node_trace],
                                   layout=go.Layout(title='Intervention Effect (Change from Baseline)',
                                                    showlegend=False, margin=dict(b=20,l=5,r=5,t=40),
                                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            st.plotly_chart(fig_impact, use_container_width=True)
            st.caption("Magnitude of the intervention effect if an intervention is applied. Red = increase, Blue = decrease.")
        else:
            st.info("Run a sensitivity analysis to see the intervention effect here.")

# ---- Cluster Breakdown ----
st.markdown("---")
st.subheader("Behavioral Distribution by Cluster")
cluster_df = twin.analytics.get_cluster_breakdown()
if pop == 0:
    st.info("No clusters yet. Upload and recalibrate data.")
else:
    render_cluster_breakdown(cluster_df, barangay_title)

# ---- Respondent Details (if calibrated) ----
if st.session_state.data_calibrated and st.session_state.respondent_clusters is not None:
    st.markdown("---")
    st.subheader("📋 Respondent Details by Cluster")
    st.caption("Names and barangays of each respondent, grouped by their psychological cluster. Useful for targeted interventions.")
    df_resp = st.session_state.respondent_clusters
    profile_names = list(twin.cluster_profiles.keys())
    label_to_name = {i: profile_names[i] for i in range(len(profile_names))}
    df_resp['Cluster Name'] = df_resp['Cluster'].map(label_to_name)
    for cname, group in df_resp.groupby('Cluster Name'):
        with st.expander(f"{cname} ({len(group)} respondents)"):
            st.dataframe(group[['Respondent_Name', 'Barangay_Name']], use_container_width=True)
    csv_full = df_resp.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Respondent List with Cluster Labels", csv_full,
                       file_name='respondents_with_clusters.csv', mime='text/csv')

# ---- Cluster Profiles ----
st.markdown("---")
st.subheader("Cluster Profiles (K-Means Nomenclature)")
if st.session_state.data_calibrated:
    st.success(f"Profiles extracted from uploaded survey data (K={len(twin.cluster_profiles)}).")
else:
    st.info("No profiles yet. Upload and recalibrate data.")
profiles = [{"Cluster": name, "Share": f"{p.population_ratio*100:.1f}%",
             "Dominant Driver": p.dominant_driver}
            for name, p in twin.cluster_profiles.items()]
if profiles:
    st.dataframe(pd.DataFrame(profiles), use_container_width=True, hide_index=True)

# ---- CAC Bubble ----
st.markdown("---")
st.subheader("CAC Breakdown (Bubble: Commitment)")
cac_avgs = twin.analytics.get_cac_averages(col_map)
if pop == 0:
    st.info("CAC data will appear after calibration.")
else:
    render_cac_bubble(cac_avgs, col_map, barangay_title)

# ---- Policy Insights ----
st.markdown("---")
st.subheader("Policy Insights & Actionable Recommendations")
if pop == 0:
    st.info("Insights will appear after data is loaded and calibrated.")
else:
    render_policy_insights(
        twin, metrics, advanced, flood_sev, pagasa_label, barangay_title,
        sensitivity_active=st.session_state.sensitivity_active,
        sensitivity_param=st.session_state.get('sensitivity_param', ''),
        sensitivity_start_val=st.session_state.get('sensitivity_start_val'),
        sensitivity_end_val=st.session_state.get('sensitivity_end_val'),
        sensitivity_unit=st.session_state.get('sensitivity_unit', '')
    )

# ---- Sensitivity Results ----
if st.session_state.sensitivity_results is not None:
    st.markdown("---")
    st.subheader("📈 Sensitivity Analysis Results")
    st.caption(f"Varying **{st.session_state.sensitivity_param}**")
    df = st.session_state.sensitivity_results
    param_type_for_label = st.session_state.get('sensitivity_param_type', 'Flood Severity')
    if param_type_for_label == "Flood Severity":
        x_label = "Rainfall (mm)"
    else:
        x_label = df.columns[0]
    fig_sens = px.line(df, x=df.columns[0], y=['Relocate %', 'Evacuating %', 'Resisting LGU %',
                                                'Proactive %', 'LGU Trust %', 'Heritage Refusal %',
                                                'Demolition Anxiety %', 'Relocation Readiness %'],
                       title='Outcome vs Parameter',
                       labels={df.columns[0]: x_label})
    st.plotly_chart(fig_sens, use_container_width=True)
    st.dataframe(df, use_container_width=True)
