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
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None
    }
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
    'data_calibrated': False,
    'respondent_clusters': None,
    'current_barangay': "All Barangays",
    'raw_data': None,
    'disable_flashing': False,
    'use_pagasa_auto': True,
    'pagasa_severity': None,
    'prev_pagasa_severity': None,
    'baseline_params': None,
    'sensitivity_results': None,
    'sensitivity_param': "",
    'sensitivity_active': False,
    'baseline_node_scores': None,
    'final_node_scores': None,
    'sensitivity_start_val': None,
    'sensitivity_end_val': None,
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
        k_mode = st.radio(
            "How many clusters should we create?",
            ["Auto (silhouette)", "Fixed (3 clusters)", "Manual"],
            index=0, key="k_mode"
        )
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
    new_pop = st.slider(
        "Projected Population Size",
        10, 5000, st.session_state.twin.total_population,
        key="pop_slider", disabled=pop_disabled,
        help="Applies the current cluster proportions to a different total number of residents."
    )
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
                st.info(f"ℹ️ **{chosen_construct}** does not directly appear in any decision formula. Run simulation steps to see indirect effects.")
            start_val = st.number_input(f"Start {component} (%)", 0.0, 100.0, 0.0, 1.0)
            end_val   = st.number_input(f"End {component} (%)",   0.0, 100.0, 100.0, 1.0)

        n_steps = st.slider("Number of steps", 5, 30, 10)

        if st.button("▶️ Run Sensitivity Analysis", disabled=run_disabled):
            df_result, baseline_scores, final_scores = run_sensitivity(
                st.session_state.twin, param_type, chosen_construct, component,
                start_val, end_val, n_steps
            )
            # Store sensitivity range for insights
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

# ---- Official Static Map (Zoomable) ----
st.subheader("🗺️ Tagoloan River Basin (Official DOST-PAGASA Map)")
try:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get("https://pubfiles.pagasa.dost.gov.ph/pagasaweb/images/basins/tagoloan-river-basin.jpg", headers=headers)
    img = Image.open(BytesIO(resp.content))
    fig_map = px.imshow(img)
    fig_map.update_layout(
        title="Tagoloan River Basin (Official DOST-PAGASA Map)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    fig_map.update_xaxes(showgrid=False)
    fig_map.update_yaxes(showgrid=False)
    st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})
    st.caption("Official DOST-PAGASA map. Drag to pan, scroll to zoom.")
    if st.button("📥 Download Map (interactive HTML)"):
        html_str = fig_map.to_html(include_plotlyjs='cdn')
        st.download_button("Save map.html", html_str, "tagoloan_river_basin.html", "text/html")
except Exception as e:
    st.warning("Official map could not be loaded. Please check the image URL.")

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
