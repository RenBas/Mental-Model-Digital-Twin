import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
import hashlib
import datetime

# ==============================================================================
# 1. DATA CLASSES
# ==============================================================================

class MentalModelNode:
    def __init__(self, name, challenge, acceptance, commitment, corr_c_a, corr_a_ch, corr_c_ch):
        self.name = name
        self.baseline_cac = {'Challenge': challenge, 'Acceptance': acceptance, 'Commitment': commitment}
        self.correlation_matrix = np.array([[1.0, corr_c_a, corr_c_ch],
                                            [corr_c_a, 1.0, corr_a_ch],
                                            [corr_c_ch, corr_a_ch, 1.0]])
        self.current_score = self._compute_score()
        self.previous_delta = 0.0

    def _compute_score(self):
        c = self.baseline_cac['Challenge']
        a = self.baseline_cac['Acceptance']
        co = self.baseline_cac['Commitment']
        raw = (a + co) / 2.0 + (50.0 - c) / 2.0
        return max(0.0, min(100.0, raw))

    def update_cac(self, challenge=None, acceptance=None, commitment=None):
        if challenge is not None:
            self.baseline_cac['Challenge'] = challenge
        if acceptance is not None:
            self.baseline_cac['Acceptance'] = acceptance
        if commitment is not None:
            self.baseline_cac['Commitment'] = commitment
        self.current_score = self._compute_score()

class MentalModelEdge:
    def __init__(self, source_name, target_name, coefficient, r_square, nodes_dict):
        self.source_name = source_name
        self.target_name = target_name
        self.coefficient = coefficient
        self.r_square = r_square

class ResidentAgent:
    def __init__(self, agent_id, cluster_name, node_states, cac_states, archetype):
        self.agent_id = agent_id
        self.cluster_name = cluster_name
        self.node_states = node_states
        self.cac_states = cac_states
        self.archetype = archetype
        self.has_relocated = False
        self.will_evacuate = False
        self.is_adapting_in_place = False
        self.is_resisting_lgu = False

    def evaluate_decisions(self, flood_severity=0.0, lgu_demolition_threat=False):
        desire = self.node_states.get("Desire for relocation", 50.0)
        feasibility = self.node_states.get("Feasibility of relocation", 50.0)
        assistance = self.node_states.get("Assistance for relocation", 50.0)
        fear = self.node_states.get("Fear of housing demolition", 50.0)
        family_ties = self.node_states.get("Family history and identity", 50.0)

        relocation_utility = (desire*0.35) + (feasibility*0.30) + (assistance*0.25) - (fear*0.05) - (family_ties*0.05)
        relocation_utility = max(0, min(100, relocation_utility))
        if relocation_utility >= self.archetype.relocation_threshold and not self.has_relocated:
            self.has_relocated = True

        coping = self.node_states.get("Coping during flooding", 50.0)
        prevention = self.node_states.get("Prevention and flooding", 50.0)
        lgu_view = self.node_states.get("Viewpoints towards LGU", 50.0)
        evac_utility = (coping*0.4) + (prevention*0.3) + (lgu_view*0.3)
        self.will_evacuate = evac_utility >= (50.0 + 20.0 * (1.0 - flood_severity))

        preference = self.node_states.get("Preference and adaptation", 50.0)
        self.is_adapting_in_place = (not self.has_relocated) and (preference >= 40.0)

        if lgu_demolition_threat:
            rights = self.node_states.get("Rights to live in the area", 50.0)
            resist_utility = (rights*0.5) + (family_ties*0.3) + (fear*0.2)
            self.is_resisting_lgu = resist_utility >= self.archetype.resistance_threshold
        else:
            self.is_resisting_lgu = False

class ClusterArchetype:
    def __init__(self, name, population_ratio, node_baseline_scores, dominant_driver,
                 relocation_threshold=35.0, resistance_threshold=35.0):
        self.name = name
        self.population_ratio = population_ratio
        self.node_baseline_scores = node_baseline_scores
        self.dominant_driver = dominant_driver
        self.relocation_threshold = relocation_threshold
        self.resistance_threshold = resistance_threshold

# ==============================================================================
# 2. K‑MEANS NOMENCLATURE ENGINE
# ==============================================================================
PERSONA_MAP = {
    "Prevention and flooding": "Proactive Risk Mitigators",
    "Coping during flooding": "Resilient Flood Adapters",
    "Flooding and Family": "Family-Centric Survivors",
    "Desire for relocation": "Transition-Seeking Residents",
    "Preference and adaptation": "Adaptive Planners",
    "Feasibility of relocation": "Pragmatic Relocators",
    "Fear of housing demolition": "Anxious Property Owners",
    "Viewpoints towards LGU": "Civic-Engaged Constituents",
    "Assistance for relocation": "Resource-Dependent Pragmatists",
    "Rights to live in the area": "Rights-Asserting Residents",
    "Living in the disaster area": "Resilient Locals",
    "Family history and identity": "Heritage-Bound Residents"
}

def generate_lgu_cluster_name(centroids, col_map):
    macro_scores = {}
    for node, clean in col_map.items():
        ac = centroids.get(f"{clean}_Acceptance", 0)
        co = centroids.get(f"{clean}_Commitment", 0)
        macro_scores[node] = ac + co
    sorted_nodes = sorted(macro_scores.items(), key=lambda x: x[1], reverse=True)
    top_node = sorted_nodes[0][0]
    return PERSONA_MAP.get(top_node, "Balanced Community Segment"), top_node

# ==============================================================================
# 3. POPULATION GENERATOR (accepts optional seed)
# ==============================================================================
class PopulationGenerator:
    def __init__(self, col_map):
        self.col_map = col_map
        self.node_names = list(col_map.keys())

    def generate_population(self, total_population_size, cluster_profiles, engine_nodes, seed=None):
        agents = []
        if total_population_size == 0 or not cluster_profiles:
            return agents
        rng = np.random.RandomState(seed)
        current_agent_id = 1
        cluster_counts = {}
        assigned_total = 0
        sorted_clusters = sorted(cluster_profiles.items(), key=lambda x: x[1].population_ratio, reverse=True)

        for name, profile in sorted_clusters:
            if name == sorted_clusters[-1][0]:
                count = total_population_size - assigned_total
            else:
                count = round(total_population_size * profile.population_ratio)
                assigned_total += count
            cluster_counts[name] = count

        for name, profile in cluster_profiles.items():
            for _ in range(cluster_counts[name]):
                individual_cac_states = {}
                agent_macro_states = {}

                is_cac_profile = any("_Challenge" in k for k in profile.node_baseline_scores.keys())

                if is_cac_profile:
                    for node_name, clean_name in self.col_map.items():
                        for cac in ['Challenge', 'Acceptance', 'Commitment']:
                            key = f"{clean_name}_{cac}"
                            baseline = profile.node_baseline_scores.get(key, 33.3)
                            noise = rng.normal(0, 5.0)
                            individual_cac_states[key] = max(0.0, min(100.0, baseline + noise))

                    for node_name, clean_name in self.col_map.items():
                        ac_val = individual_cac_states.get(f"{clean_name}_Acceptance", 33.3)
                        co_val = individual_cac_states.get(f"{clean_name}_Commitment", 33.3)
                        macro_base = (ac_val + co_val) / 2.0
                        macro_deviation = engine_nodes[node_name].current_score - 50.0
                        noise = rng.normal(0, 3.0)
                        agent_macro_states[node_name] = max(0.0, min(100.0, macro_base + macro_deviation + noise))
                else:
                    for node_name in self.node_names:
                        baseline = profile.node_baseline_scores.get(node_name, 50.0)
                        macro_deviation = engine_nodes[node_name].current_score - 50.0
                        noise = rng.normal(0, 5.0)
                        agent_macro_states[node_name] = max(0.0, min(100.0, baseline + macro_deviation + noise))
                        clean = self.col_map[node_name]
                        individual_cac_states[f"{clean}_Challenge"] = 33.3
                        individual_cac_states[f"{clean}_Acceptance"] = 33.3
                        individual_cac_states[f"{clean}_Commitment"] = 33.3

                agents.append(ResidentAgent(current_agent_id, name, agent_macro_states, individual_cac_states, profile))
                current_agent_id += 1
        return agents

# ==============================================================================
# 4. COMMUNITY ANALYTICS
# ==============================================================================
class CommunityAnalytics:
    def __init__(self, agents_list, node_names):
        self.agents = agents_list
        self.node_names = node_names
        self.total_population = len(agents_list)
        self._build_dataframe()

    def _build_dataframe(self):
        data = []
        for agent in self.agents:
            row = {
                'Agent_ID': agent.agent_id,
                'Cluster': agent.cluster_name,
                'Has_Relocated': agent.has_relocated,
                'Will_Evacuate': agent.will_evacuate,
                'Is_Adapting': agent.is_adapting_in_place,
                'Is_Resisting': agent.is_resisting_lgu
            }
            for node_name in self.node_names:
                row[node_name] = agent.node_states.get(node_name, 0.0)
            data.append(row)
        self.df = pd.DataFrame(data)

    def get_behavioral_metrics(self):
        if self.total_population == 0:
            return {
                "Total Population": 0,
                "Projected to Relocate (%)": 0.0,
                "Evacuating (%)": 0.0,
                "Adapting In-Place (%)": 0.0,
                "Resisting LGU (%)": 0.0
            }
        return {
            "Total Population": self.total_population,
            "Projected to Relocate (%)": (self.df['Has_Relocated'].sum() / self.total_population) * 100,
            "Evacuating (%)": (self.df['Will_Evacuate'].sum() / self.total_population) * 100,
            "Adapting In-Place (%)": (self.df['Is_Adapting'].sum() / self.total_population) * 100,
            "Resisting LGU (%)": (self.df['Is_Resisting'].sum() / self.total_population) * 100
        }

    def get_advanced_metrics(self):
        if self.total_population == 0:
            return {
                "Proactive Preparedness (%)": 0.0,
                "LGU Trust & Cooperation (%)": 0.0,
                "Heritage-Based Refusal (%)": 0.0,
                "Demolition Anxiety (%)": 0.0,
                "Relocation Readiness (%)": 0.0
            }
        proactive = self.df[
            (self.df["Prevention and flooding"] > 60) & 
            (self.df["Coping during flooding"] > 60)
        ].shape[0]
        proactive_pct = (proactive / self.total_population) * 100

        trust_coop = self.df[
            (self.df["Viewpoints towards LGU"] > 50) & 
            (self.df["Assistance for relocation"] > 30)
        ].shape[0]
        trust_pct = (trust_coop / self.total_population) * 100

        heritage_refusal = self.df[
            (self.df["Has_Relocated"] == False) & 
            (self.df["Family history and identity"] > 48)
        ].shape[0]
        heritage_pct = (heritage_refusal / self.total_population) * 100

        anxiety = self.df[self.df["Fear of housing demolition"] > 50].shape[0]
        anxiety_pct = (anxiety / self.total_population) * 100

        ready = self.df[
            (self.df["Has_Relocated"] == True) & 
            (self.df["Preference and adaptation"] > 60)
        ].shape[0]
        readiness_pct = (ready / self.total_population) * 100

        return {
            "Proactive Preparedness (%)": proactive_pct,
            "LGU Trust & Cooperation (%)": trust_pct,
            "Heritage-Based Refusal (%)": heritage_pct,
            "Demolition Anxiety (%)": anxiety_pct,
            "Relocation Readiness (%)": readiness_pct
        }

    def get_cluster_breakdown(self):
        if self.total_population == 0:
            return pd.DataFrame(columns=['Cluster', 'Population Count', 'Projected to Relocate %', 'Evacuating %', 'Adapting %', 'Resisting %'])
        cluster_stats = self.df.groupby('Cluster').agg(
            Count=('Agent_ID', 'count'),
            Relocated=('Has_Relocated', 'mean'),
            Evacuating=('Will_Evacuate', 'mean'),
            Adapting=('Is_Adapting', 'mean'),
            Resisting=('Is_Resisting', 'mean')
        )
        cluster_stats[['Relocated','Evacuating','Adapting','Resisting']] *= 100
        cluster_stats.columns = ['Population Count', 'Projected to Relocate %',
                                 'Evacuating %', 'Adapting %', 'Resisting %']
        return cluster_stats.reset_index()

    def get_cac_averages(self, col_map):
        cac_avgs = {}
        for node, clean in col_map.items():
            for cac in ['Challenge', 'Acceptance', 'Commitment']:
                key = f"{clean}_{cac}"
                vals = [getattr(agent, 'cac_states', {}).get(key, 0) for agent in self.agents]
                cac_avgs[f"{node} ({cac})"] = np.mean(vals) if vals else 0
        return cac_avgs

# ==============================================================================
# 5. UNIFIED DIGITAL TWIN
# ==============================================================================
class DigitalTwin:
    def __init__(self, nodes, edges, cluster_profiles, total_population, flood_severity, lgu_threat,
                 damping_factor=0.5, seed=None):
        self.nodes = nodes
        self.edges = edges
        self.damping_factor = damping_factor
        self.flood_severity = flood_severity
        self.lgu_threat = lgu_threat
        self.cluster_profiles = cluster_profiles if cluster_profiles else {}
        self.total_population = total_population
        self.step_count = 0
        self.history = []

        self.incoming_edges = {name: [] for name in self.nodes.keys()}
        for edge in self.edges:
            self.incoming_edges[edge.target_name].append((edge.source_name, edge.coefficient))

        self.generator = PopulationGenerator(col_map)
        self.agents = self.generator.generate_population(
            self.total_population, self.cluster_profiles, self.nodes, seed=seed
        )
        for agent in self.agents:
            agent.evaluate_decisions(self.flood_severity, self.lgu_threat)
        self.analytics = CommunityAnalytics(self.agents, list(self.nodes.keys()))

    def step(self):
        if self.total_population == 0:
            return
        net_influences = {name: 0.0 for name in self.nodes}
        for target_name, sources in self.incoming_edges.items():
            for source_name, coeff in sources:
                deviation = self.nodes[source_name].current_score - 50.0
                net_influences[target_name] += coeff * deviation

        for node_name, node in self.nodes.items():
            delta = net_influences[node_name] * self.damping_factor
            node.current_score += delta
            node.current_score = max(0.0, min(100.0, node.current_score))
            node.previous_delta = delta

        for agent in self.agents:
            for node_name in self.nodes.keys():
                base_score = self.nodes[node_name].current_score
                noise = np.random.normal(0, 3.0)
                agent.node_states[node_name] = max(0.0, min(100.0, base_score + noise))

        for agent in self.agents:
            agent.evaluate_decisions(self.flood_severity, self.lgu_threat)

        self.analytics = CommunityAnalytics(self.agents, list(self.nodes.keys()))
        self.history.append(self.analytics.get_behavioral_metrics())
        self.step_count += 1

    def reset(self, new_flood_severity=None, new_lgu_threat=None):
        for node in self.nodes.values():
            if hasattr(node, '_compute_score'):
                node.current_score = node._compute_score()
            else:
                c = node.baseline_cac['Challenge']
                a = node.baseline_cac['Acceptance']
                co = node.baseline_cac['Commitment']
                node.current_score = max(0.0, min(100.0, (a + co) / 2.0 + (50.0 - c) / 2.0))
            node.previous_delta = 0.0
        if new_flood_severity is not None:
            self.flood_severity = new_flood_severity
        if new_lgu_threat is not None:
            self.lgu_threat = new_lgu_threat
        self.agents = self.generator.generate_population(
            self.total_population, self.cluster_profiles, self.nodes
        )
        for agent in self.agents:
            agent.evaluate_decisions(self.flood_severity, self.lgu_threat)
        self.analytics = CommunityAnalytics(self.agents, list(self.nodes.keys()))
        self.history = []
        self.step_count = 0

    def update_population_size(self, new_size):
        self.total_population = new_size
        self.reset()

    def update_cluster_profiles(self, new_profiles):
        self.cluster_profiles = new_profiles
        self.reset()

    def get_metrics(self):
        return self.analytics.get_behavioral_metrics()

    def get_advanced_metrics(self):
        return self.analytics.get_advanced_metrics()

# ==============================================================================
# 6. BUILD BASE MODEL DATA
# ==============================================================================
@st.cache_resource
def build_base_nodes_and_edges():
    nodes_data = {
        "Prevention and flooding": (23.6, 36.3, 40.1, 0.652, 0.662, 0.827),
        "Coping during flooding": (23.5, 30.0, 46.5, 0.725, 0.886, 0.710),
        "Flooding and Family": (34.4, 22.9, 42.7, 0.505, 0.200, 0.386),
        "Desire for relocation": (16.8, 50.3, 32.8, 0.766, 0.685, 0.505),
        "Preference and adaptation": (27.1, 36.4, 36.5, 0.780, 0.544, 0.577),
        "Feasibility of relocation": (60.3, 6.8, 32.9, 0.537, 0.701, 0.741),
        "Fear of housing demolition": (30.2, 46.0, 23.8, 0.681, 0.797, 0.856),
        "Viewpoints towards LGU": (17.2, 33.3, 49.4, 0.782, 0.444, 0.211),
        "Assistance for relocation": (47.3, 36.2, 16.5, 0.831, 0.621, 0.672),
        "Rights to live in the area": (16.8, 33.8, 49.4, 0.608, 0.542, 0.300),
        "Living in the disaster area": (30.3, 29.8, 39.9, 0.485, 0.535, 0.862),
        "Family history and identity": (34.1, 23.0, 42.9, 0.601, 0.414, 0.506)
    }
    nodes = {}
    for name, vals in nodes_data.items():
        ch, ac, co, corr_ca, corr_ac, corr_cc = vals
        nodes[name] = MentalModelNode(name, ch, ac, co, corr_ca, corr_ac, corr_cc)

    edge_data = [
        ("Prevention and flooding", "Living in the disaster area", 0.715, 0.502),
        ("Coping during flooding", "Prevention and flooding", 1.003, 0.692),
        ("Coping during flooding", "Flooding and Family", 0.734, 0.756),
        ("Flooding and Family", "Desire for relocation", 1.109, 0.894),
        ("Desire for relocation", "Preference and adaptation", 0.941, 0.933),
        ("Fear of housing demolition", "Coping during flooding", 1.021, 0.932),
        ("Feasibility of relocation", "Desire for relocation", 0.961, 0.969),
        ("Preference and adaptation", "Fear of housing demolition", 0.867, 0.780),
        ("Preference and adaptation", "Feasibility of relocation", 1.020, 0.939),
        ("Fear of housing demolition", "Rights to live in the area", 0.805, 0.817),
        ("Rights to live in the area", "Viewpoints towards LGU", 0.931, 0.774),
        ("Viewpoints towards LGU", "Fear of housing demolition", 0.976, 0.846),
        ("Viewpoints towards LGU", "Assistance for relocation", 0.943, 0.718),
        ("Assistance for relocation", "Feasibility of relocation", 0.991, 0.939),
        ("Family history and identity", "Rights to live in the area", 0.717, 0.698),
        ("Living in the disaster area", "Rights to live in the area", 0.595, 0.337),
        ("Assistance for relocation", "Desire for relocation", 0.980, 0.957),
        ("Family history and identity", "Living in the disaster area", 0.373, 0.198)
    ]
    edges = [MentalModelEdge(s, t, c, r2, nodes) for s, t, c, r2 in edge_data]
    return nodes, edges

base_nodes, base_edges = build_base_nodes_and_edges()

col_map = {
    "Prevention and flooding": "Prevention_flooding",
    "Coping during flooding": "Coping_flooding",
    "Flooding and Family": "Flooding_Family",
    "Desire for relocation": "Desire_relocation",
    "Preference and adaptation": "Preference_adaptation",
    "Feasibility of relocation": "Feasibility_relocation",
    "Fear of housing demolition": "Fear_demolition",
    "Viewpoints towards LGU": "Viewpoints_LGU",
    "Assistance for relocation": "Assistance_relocation",
    "Rights to live in the area": "Rights_live_area",
    "Living in the disaster area": "Living_disaster_area",
    "Family history and identity": "Family_history_identity"
}

# ==============================================================================
# 7. PAGASA ADVISORY LIVE GAUGE FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=3600)
def fetch_pagasa_advisory():
    url = "https://www.pagasa.dost.gov.ph/flood/tagoloan"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, "html.parser")
        content_div = soup.find("div", class_="entry-content")
        if not content_div:
            content_div = soup.find("body")
        text = content_div.get_text(separator="\n") if content_div else soup.get_text()

        issued_match = re.search(r"ISSUED AT (.*?)\n", text)
        timestamp_str = issued_match.group(1).strip() if issued_match else "Unknown time"

        obs_match = re.search(r"OBSERVED 24-HR RAINFALL:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
        forecast_match = re.search(r"FORECAST 24-HR RAINFALL:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
        observed_text = obs_match.group(1).strip() if obs_match else ""
        forecast_text = forecast_match.group(1).strip() if forecast_match else ""

        rainfall_desc = forecast_text if forecast_text else observed_text
        if not rainfall_desc:
            return None

        severity = None
        label = "Unknown"
        color = "grey"
        if re.search(r"\b(torrential|intense|extreme)\b", rainfall_desc, re.IGNORECASE):
            severity = 0.90
            color = "red"
            label = "Torrential Rain (Red)"
        elif re.search(r"\bheavy\b", rainfall_desc, re.IGNORECASE):
            severity = 0.65
            color = "orange"
            label = "Heavy Rain (Orange)"
        elif re.search(r"\bmoderate\b", rainfall_desc, re.IGNORECASE):
            severity = 0.35
            color = "#FFC107"
            label = "Moderate Rain (Yellow)"
        elif re.search(r"\blight\b", rainfall_desc, re.IGNORECASE):
            severity = 0.10
            color = "green"
            label = "Light Rain (Green)"
        else:
            return None

        return {
            "severity": severity,
            "color": color,
            "label": label,
            "timestamp": timestamp_str,
            "raw_text": rainfall_desc
        }
    except Exception:
        return None

# ==============================================================================
# 8. STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="Tagoloan Flood-Prone Communities Digital Twin", layout="wide")

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def apply_dark_mode():
    if st.session_state.dark_mode:
        dark_css = """
        <style>
            .stApp, .main, .stSidebar, .st-expander, .stMetric, .stDataFrame {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
            div[data-testid="metric-container"] {
                background-color: #2C2C2C;
                border: 1px solid #444;
                border-radius: 8px;
            }
            div[data-testid="metric-container"] > label {
                color: #E0E0E0 !important;
            }
            .stSidebar {
                background-color: #252525;
            }
            .streamlit-expanderHeader {
                color: #E0E0E0;
            }
            .stButton>button {
                background-color: #3A3A3A;
                color: #E0E0E0;
                border: 1px solid #555;
            }
            .stSlider>div>div>div>div {
                color: #E0E0E0;
            }
            .caption, .stCaption {
                color: #B0B0B0;
            }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)

apply_dark_mode()

# ---------- Session State Initialisation ----------
if 'twin' not in st.session_state:
    st.session_state.twin = DigitalTwin(
        nodes=base_nodes,
        edges=base_edges,
        cluster_profiles={},
        total_population=0,
        flood_severity=0.0,
        lgu_threat=False
    )

defaults = {
    'data_calibrated': False,
    'respondent_clusters': None,
    'current_barangay': "All Barangays",
    'raw_data': None,
    'disable_flashing': False,
    'use_pagasa_auto': True,
    'pagasa_severity': None,
    'prev_pagasa_severity': None,
    'baseline_params': None,
    'log_entries': [],
    'auto_log': True,
    'prev_k_mode': "Auto (silhouette)",
    'dark_mode': False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- Sidebar ----------
with st.sidebar:
    st.toggle("🌙 Dark Mode", key="dark_mode")

    st.header("📊 Data Upload & Calibration")
    cac_vars = ['Challenge', 'Acceptance', 'Commitment']
    csv_columns = ['Respondent_Name', 'Barangay_Name']
    for node, clean_name in col_map.items():
        for cac in cac_vars:
            csv_columns.append(f"{clean_name}_{cac}")

    template_df = pd.DataFrame({col: ['Sample Name', 'Sample Barangay'] if col in ['Respondent_Name', 'Barangay_Name'] else [50, 60] for col in csv_columns})
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
            index=0,
            key="k_mode"
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
                    numeric_cols = [c for c in csv_columns if c not in ['Respondent_Name', 'Barangay_Name']]
                    df_num = df_filtered[numeric_cols]
                    scaler = MinMaxScaler(feature_range=(0, 100))
                    X_scaled = scaler.fit_transform(df_num)

                    if k_mode == "Auto (silhouette)":
                        best_k = 3
                        best_sil = -1
                        for k in range(2, min(6, len(df_filtered))):
                            km = KMeans(n_clusters=k, random_state=42, n_init=10)
                            labels = km.fit_predict(X_scaled)
                            sil = silhouette_score(X_scaled, labels)
                            if sil > best_sil:
                                best_sil = sil
                                best_k = k
                        chosen_k = best_k
                    elif k_mode == "Fixed (3 clusters)":
                        chosen_k = 3
                    else:
                        chosen_k = manual_k

                    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
                    final_labels = kmeans.fit_predict(X_scaled)

                    df_labeled = df_filtered.copy()
                    df_labeled['Cluster'] = final_labels
                    st.session_state.respondent_clusters = df_labeled

                    df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
                    df_scaled['Cluster'] = final_labels

                    new_profiles = {}
                    for i in range(chosen_k):
                        cluster_data = df_scaled[df_scaled['Cluster'] == i]
                        ratio = len(cluster_data) / len(df_scaled)
                        centroids = cluster_data[numeric_cols].mean().to_dict()
                        base_name, driver = generate_lgu_cluster_name(centroids, col_map)

                        final_name = base_name
                        counter = 1
                        while final_name in new_profiles:
                            final_name = f"{base_name} (Segment {counter})"
                            counter += 1

                        new_profiles[final_name] = ClusterArchetype(
                            name=final_name,
                            population_ratio=ratio,
                            node_baseline_scores=centroids,
                            dominant_driver=driver
                        )

                    data_hash = hashlib.md5(df_filtered.to_csv(index=False).encode()).hexdigest()
                    seed = int(data_hash, 16) % (2**32)

                    baseline_severity = st.session_state.twin.flood_severity

                    st.session_state.twin = DigitalTwin(
                        nodes=base_nodes,
                        edges=base_edges,
                        cluster_profiles=new_profiles,
                        total_population=len(df_filtered),
                        flood_severity=baseline_severity,
                        lgu_threat=False,
                        seed=seed
                    )
                    st.session_state.data_calibrated = True
                    st.session_state.baseline_params = dict(
                        cluster_profiles=new_profiles,
                        total_population=len(df_filtered),
                        flood_severity=baseline_severity,
                        lgu_threat=False,
                        seed=seed
                    )
                    st.success(f"Baseline established. K = {chosen_k}, population = {len(df_filtered)}. Ready for simulation or advisory updates.")
                    st.rerun()
    else:
        st.info("📝 Upload a 38‑column CSV to begin.")

    st.markdown("---")
    st.header("🎛️ Simulation Controls")
    steps_to_run = st.selectbox("Steps to run", list(range(1, 11)), index=0)
    run_disabled = (st.session_state.twin.total_population == 0)
    if st.button("▶️ Run", use_container_width=True, type="primary", disabled=run_disabled):
        for _ in range(steps_to_run):
            st.session_state.twin.step()
        st.rerun()
    if st.button("♻️ Restore Baseline", use_container_width=True, disabled=run_disabled):
        if st.session_state.baseline_params is not None:
            bp = st.session_state.baseline_params
            st.session_state.twin = DigitalTwin(
                nodes=base_nodes,
                edges=base_edges,
                cluster_profiles=bp['cluster_profiles'],
                total_population=bp['total_population'],
                flood_severity=bp['flood_severity'],
                lgu_threat=bp['lgu_threat'],
                seed=bp['seed']
            )
            st.session_state.use_pagasa_auto = True
            st.success("Baseline restored.")
        else:
            st.warning("No baseline available. Upload data first.")
        st.rerun()

    st.markdown("---")
    st.header("⚙️ Settings")
    pop_disabled = (len(st.session_state.twin.cluster_profiles) == 0)
    new_pop = st.slider("Total Residents to Simulate", 10, 5000, st.session_state.twin.total_population,
                        key="pop_slider", disabled=pop_disabled,
                        help="Changing this value immediately regenerates the population.")
    if not pop_disabled and new_pop != st.session_state.twin.total_population:
        st.session_state.twin.update_population_size(new_pop)
        st.warning(f"Population updated to {new_pop}. Simulation reset, history cleared.")
        st.rerun()

    # PAGASA Advisory section
    st.markdown("---")
    st.subheader("🌧️ PAGASA Advisory (Tagoloan River Basin)")
    advisory = fetch_pagasa_advisory()
    if advisory:
        pagasa_severity = advisory["severity"]
        ts = advisory["timestamp"]
        label = advisory["label"]
        color = advisory["color"]
    else:
        pagasa_severity = None
        ts = "Advisory unavailable"
        label = "No data"
        color = "grey"

    if st.session_state.use_pagasa_auto and pagasa_severity is not None:
        st.session_state.twin.flood_severity = pagasa_severity

    if st.session_state.auto_log and pagasa_severity is not None:
        if st.session_state.prev_pagasa_severity is None:
            st.session_state.prev_pagasa_severity = pagasa_severity
        elif pagasa_severity != st.session_state.prev_pagasa_severity:
            metrics = st.session_state.twin.get_metrics()
            advanced = st.session_state.twin.get_advanced_metrics()
            log_entry = {
                "Barangay": st.session_state.current_barangay,
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Severity": f"{pagasa_severity:.2f} ({label})",
                "Total Population": metrics['Total Population'],
                "Relocate %": metrics['Projected to Relocate (%)'],
                "Evacuating %": metrics['Evacuating (%)'],
                "Resisting LGU %": metrics['Resisting LGU (%)'],
                "Proactive %": advanced['Proactive Preparedness (%)'],
                "LGU Trust %": advanced['LGU Trust & Cooperation (%)'],
                "Heritage Refusal %": advanced['Heritage-Based Refusal (%)'],
                "Demolition Anxiety %": advanced['Demolition Anxiety (%)'],
                "Relocation Readiness %": advanced['Relocation Readiness (%)']
            }
            st.session_state.log_entries.append(log_entry)
            st.session_state.prev_pagasa_severity = pagasa_severity

    flash_class = ""
    if pagasa_severity is not None and not st.session_state.disable_flashing:
        if pagasa_severity >= 0.60:
            flash_class = "fast-flash"
        else:
            flash_class = "slow-flash"
    warning_icon = "⚠️" if pagasa_severity and pagasa_severity >= 0.60 else ""

    gauge_html = f"""
    <div class="live-gauge {flash_class}" style="margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:bold; color: {color};">{warning_icon} {label}</span>
            <span style="font-size:0.8rem; color: #888;">Last updated: {ts}</span>
        </div>
        <div style="width:100%; height:25px; background:linear-gradient(to right, green 0%, green 25%, #FFC107 25%, #FFC107 50%, orange 50%, orange 75%, red 75%, red 100%); border-radius:5px; position:relative; margin-top:5px; border: 1px solid #555;">
            <div style="position:absolute; left:{pagasa_severity*100 if pagasa_severity else 0}%; top:-4px; width:4px; height:33px; background:black; border-radius:2px;"></div>
        </div>
        <p style="margin-top:5px; font-size:11px; color: #aaa;">Based on PAGASA advisory (updated daily). Simulation slider below.</p>
    </div>
    <style>
        @keyframes slowPulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        @keyframes fastPulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
            100% {{ opacity: 1; }}
        }}
        .slow-flash {{
            animation: slowPulse 2s infinite;
        }}
        .fast-flash {{
            animation: fastPulse 0.8s infinite;
        }}
    </style>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

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

    if flood_sev <= 0.25:
        rainfall_mm = flood_sev * 40
        label = "Light rain"
        color_code = "green"
    elif flood_sev <= 0.50:
        rainfall_mm = 10 + (flood_sev - 0.25) * 80
        label = "Moderate rain (Yellow)"
        color_code = "#FFC107"
    elif flood_sev <= 0.75:
        rainfall_mm = 30 + (flood_sev - 0.50) * 120
        label = "Heavy rain (Orange)"
        color_code = "orange"
    else:
        rainfall_mm = 60 + (flood_sev - 0.75) * 160
        label = "Torrential rain (Red)"
        color_code = "red"
    st.markdown(f"🌧️ **{rainfall_mm:.0f} mm** – {label}")
    gauge_html = f"""
    <div style="width:100%; height:30px; background:linear-gradient(to right, green 0%, green 25%, #FFC107 25%, #FFC107 50%, orange 50%, orange 75%, red 75%, red 100%); border-radius:5px; position:relative; margin-bottom:10px;">
        <div style="position:absolute; left:{flood_sev*100}%; top:-5px; width:4px; height:40px; background:black; border-radius:2px;"></div>
    </div>
    <p style="margin-top:5px; font-size:12px;">🟢 0-10 mm &nbsp; 🟡 10-30 mm &nbsp; 🟠 30-60 mm &nbsp; 🔴 >60 mm</p>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)
    lgu_threat = st.toggle("LGU Demolition Threat", value=st.session_state.twin.lgu_threat)
    if flood_sev != st.session_state.twin.flood_severity or lgu_threat != st.session_state.twin.lgu_threat:
        if st.button("Apply Environmental Triggers", disabled=run_disabled):
            st.session_state.twin.reset(new_flood_severity=flood_sev, new_lgu_threat=lgu_threat)
            st.rerun()

    # Inactive Water‑Level Gauge
    st.markdown("---")
    st.subheader("🌊 Tagoloan River Water Level")
    wl_gauge = f"""
    <div style="filter: grayscale(100%); opacity: 0.5; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:bold; color: grey;">🔒 No Data</span>
            <span style="font-size:0.8rem; color: #888;">Awaiting data feed</span>
        </div>
        <div style="width:100%; height:25px; background:linear-gradient(to right, green 0%, green 37%, #FFC107 37%, #FFC107 62%, orange 62%, orange 87%, red 87%, red 100%); border-radius:5px; position:relative; margin-top:5px; border: 1px solid #555;">
            <div style="position:absolute; left:0%; top:-4px; width:4px; height:33px; background:black; border-radius:2px;"></div>
        </div>
        <p style="margin-top:5px; font-size:11px; color: #aaa;">Water level data not yet available – awaiting PAGASA PREDICT / MDRRMO</p>
    </div>
    """
    st.markdown(wl_gauge, unsafe_allow_html=True)

    st.markdown("---")
    st.header("🎚️ LGU Intervention Sliders")
    st.caption("Adjust the three CAC components directly. Changing them updates the psychological baseline. "
               "Click **'🧬 Rebuild Population with Current Settings'** to see the new behavioral outcomes.")
    with st.expander("Expand to modify constructs", expanded=False):
        for node_name, node in st.session_state.twin.nodes.items():
            st.markdown(f"**{node_name}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                new_ch = st.slider(f"Challenge", 0.0, 100.0, float(node.baseline_cac['Challenge']), key=f"{node_name}_ch")
            with col2:
                new_ac = st.slider(f"Acceptance", 0.0, 100.0, float(node.baseline_cac['Acceptance']), key=f"{node_name}_ac")
            with col3:
                new_co = st.slider(f"Commitment", 0.0, 100.0, float(node.baseline_cac['Commitment']), key=f"{node_name}_co")
            node.baseline_cac['Challenge'] = new_ch
            node.baseline_cac['Acceptance'] = new_ac
            node.baseline_cac['Commitment'] = new_co
            node.current_score = max(0.0, min(100.0, (new_ac + new_co) / 2.0 + (50.0 - new_ch) / 2.0))
    if st.button("🧬 Rebuild Population with Current Settings", use_container_width=True, disabled=run_disabled):
        st.session_state.twin.reset()
        st.rerun()

# ---------- Main Dashboard ----------
barangay_title = st.session_state.current_barangay if st.session_state.current_barangay != "All Barangays" else "Municipal"
st.title(f"Tagoloan Flood-Prone Communities Digital Twin ({barangay_title})")
st.markdown("*Municipality of Tagoloan, Misamis Oriental*")

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
    st.caption("Official DOST-PAGASA map. Use the toolbar or mouse wheel to zoom and pan.")
except Exception as e:
    st.warning("Official map could not be loaded. Please check the image URL.")

# ---- Basic Behavioral Outcomes ----
st.subheader("Community Behavioral Outcomes")
if pop == 0:
    st.info("Upload and calibrate your survey data to populate the indicators.")
else:
    st.caption(f"Realistic baselines from survey CAC data. Current flood severity: **{flood_sev:.2f}** – {label}")
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

st.markdown("---")

st.subheader("Socio-Psychological Network Graph")
if pop == 0:
    st.info("Upload and calibrate data to display the psychological network graph.")
else:
    G = nx.DiGraph()
    for name, node in twin.nodes.items():
        G.add_node(name, score=node.current_score)
    for edge in twin.edges:
        G.add_edge(edge.source_name, edge.target_name, weight=edge.coefficient)

    pos = nx.spring_layout(G, k=0.35, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'),
                            hoverinfo='text',
                            text=[f"{edge.source_name} → {edge.target_name}<br>Regression coeff.: {edge.coefficient:+.3f}<br>R²: {edge.r_square:.3f}" for edge in twin.edges],
                            mode='lines')
    node_x, node_y, node_text, node_color = [], [], [], []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(f"{n}<br>Score: {d['score']:.1f}")
        node_color.append(d['score'])
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=list(G.nodes()), textposition="bottom center",
                            hovertext=node_text, hoverinfo='text',
                            marker=dict(showscale=True, colorscale='Viridis', reversescale=True,
                                        color=node_color, size=25,
                                        colorbar=dict(thickness=10, title='Score')))
    fig_net = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='12 Nodes & 18 Causal Pathways', showlegend=False,
                                         margin=dict(b=20,l=5,r=5,t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig_net, use_container_width=True)

    high_nodes = [n for n, d in G.nodes(data=True) if d['score'] > 60]
    low_nodes = [n for n, d in G.nodes(data=True) if d['score'] < 40]
    if high_nodes:
        st.caption(f"🔵 **Strong drivers in {barangay_title}:** {', '.join(high_nodes)} show high intensity, which can be leveraged for positive behavioral change.")
    if low_nodes:
        st.caption(f"🟡 **Weak areas in {barangay_title}:** {', '.join(low_nodes)} are psychologically fragile; interventions here may have the greatest impact.")
    if not high_nodes and not low_nodes:
        st.caption(f"⚪ **Balanced profile in {barangay_title}:** all nodes are in the moderate range, suggesting a stable psychological landscape.")

st.markdown("---")

st.subheader("Behavioral Distribution by Cluster")
cluster_df = twin.analytics.get_cluster_breakdown()
if pop == 0:
    st.info("No clusters yet. Upload and recalibrate data.")
else:
    st.caption("Overall metrics are the weighted average of these per‑cluster percentages. Population counts shown in the table below.")

    fig_cluster = go.Figure()
    behaviors = ['Projected to Relocate %', 'Evacuating %', 'Adapting %', 'Resisting %']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    for i, beh in enumerate(behaviors):
        fig_cluster.add_trace(go.Bar(
            x=cluster_df['Cluster'],
            y=cluster_df[beh],
            name=beh,
            text=[f"{v:.1f}%" for v in cluster_df[beh]],
            textposition='outside',
            marker_color=colors[i]
        ))
    fig_cluster.update_layout(barmode='group', title='Behavioral Distribution by Cluster',
                              yaxis_title='Percentage', height=450)
    st.plotly_chart(fig_cluster, use_container_width=True)

    dominating_clusters = []
    for _, row in cluster_df.iterrows():
        if row['Projected to Relocate %'] > 50 or row['Evacuating %'] > 60 or row['Resisting %'] > 30:
            dominating_clusters.append(row['Cluster'])
    if dominating_clusters:
        st.caption(f"🔍 **Clusters driving {barangay_title} outcomes:** {', '.join(dominating_clusters)} have notably high behavioral percentages. Targeting these groups can shift aggregate outcomes significantly.")
    else:
        st.caption(f"🔍 **Balanced cluster behavior in {barangay_title}:** no single cluster dominates the outcomes, indicating a mixed community profile.")

    st.markdown("**Cluster populations:**")
    pop_summary = cluster_df[['Cluster', 'Population Count']].set_index('Cluster')
    st.dataframe(pop_summary.T, use_container_width=True)
    st.caption("The sum of these counts equals the total population shown above.")

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

st.markdown("---")
st.subheader("CAC Breakdown (Bubble: Commitment)")
cac_avgs = twin.analytics.get_cac_averages(col_map)
scatter = [{"Construct": node, "Challenge": cac_avgs[f"{node} (Challenge)"],
            "Acceptance": cac_avgs[f"{node} (Acceptance)"],
            "Commitment": cac_avgs[f"{node} (Commitment)"]} for node in col_map]
if pop == 0:
    st.info("CAC data will appear after calibration.")
else:
    fig_cac = px.scatter(pd.DataFrame(scatter), x="Challenge", y="Acceptance",
                         size="Commitment", color="Construct", size_max=60,
                         title="Community CAC Profile")
    st.plotly_chart(fig_cac, use_container_width=True)

    high_challenge = [d['Construct'] for d in scatter if d['Challenge'] > 60]
    low_acceptance = [d['Construct'] for d in scatter if d['Acceptance'] < 40]
    if high_challenge:
        st.caption(f"🔴 **High Challenge in {barangay_title}:** {', '.join(high_challenge)} – residents perceive significant barriers, requiring policy support to lower perceived difficulty.")
    if low_acceptance:
        st.caption(f"🟠 **Low Acceptance in {barangay_title}:** {', '.join(low_acceptance)} – these areas lack community buy‑in; awareness campaigns or incentives may be needed.")
    if not high_challenge and not low_acceptance:
        st.caption(f"⚪ **Balanced CAC profile in {barangay_title}:** all constructs are within moderate range, indicating a generally stable psychological state.")

st.markdown("---")
st.subheader("Policy Insights & Actionable Recommendations")
if pop == 0:
    st.info("Insights will appear after data is loaded and calibrated.")
else:
    insight_parts = []
    insight_parts.append(
        f"**{barangay_title}** analysis covers **{pop:,} residents** (uploaded survey data). "
        f"Under the current PAGASA advisory ({label}, severity {flood_sev:.2f}), "
        f"**{reloc_pct:.1f}%** are projected to relocate, **{evac_pct:.1f}%** are prepared to evacuate, "
        f"and **{resist_pct:.1f}%** show resistance to LGU initiatives. "
        f"Advanced indicators: proactive preparedness {advanced['Proactive Preparedness (%)']:.1f}%, "
        f"LGU trust {advanced['LGU Trust & Cooperation (%)']:.1f}%, "
        f"heritage refusal {advanced['Heritage-Based Refusal (%)']:.1f}%, "
        f"demolition anxiety {advanced['Demolition Anxiety (%)']:.1f}%, "
        f"relocation readiness {advanced['Relocation Readiness (%)']:.1f}%."
    )

    if reloc_pct > 30:
        insight_parts.append(
            f"🔴 **High relocation pressure ({reloc_pct:.1f}%):** This indicates strong desire or feasibility for moving, "
            "likely driven by elevated 'Desire for relocation' and 'Feasibility of relocation' scores in the network graph. "
            "If unmanaged, this could lead to unplanned out‑migration or strain on resettlement programs."
        )
    elif reloc_pct > 10:
        insight_parts.append(
            f"🟡 **Moderate relocation interest ({reloc_pct:.1f}%):** A notable segment considers moving, "
            "but most residents prefer to stay. The cluster breakdown can pinpoint which groups are relocation‑ready "
            "so that assistance can be targeted without triggering unnecessary displacement."
        )
    else:
        insight_parts.append(
            f"🟢 **Low relocation intent ({reloc_pct:.1f}%):** The majority of residents are rooted in place. "
            "Policies should focus on in‑situ adaptation, livelihood support, and strengthening local capacities "
            "rather than resettlement."
        )

    if evac_pct > 80:
        insight_parts.append(
            f"🔵 **Very high evacuation readiness ({evac_pct:.1f}%):** Almost the entire community is willing to evacuate "
            "when warned. This reflects strong 'Coping during flooding' and 'Prevention and flooding' scores. "
            "Maintain early warning systems and conduct regular drills to sustain this readiness."
        )
    elif evac_pct > 50:
        insight_parts.append(
            f"🟡 **Moderate evacuation readiness ({evac_pct:.1f}%):** More than half would evacuate, but a significant "
            "minority remains hesitant. The cluster chart highlights which groups are least likely to evacuate; "
            "targeted awareness campaigns and incentives could raise this figure."
        )
    else:
        insight_parts.append(
            f"🔴 **Low evacuation readiness ({evac_pct:.1f}%):** A majority of residents may not respond to evacuation orders. "
            "This is a critical gap in disaster preparedness. Strengthen 'Coping during flooding' and 'Viewpoints towards LGU' "
            "through community engagement and trust‑building, as suggested by the network graph."
        )

    if resist_pct > 20:
        insight_parts.append(
            f"🟠 **Significant LGU resistance ({resist_pct:.1f}%):** Friction is apparent, especially if a demolition threat "
            "is active. The CAC profile likely shows high 'Fear of housing demolition' and low 'Viewpoints towards LGU'. "
            "Immediate action: issue housing tenure guarantees and open a transparent dialogue with affected clusters."
        )
    elif resist_pct > 5:
        insight_parts.append(
            f"🟡 **Low‑to‑moderate resistance ({resist_pct:.1f}%):** A small fraction resists LGU efforts. "
            "Monitor the clusters that contribute to this resistance; even a few vocal opponents can escalate tensions."
        )
    else:
        insight_parts.append(
            f"🟢 **Negligible resistance ({resist_pct:.1f}%):** The community is largely cooperative, "
            "providing a favourable environment for new programs and policies."
        )

    if advanced['Proactive Preparedness (%)'] > 60:
        insight_parts.append("🛠️ **High proactive preparedness** – community champions exist; engage them as partners.")
    if advanced['LGU Trust & Cooperation (%)'] < 40:
        insight_parts.append("🤝 **Low LGU trust** – major barrier to policy rollouts; invest in trust‑building before launching new programs.")
    if advanced['Heritage-Based Refusal (%)'] > 30:
        insight_parts.append("🏡 **High heritage‑based refusal** – relocation programs must include psychosocial and cultural components, not just financial aid.")
    if advanced['Demolition Anxiety (%)'] > 30:
        insight_parts.append("⚠️ **Elevated demolition anxiety** – deploy MHPSS and issue clear housing guarantees immediately.")
    if advanced['Relocation Readiness (%)'] > 20:
        insight_parts.append("🚀 **Relocation readiness high** – a pilot relocation program can succeed quickly if targeted at these early adopters.")

    if twin.nodes["Fear of housing demolition"].current_score > 60:
        insight_parts.append(
            "⚠️ **Elevated fear of demolition** – The network shows this node is particularly hot (high score). "
            "It will likely amplify resistance and reduce relocation willingness. Address it with clear communication "
            "about housing security before any infrastructure project."
        )
    if twin.nodes["Viewpoints towards LGU"].current_score < 40:
        insight_parts.append(
            "⚠️ **Low LGU trust** – Trust is a multiplier across all behaviours. "
            "Implement visible, participatory projects to improve this score; the network shows it directly influences "
            "'Assistance for relocation' and 'Fear of housing demolition'."
        )

    if reloc_pct < 10 and evac_pct > 80 and resist_pct < 5:
        insight_parts.append(
            "✅ **Optimal profile:** This barangay exhibits high resilience with low resistance and relocation pressure. "
            "The primary strategy is to **maintain current interventions** and monitor for any shifts, especially if "
            "environmental conditions change."
        )
    elif reloc_pct > 30 and evac_pct < 50:
        insight_parts.append(
            "⚠️ **Dual vulnerability:** Many want to leave but few would evacuate in an emergency. "
            "This contradictory pattern suggests deep‑seated distrust or fear. Prioritise building evacuation capacity "
            "while simultaneously offering voluntary, dignified relocation options."
        )

    st.markdown(" ".join(insight_parts))
    st.caption(
        "🔗 **How to use these insights:** The network graph identifies which psychological drivers to adjust, "
        "the cluster breakdown shows exactly which groups drive each behaviour, and the CAC bubble chart reveals "
        "the underlying community profile. Adjust the intervention sliders or environmental triggers, then "
        "click 'Rebuild Population' or 'Run' to test policy scenarios."
    )

st.markdown("---")
st.subheader("Simulation Timeline")
st.caption("Shows how the three key macro‑metrics evolve step‑by‑step, revealing the delayed effects of interventions – vital for policy planning.")
if len(twin.history) > 1:
    hist = pd.DataFrame(twin.history)
    hist['Step'] = range(len(hist))
    hist_melt = hist.melt(id_vars='Step', value_vars=['Projected to Relocate (%)','Evacuating (%)','Resisting LGU (%)'],
                          var_name='Metric', value_name='Percentage')
    fig_time = px.line(hist_melt, x='Step', y='Percentage', color='Metric',
                       title='Macro-Metrics Over Time', markers=True)
    st.plotly_chart(fig_time, use_container_width=True)
    recent = hist.tail(3)
    relocate_trend = recent['Projected to Relocate (%)'].diff().mean()
    evac_trend = recent['Evacuating (%)'].diff().mean()
    resist_trend = recent['Resisting LGU (%)'].diff().mean()
    if relocate_trend > 0.5:
        st.caption(f"📈 **{barangay_title}:** Relocation is trending upward – if this continues, resettlement demand may exceed current capacity.")
    elif relocate_trend < -0.5:
        st.caption(f"📉 **{barangay_title}:** Relocation is declining, suggesting that interventions are reducing the desire to leave.")
    if evac_trend > 0.5:
        st.caption(f"📈 **{barangay_title}:** Evacuation readiness is improving; maintain current awareness efforts.")
    elif evac_trend < -0.5:
        st.caption(f"📉 **{barangay_title}:** Evacuation readiness is dropping – review early warning effectiveness.")
    if resist_trend > 0.3:
        st.caption(f"⚠️ **{barangay_title}:** Resistance is growing; potential trigger events (like demolition threats) may be escalating tensions.")
else:
    st.info("Run at least two steps to see the timeline.")

st.markdown("---")

# Logging and download
st.subheader("📋 Advisory Response Log")
log_disabled = (pop == 0)
if st.button("📌 Log Current Snapshot", disabled=log_disabled):
    log_entry = {
        "Barangay": st.session_state.current_barangay,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Severity": f"{flood_sev:.2f} ({label})",
        "Total Population": pop,
        "Relocate %": reloc_pct,
        "Evacuating %": evac_pct,
        "Resisting LGU %": resist_pct,
        "Proactive %": advanced['Proactive Preparedness (%)'],
        "LGU Trust %": advanced['LGU Trust & Cooperation (%)'],
        "Heritage Refusal %": advanced['Heritage-Based Refusal (%)'],
        "Demolition Anxiety %": advanced['Demolition Anxiety (%)'],
        "Relocation Readiness %": advanced['Relocation Readiness (%)']
    }
    st.session_state.log_entries.append(log_entry)
    st.success("Snapshot logged.")

if st.session_state.log_entries:
    log_df = pd.DataFrame(st.session_state.log_entries)
    st.dataframe(log_df, use_container_width=True)
    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Advisory History (CSV)", csv, "advisory_log.csv", "text/csv")

st.checkbox("Auto‑log on advisory change", key="auto_log")
