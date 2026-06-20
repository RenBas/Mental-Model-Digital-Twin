import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

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
        self.will_evacuate = evac_utility >= (50.0 - (flood_severity * 30.0))

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
# 2. K-MEANS NOMENCLATURE ENGINE
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
# 3. POPULATION GENERATOR
# ==============================================================================
class PopulationGenerator:
    def __init__(self, col_map):
        self.col_map = col_map
        self.node_names = list(col_map.keys())

    def generate_population(self, total_population_size, cluster_profiles, engine_nodes):
        agents = []
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
                            noise = np.random.normal(0, 5.0)
                            individual_cac_states[key] = max(0.0, min(100.0, baseline + noise))

                    for node_name, clean_name in self.col_map.items():
                        ac_val = individual_cac_states.get(f"{clean_name}_Acceptance", 33.3)
                        co_val = individual_cac_states.get(f"{clean_name}_Commitment", 33.3)
                        macro_base = (ac_val + co_val) / 2.0
                        macro_deviation = engine_nodes[node_name].current_score - 50.0
                        noise = np.random.normal(0, 3.0)
                        agent_macro_states[node_name] = max(0.0, min(100.0, macro_base + macro_deviation + noise))
                else:
                    for node_name in self.node_names:
                        baseline = profile.node_baseline_scores.get(node_name, 50.0)
                        macro_deviation = engine_nodes[node_name].current_score - 50.0
                        noise = np.random.normal(0, 5.0)
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

    def get_cluster_breakdown(self):
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
                 damping_factor=0.5):
        self.nodes = nodes
        self.edges = edges
        self.damping_factor = damping_factor
        self.flood_severity = flood_severity
        self.lgu_threat = lgu_threat
        self.cluster_profiles = cluster_profiles
        self.total_population = total_population
        self.step_count = 0
        self.history = []

        self.incoming_edges = {name: [] for name in self.nodes.keys()}
        for edge in self.edges:
            self.incoming_edges[edge.target_name].append((edge.source_name, edge.coefficient))

        self.generator = PopulationGenerator(col_map)
        self.agents = self.generator.generate_population(
            self.total_population, self.cluster_profiles, self.nodes
        )
        for agent in self.agents:
            agent.evaluate_decisions(self.flood_severity, self.lgu_threat)
        self.analytics = CommunityAnalytics(self.agents, list(self.nodes.keys()))

    def step(self):
        # System dynamics
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

        # Agent coupling
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
            # Fallback if node is from an older class without _compute_score
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

def get_mock_clusters():
    return {
        "Transition-Seeking Residents": ClusterArchetype(
            "Transition-Seeking Residents", 0.45,
            {n: 65.0 for n in base_nodes.keys()}, "Desire for relocation"),
        "Anxious Property Owners": ClusterArchetype(
            "Anxious Property Owners", 0.30,
            {n: 40.0 for n in base_nodes.keys()}, "Fear of housing demolition"),
        "Resilient Flood Adapters": ClusterArchetype(
            "Resilient Flood Adapters", 0.25,
            {n: 50.0 for n in base_nodes.keys()}, "Coping during flooding")
    }

# ==============================================================================
# 7. STREAMLIT APP
# ==============================================================================
st.set_page_config(page_title="Tagoloan Flood-Prone Communities Digital Twin", layout="wide")

# ---------- Session State ----------
if 'twin' not in st.session_state:
    st.session_state.twin = DigitalTwin(
        nodes=base_nodes,
        edges=base_edges,
        cluster_profiles=get_mock_clusters(),
        total_population=140,   # baseline N for Sitio Dal-og
        flood_severity=0.3,
        lgu_threat=False
    )
    st.session_state.data_calibrated = False
    st.session_state.respondent_clusters = None

# ---------- Sidebar ----------
with st.sidebar:
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
                if len(df_raw) < 3:
                    st.error("Need at least 3 respondents for clustering.")
                else:
                    if st.button("🔄 Recalibrate Model with Uploaded CAC Data", use_container_width=True):
                        with st.spinner("Running K-Means & determining optimal clusters..."):
                            numeric_cols = [c for c in csv_columns if c not in ['Respondent_Name', 'Barangay_Name']]
                            df_num = df_raw[numeric_cols]
                            scaler = MinMaxScaler(feature_range=(0, 100))
                            X_scaled = scaler.fit_transform(df_num)

                            # Auto-determine K via silhouette score
                            best_k = 3
                            best_sil = -1
                            for k in range(2, min(6, len(df_raw))):
                                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                                labels = km.fit_predict(X_scaled)
                                sil = silhouette_score(X_scaled, labels)
                                if sil > best_sil:
                                    best_sil = sil
                                    best_k = k

                            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                            final_labels = kmeans.fit_predict(X_scaled)

                            df_labeled = df_raw.copy()
                            df_labeled['Cluster'] = final_labels
                            st.session_state.respondent_clusters = df_labeled

                            df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
                            df_scaled['Cluster'] = final_labels

                            new_profiles = {}
                            for i in range(best_k):
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

                            st.session_state.twin.update_cluster_profiles(new_profiles)
                            st.session_state.twin.update_population_size(len(df_raw))
                            st.session_state.data_calibrated = True
                            st.success(f"Model recalibrated! Optimal K = {best_k}, population = {len(df_raw)}. Ready to simulate.")
                            st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("📝 Using mock data (N=140). Upload a 38‑column CSV to calibrate with real survey data.")

    st.markdown("---")
    st.header("🎛️ Simulation Controls")

    steps_to_run = st.selectbox("Steps to run", list(range(1, 11)), index=0)

    if st.button("▶️ Run", use_container_width=True, type="primary"):
        for _ in range(steps_to_run):
            st.session_state.twin.step()
        st.rerun()

    if st.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state.twin.reset()
        st.rerun()

    st.markdown("---")
    st.header("⚙️ Settings")
    pop_size = st.slider("Total Residents to Simulate", 10, 5000,
                         st.session_state.twin.total_population,
                         key="pop_slider",
                         help="After calibration, this matches the CSV size. Change to project to a larger population.")
    if pop_size != st.session_state.twin.total_population:
        if st.button("Apply Population Change"):
            st.session_state.twin.update_population_size(pop_size)
            st.rerun()

    flood_sev = st.slider("Flood Severity", 0.0, 1.0, st.session_state.twin.flood_severity, 0.1)
    lgu_threat = st.toggle("LGU Demolition Threat", value=st.session_state.twin.lgu_threat)
    if flood_sev != st.session_state.twin.flood_severity or lgu_threat != st.session_state.twin.lgu_threat:
        if st.button("Apply Environmental Triggers"):
            st.session_state.twin.reset(new_flood_severity=flood_sev, new_lgu_threat=lgu_threat)
            st.rerun()

    st.markdown("---")
    st.header("🎚️ LGU Intervention Sliders")
    st.caption("Adjust the three CAC components directly. Changing them recalculates the node score instantly.")
    with st.expander("Expand to modify constructs", expanded=False):
        for node_name, node in st.session_state.twin.nodes.items():
            st.markdown(f"**{node_name}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                new_ch = st.slider(f"Challenge", 0.0, 100.0, float(node.baseline_cac['Challenge']),
                                   key=f"{node_name}_ch")
            with col2:
                new_ac = st.slider(f"Acceptance", 0.0, 100.0, float(node.baseline_cac['Acceptance']),
                                   key=f"{node_name}_ac")
            with col3:
                new_co = st.slider(f"Commitment", 0.0, 100.0, float(node.baseline_cac['Commitment']),
                                   key=f"{node_name}_co")
            if (new_ch != node.baseline_cac['Challenge'] or
                new_ac != node.baseline_cac['Acceptance'] or
                new_co != node.baseline_cac['Commitment']):
                node.update_cac(challenge=new_ch, acceptance=new_ac, commitment=new_co)

# ---------- Main Dashboard ----------
st.title("Tagoloan Flood-Prone Communities Digital Twin")
st.markdown("*Municipality of Tagoloan, Misamis Oriental*")

twin = st.session_state.twin
metrics = twin.get_metrics()

st.subheader("Community Behavioral Outcomes")
st.caption("Realistic baselines from survey CAC data. Use 'Run' to see dynamic changes.")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Population", f"{metrics['Total Population']:,}")
col2.metric("Projected to Relocate", f"{metrics['Projected to Relocate (%)']:.1f}%")
col3.metric("Evacuating", f"{metrics['Evacuating (%)']:.1f}%")
col4.metric("Resisting LGU", f"{metrics['Resisting LGU (%)']:.1f}%")

st.markdown("---")

st.subheader("Socio-Psychological Network Graph")
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
                        hoverinfo='none', mode='lines')
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
st.caption("🎨 Node colors: yellow (low) → purple (high).")

st.markdown("---")

st.subheader("Behavioral Distribution by Cluster")
cluster_df = twin.analytics.get_cluster_breakdown()
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
    st.info("Showing mock profiles. Upload and recalibrate for data‑driven names.")
profiles = [{"Cluster": name, "Share": f"{p.population_ratio*100:.1f}%",
             "Dominant Driver": p.dominant_driver}
            for name, p in twin.cluster_profiles.items()]
st.dataframe(pd.DataFrame(profiles), use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("CAC Breakdown (Bubble: Commitment)")
cac_avgs = twin.analytics.get_cac_averages(col_map)
scatter = [{"Construct": node, "Challenge": cac_avgs[f"{node} (Challenge)"],
            "Acceptance": cac_avgs[f"{node} (Acceptance)"],
            "Commitment": cac_avgs[f"{node} (Commitment)"]} for node in col_map]
fig_cac = px.scatter(pd.DataFrame(scatter), x="Challenge", y="Acceptance",
                     size="Commitment", color="Construct", size_max=60,
                     title="Community CAC Profile")
st.plotly_chart(fig_cac, use_container_width=True)
st.caption(
    "ℹ️ The intervention sliders set the **baseline CAC values for each psychological construct**, "
    "which directly determine the node's current score. The bubble chart shows the **average CAC components "
    "across the entire agent population** (including cluster‑specific profiles and individual noise), "
    "so the numbers may differ from the sliders. This chart reflects the community's aggregated psychological state."
)

st.markdown("---")
st.subheader("Policy Insights & Actionable Recommendations")
insights = []
if metrics['Projected to Relocate (%)'] > 30:
    insights.append("🔴 **High relocation desire** – Consider expanding resettlement and financial assistance programs.")
if metrics['Evacuating (%)'] < 50:
    insights.append("🟠 **Low evacuation readiness** – Strengthen early warning systems and conduct drills.")
if metrics['Resisting LGU (%)'] > 20:
    insights.append("🟡 **Significant LGU resistance** – Improve transparency and dialogue with affected clusters.")
if twin.nodes["Fear of housing demolition"].current_score > 60:
    insights.append("⚠️ **Elevated fear of demolition** – Issue clear housing security guarantees.")
if twin.nodes["Viewpoints towards LGU"].current_score < 40:
    insights.append("⚠️ **Low LGU trust** – Implement visible community projects and participatory planning.")
if not insights:
    insights.append("✅ **Community resilience is stable** – Maintain current engagement and monitoring.")
for ins in insights:
    st.markdown(ins)

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
else:
    st.info("Run at least two steps to see the timeline.")

st.markdown("---")
st.caption("Calibrated with baseline data from Sitio Dal-og (N=140).")
