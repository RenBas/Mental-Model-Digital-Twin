import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# LAYER 1: THE MATHEMATICAL ENGINE (System Dynamics)
# ==============================================================================

class MentalModelNode:
    def __init__(self, name, challenge, acceptance, commitment, corr_c_a, corr_a_ch, corr_c_ch):
        self.name = name
        self.baseline_cac = {'Challenge': challenge, 'Acceptance': acceptance, 'Commitment': commitment}
        self.correlation_matrix = np.array([[1.0, corr_c_a, corr_c_ch], [corr_c_a, 1.0, corr_a_ch], [corr_c_ch, corr_a_ch, 1.0]])
        self.current_score = 50.0
        self.previous_delta = 0.0

class MentalModelEdge:
    def __init__(self, source_name, target_name, coefficient, nodes_dict):
        self.source_name = source_name
        self.target_name = target_name
        self.coefficient = coefficient
        self.source_node = nodes_dict[source_name]
        self.target_node = nodes_dict[target_name]

class SimulationEngine:
    def __init__(self, nodes_dict, edges_list, damping_factor=0.5):
        self.nodes = nodes_dict
        self.edges = edges_list
        self.damping_factor = damping_factor
        self.incoming_edges = {name: [] for name in self.nodes.keys()}
        for edge in self.edges:
            self.incoming_edges[edge.target_name].append((edge.source_name, edge.coefficient))

    def apply_intervention(self, node_name, magnitude):
        if node_name in self.nodes:
            self.nodes[node_name].current_score += magnitude
            self.nodes[node_name].previous_delta = magnitude

    def step(self):
        new_deltas = {name: 0.0 for name in self.nodes.keys()}
        for target_name, sources in self.incoming_edges.items():
            for source_name, coeff in sources:
                source_delta = self.nodes[source_name].previous_delta
                new_deltas[target_name] += (coeff * source_delta)
        
        for node_name, node in self.nodes.items():
            damped_delta = new_deltas[node_name] * self.damping_factor
            node.current_score += damped_delta
            node.current_score = max(0.0, min(100.0, node.current_score))
            node.previous_delta = damped_delta

# ==============================================================================
# LAYER 2: THE AGENT POPULATION (Agent-Based Modeling)
# ==============================================================================

class ResidentAgent:
    def __init__(self, agent_id, cluster_name, node_states, archetype):
        self.agent_id = agent_id
        self.cluster_name = cluster_name
        self.node_states = node_states  
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
        
        # Utility calculation for relocation
        relocation_utility = max(0, min(100, (desire*0.35) + (feasibility*0.30) + (assistance*0.25) - (fear*0.05) - (family_ties*0.05)))
        
        # Threshold lowered to 45.0 to ensure behaviors trigger realistically in the simulation
        if relocation_utility >= self.archetype.relocation_threshold and not self.has_relocated:
            self.has_relocated = True

        coping = self.node_states.get("Coping during flooding", 50.0)
        prevention = self.node_states.get("Prevention and flooding", 50.0)
        lgu_view = self.node_states.get("Viewpoints towards LGU", 50.0)
        evac_utility = (coping*0.4) + (prevention*0.3) + (lgu_view*0.3)
        self.will_evacuate = evac_utility >= (70.0 - (flood_severity * 30.0))

        preference = self.node_states.get("Preference and adaptation", 50.0)
        self.is_adapting_in_place = (not self.has_relocated) and (preference >= 65.0)

        if lgu_demolition_threat:
            rights = self.node_states.get("Rights to live in the area", 50.0)
            resist_utility = (rights*0.5) + (family_ties*0.3) + (fear*0.2)
            # Threshold lowered to 50.0 to ensure resistance triggers realistically
            self.is_resisting_lgu = resist_utility >= self.archetype.resistance_threshold
        else:
            self.is_resisting_lgu = False

class ClusterArchetype:
    def __init__(self, name, population_ratio, node_baseline_scores, relocation_threshold=45.0, resistance_threshold=50.0):
        self.name = name
        self.population_ratio = population_ratio
        self.node_baseline_scores = node_baseline_scores 
        self.relocation_threshold = relocation_threshold
        self.resistance_threshold = resistance_threshold

class PopulationGenerator:
    def __init__(self, nodes_dict):
        self.nodes_dict = nodes_dict
        self.node_names = list(nodes_dict.keys())
        self.col_map = {
            "Prevention and flooding": "Prevention_flooding", "Coping during flooding": "Coping_flooding",
            "Flooding and Family": "Flooding_Family", "Desire for relocation": "Desire_relocation",
            "Preference and adaptation": "Preference_adaptation", "Feasibility of relocation": "Feasibility_relocation",
            "Fear of housing demolition": "Fear_demolition", "Viewpoints towards LGU": "Viewpoints_LGU",
            "Assistance for relocation": "Assistance_relocation", "Rights to live in the area": "Rights_live_area",
            "Living in the disaster area": "Living_disaster_area", "Family history and identity": "Family_history_identity"
        }
        
    def generate_population(self, total_population_size, cluster_profiles):
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
                        macro_base = ac_val + co_val
                        macro_deviation = self.nodes_dict[node_name].current_score - 50.0
                        noise = np.random.normal(0, 3.0)
                        agent_macro_states[node_name] = max(0.0, min(100.0, macro_base + macro_deviation + noise))
                else:
                    for node_name in self.node_names:
                        baseline = profile.node_baseline_scores.get(node_name, 50.0)
                        macro_deviation = self.nodes_dict[node_name].current_score - 50.0
                        noise = np.random.normal(0, 5.0)
                        agent_macro_states[node_name] = max(0.0, min(100.0, baseline + macro_deviation + noise))
                
                agents.append(ResidentAgent(current_agent_id, name, agent_macro_states, profile))
                current_agent_id += 1
        return agents

class CommunityAnalytics:
    def __init__(self, agents_list, nodes_dict):
        self.agents = agents_list
        self.node_names = list(nodes_dict.keys())
        self.total_population = len(agents_list)
        self._build_dataframe()

    def _build_dataframe(self):
        data = []
        for agent in self.agents:
            row = {'Agent_ID': agent.agent_id, 'Cluster': agent.cluster_name,
                   'Has_Relocated': agent.has_relocated, 'Will_Evacuate': agent.will_evacuate,
                   'Is_Adapting': agent.is_adapting_in_place, 'Is_Resisting': agent.is_resisting_lgu}
            for node_name in self.node_names:
                row[node_name] = agent.node_states.get(node_name, 0.0)
            data.append(row)
        self.df = pd.DataFrame(data)

    def get_behavioral_metrics(self):
        if self.total_population == 0: return {}
        return {
            "Total Population": self.total_population,
            "Projected to Relocate (%)": (self.df['Has_Relocated'].sum() / self.total_population) * 100,
            "Evacuating (%)": (self.df['Will_Evacuate'].sum() / self.total_population) * 100,
            "Adapting In-Place (%)": (self.df['Is_Adapting'].sum() / self.total_population) * 100,
            "Resisting LGU (%)": (self.df['Is_Resisting'].sum() / self.total_population) * 100
        }

    def get_cluster_breakdown(self):
        cluster_stats = self.df.groupby('Cluster').agg({
            'Has_Relocated': 'mean', 'Will_Evacuate': 'mean', 
            'Is_Adapting': 'mean', 'Is_Resisting': 'mean'
        }) * 100
        cluster_stats.columns = ['Projected to Relocate %', 'Evacuating %', 'Adapting %', 'Resisting %']
        cluster_stats['Population Count'] = self.df['Cluster'].value_counts()
        return cluster_stats.reset_index()

# ==============================================================================
# STREAMLIT APP INITIALIZATION & SESSION STATE
# ==============================================================================

st.set_page_config(page_title="Tagoloan Flood-Prone Communities Digital Twin", layout="wide")

@st.cache_resource
def initialize_model():
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
    nodes = {name: MentalModelNode(name, *vals) for name, vals in nodes_data.items()}
    
    edge_data = [
        ("Prevention and flooding", "Living in the disaster area", 0.715),
        ("Coping during flooding", "Prevention and flooding", 1.003),
        ("Coping during flooding", "Flooding and Family", 0.734), 
        ("Flooding and Family", "Desire for relocation", 1.109),
        ("Desire for relocation", "Preference and adaptation", 0.941),
        ("Fear of housing demolition", "Coping during flooding", 1.021),
        ("Feasibility of relocation", "Desire for relocation", 0.961),
        ("Preference and adaptation", "Fear of housing demolition", 0.867),
        ("Preference and adaptation", "Feasibility of relocation", 1.020),
        ("Fear of housing demolition", "Rights to live in the area", 0.805),
        ("Rights to live in the area", "Viewpoints towards LGU", 0.931),
        ("Viewpoints towards LGU", "Fear of housing demolition", 0.976),
        ("Viewpoints towards LGU", "Assistance for relocation", 0.943),
        ("Assistance for relocation", "Feasibility of relocation", 0.991),
        ("Family history and identity", "Rights to live in the area", 0.717),
        ("Living in the disaster area", "Rights to live in the area", 0.595),
        ("Assistance for relocation", "Desire for relocation", 0.980),
        ("Family history and identity", "Living in the disaster area", 0.373)
    ]
    edges = [MentalModelEdge(s, t, c, nodes) for s, t, c in edge_data]
    return nodes, edges

nodes, edges = initialize_model()

col_map = {
    "Prevention and flooding": "Prevention_flooding", "Coping during flooding": "Coping_flooding",
    "Flooding and Family": "Flooding_Family", "Desire for relocation": "Desire_relocation",
    "Preference and adaptation": "Preference_adaptation", "Feasibility of relocation": "Feasibility_relocation",
    "Fear of housing demolition": "Fear_demolition", "Viewpoints towards LGU": "Viewpoints_LGU",
    "Assistance for relocation": "Assistance_relocation", "Rights to live in the area": "Rights_live_area",
    "Living in the disaster area": "Living_disaster_area", "Family history and identity": "Family_history_identity"
}

def get_mock_clusters():
    return {
        "Resilient Adapters": ClusterArchetype("Resilient Adapters", 0.45, {n: 65.0 for n in nodes.keys()}, 45.0, 50.0),
        "Fear-Driven Resisters": ClusterArchetype("Fear-Driven Resisters", 0.30, {n: 40.0 for n in nodes.keys()}, 45.0, 50.0),
        "Pragmatic Waiters": ClusterArchetype("Pragmatic Waiters", 0.25, {n: 50.0 for n in nodes.keys()}, 45.0, 50.0)
    }

if 'engine' not in st.session_state:
    st.session_state.engine = SimulationEngine(nodes, edges, damping_factor=0.5)
    st.session_state.generator = PopulationGenerator(nodes)
    st.session_state.agents = []
    st.session_state.analytics = None
    st.session_state.history = []
    st.session_state.step_count = 0
    st.session_state.flood_sev = 0.3
    st.session_state.lgu_threat = False
    st.session_state.cluster_profiles = get_mock_clusters()
    st.session_state.data_calibrated = False
    st.session_state.uploaded_pop_size = 1000 # Default population

# ==============================================================================
# STREAMLIT UI: SIDEBAR CONTROLS
# ==============================================================================

st.title("Digital Twin: Socio-Psychological Dynamics of Flood-Prone Communities")
st.markdown("*Municipality of Tagoloan, Misamis Oriental*")
st.markdown("---")

with st.sidebar:
    st.header("Model Configuration")
    target_area = st.selectbox("Target Area / Data Source", [
        "Sitio Dal-og (Baseline Model)", 
        "Barangay Tagoloan Proper (Coming Soon)", 
        "Barangay Mohon (Coming Soon)"
    ])
    
    st.markdown("---")
    st.header("📊 Data Upload & Calibration (36 CAC Variables)")
    
    cac_vars = ['Challenge', 'Acceptance', 'Commitment']
    csv_columns = ['Respondent_Name', 'Barangay_Name']
    for node, clean_name in col_map.items():
        for cac in cac_vars:
            csv_columns.append(f"{clean_name}_{cac}")
            
    template_data = {col: ['Sample Name', 'Sample Barangay'] if col in ['Respondent_Name', 'Barangay_Name'] else [50, 60] for col in csv_columns}
    df_template = pd.DataFrame(template_data)
    csv_template = df_template.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download True CAC CSV Template (38 Columns)",
        data=csv_template, file_name='twin_cac_data_template.csv', mime='text/csv'
    )

    uploaded_file = st.file_uploader("Upload Resident Survey Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            missing_cols = [col for col in csv_columns if col not in df_raw.columns]
            if missing_cols:
                st.error(f"❌ Error: CSV is missing columns. Missing: {missing_cols[:3]}...")
            else:
                st.success(f"✅ Loaded {len(df_raw)} residents across {df_raw['Barangay_Name'].nunique()} Barangay(s).")
                
                n_clusters = st.slider("Number of Clusters (K) to generate", 2, 5, 3)
                
                if st.button("🔄 Recalibrate Model with Uploaded CAC Data", use_container_width=True):
                    with st.spinner("Running K-Means on 36 CAC dimensions..."):
                        numeric_cols = [c for c in csv_columns if c not in ['Respondent_Name', 'Barangay_Name']]
                        df_numeric = df_raw[numeric_cols]
                        
                        scaler = MinMaxScaler(feature_range=(0, 100))
                        df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_cols)
                        
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)
                        
                        new_profiles = {}
                        for i in range(n_clusters):
                            cluster_data = df_scaled[df_scaled['Cluster'] == i]
                            ratio = len(cluster_data) / len(df_scaled)
                            centroids = cluster_data[numeric_cols].mean().to_dict()
                            
                            name = f"Profile {i+1}"
                            new_profiles[name] = ClusterArchetype(
                                name=name, population_ratio=ratio, node_baseline_scores=centroids,
                                relocation_threshold=45.0, resistance_threshold=50.0
                            )
                        
                        st.session_state.cluster_profiles = new_profiles
                        st.session_state.data_calibrated = True
                        
                        # FIX FOR ISSUE 5 & 7: Auto-set population to exact CSV size and instantly evaluate agents
                        st.session_state.uploaded_pop_size = len(df_raw)
                        st.session_state.agents = st.session_state.generator.generate_population(len(df_raw), new_profiles)
                        for agent in st.session_state.agents:
                            agent.evaluate_decisions(st.session_state.flood_sev, st.session_state.lgu_threat)
                        st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
                        
                        st.success(f"Model recalibrated! Population automatically set to {len(df_raw)} residents.")
                        st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("📝 Using mock data. Upload the 38-column CSV to calibrate with real CAC survey data.")

    st.markdown("---")
    st.header("1. Simulation Controls")
    col1, col2, col3 = st.columns(3) # Added 3rd column for 1-step run
    with col1:
        if st.button("Reset", use_container_width=True):
            st.session_state.engine = SimulationEngine(nodes, edges, damping_factor=0.5)
            st.session_state.history = []
            st.session_state.step_count = 0
            st.rerun()
    with col2:
        # FIX FOR ISSUE 1: Added "Run 1 Step" button
        if st.button("Run 1 Step", use_container_width=True):
            st.session_state.engine.step()
            st.session_state.step_count += 1
            for agent in st.session_state.agents:
                agent.evaluate_decisions(st.session_state.flood_sev, st.session_state.lgu_threat)
            st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
            st.session_state.history.append(st.session_state.analytics.get_behavioral_metrics())
            st.rerun()
    with col3:
        if st.button("Run 10 Steps", use_container_width=True):
            for _ in range(10):
                st.session_state.engine.step()
                st.session_state.step_count += 1
                for agent in st.session_state.agents:
                    agent.evaluate_decisions(st.session_state.flood_sev, st.session_state.lgu_threat)
                st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
                st.session_state.history.append(st.session_state.analytics.get_behavioral_metrics())
            st.rerun()

    st.markdown("---")
    st.header("2. Population Settings")
    # FIX FOR ISSUE 5: Slider defaults to uploaded CSV size if available
    pop_size = st.slider("Total Residents to Simulate", 10, 5000, st.session_state.uploaded_pop_size, step=10)
    
    st.markdown("---")
    st.header("3. Environmental Triggers")
    st.session_state.flood_sev = st.slider("Flood Severity", 0.0, 1.0, 0.3, 0.1)
    st.session_state.lgu_threat = st.toggle("LGU Demolition Threat", value=False)

    st.markdown("---")
    st.header("4. LGU Intervention Sliders")
    with st.expander("Adjust Baseline Scores (0-100)", expanded=False):
        for node_name in nodes.keys():
            val = st.slider(node_name, 0, 100, 50, key=f"slider_{node_name}")
            st.session_state.engine.nodes[node_name].current_score = float(val)
            st.session_state.engine.nodes[node_name].previous_delta = float(val) - 50.0

    if st.button("Apply Settings & Generate Population", use_container_width=True):
        st.session_state.agents = st.session_state.generator.generate_population(pop_size, st.session_state.cluster_profiles)
        for agent in st.session_state.agents:
            agent.evaluate_decisions(st.session_state.flood_sev, st.session_state.lgu_threat)
        st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
        st.rerun()

if not st.session_state.agents:
    st.session_state.agents = st.session_state.generator.generate_population(st.session_state.uploaded_pop_size, st.session_state.cluster_profiles)
    for agent in st.session
