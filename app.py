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
    def __init__(self, source_name, target_name, coefficient, r_square, nodes_dict):
        self.source_name = source_name
        self.target_name = target_name
        self.coefficient = coefficient
        self.r_square = r_square
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
        
        relocation_utility = max(0, min(100, (desire*0.35) + (feasibility*0.30) + (assistance*0.25) - (fear*0.05) - (family_ties*0.05)))
        if relocation_utility >= self.archetype.relocation_threshold and not self.has_relocated:
            self.has_relocated = True

        coping = self.node_states.get("Coping during flooding", 50.0)
        prevention = self.node_states.get("Prevention and flooding", 50.0)
        lgu_view = self.node_states.get("Viewpoints towards LGU", 50.0)
        evac_utility = (coping*0.4) + (prevention*0.3) + (lgu_view*0.3)
        # FIX: Lowered base threshold to 50.0 so evacuation triggers more easily
        self.will_evacuate = evac_utility >= (50.0 - (flood_severity * 30.0))

        preference = self.node_states.get("Preference and adaptation", 50.0)
        # FIX: Lowered threshold to 40.0 so adaptation triggers more easily
        self.is_adapting_in_place = (not self.has_relocated) and (preference >= 40.0)

        if lgu_demolition_threat:
            rights = self.node_states.get("Rights to live in the area", 50.0)
            resist_utility = (rights*0.5) + (family_ties*0.3) + (fear*0.2)
            self.is_resisting_lgu = resist_utility >= self.archetype.resistance_threshold
        else:
            self.is_resisting_lgu = False

class ClusterArchetype:
    # FIX: Lowered default thresholds to 35.0 to make the twin highly responsive
    def __init__(self, name, population_ratio, node_baseline_scores, dominant_driver, relocation_threshold=35.0, resistance_threshold=35.0):
        self.name = name
        self.population_ratio = population_ratio
        self.node_baseline_scores = node_baseline_scores 
        self.dominant_driver = dominant_driver
        self.relocation_threshold = relocation_threshold
        self.resistance_threshold = resistance_threshold

# ==============================================================================
# REALISTIC K-MEANS NOMENCLATURE ENGINE
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
                        individual_cac_states[f"{self.col_map[node_name]}_Challenge"] = 33.3
                        individual_cac_states[f"{self.col_map[node_name]}_Acceptance"] = 33.3
                        individual_cac_states[f"{self.col_map[node_name]}_Commitment"] = 33.3
                
                agents.append(ResidentAgent(current_agent_id, name, agent_macro_states, individual_cac_states, profile))
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

    def get_cac_averages(self, col_map):
        cac_avgs = {}
        for node, clean in col_map.items():
            for cac in ['Challenge', 'Acceptance', 'Commitment']:
                key = f"{clean}_{cac}"
                # BULLETPROOF: Safely handles old agents missing cac_states
                vals = [getattr(agent, 'cac_states', {}).get(key, 0) for agent in self.agents]
                cac_avgs[f"{node} ({cac})"] = np.mean(vals) if vals else 0
        return cac_avgs

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
        "Transition-Seeking Residents": ClusterArchetype("Transition-Seeking Residents", 0.45, {n: 65.0 for n in nodes.keys()}, "Desire for relocation", 35.0, 35.0),
        "Anxious Property Owners": ClusterArchetype("Anxious Property Owners", 0.30, {n: 40.0 for n in nodes.keys()}, "Fear of housing demolition", 35.0, 35.0),
        "Resilient Flood Adapters": ClusterArchetype("Resilient Flood Adapters", 0.25, {n: 50.0 for n in nodes.keys()}, "Coping during flooding", 35.0, 35.0)
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
    st.session_state.uploaded_pop_size = 1000 
    st.session_state.pop_slider = 1000 # Initialize slider key

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
    
    st.download_button(label="📥 Download True CAC CSV Template (38 Columns)", data=csv_template, file_name='twin_cac_data_template.csv', mime='text/csv')

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
                    with st.spinner("Running K-Means & Interpreting Profiles..."):
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
                            
                            base_name, dominant_driver = generate_lgu_cluster_name(centroids, col_map)
                            
                            # BULLETPROOF: Ensure unique dictionary keys to prevent overwrites
                            final_name = base_name
                            counter = 1
                            while final_name in new_profiles:
                                final_name = f"{base_name} (Segment {counter})"
                                counter += 1
                            
                            # FIX: Pass the lowered thresholds (35.0)
                            new_profiles[final_name] = ClusterArchetype(
                                name=final_name, population_ratio=ratio, node_baseline_scores=centroids,
                                dominant_driver=dominant_driver, relocation_threshold=35.0, resistance_threshold=35.0
                            )
                        
                        st.session_state.cluster_profiles = new_profiles
                        st.session_state.data_calibrated = True
                        
                        # FIX: Force the population slider to match the exact CSV size
                        st.session_state.uploaded_pop_size = len(df_raw)
                        st.session_state.pop_slider = len(df_raw) 
                        
                        # Generate agents but DO NOT evaluate decisions yet (Starts at 0%)
                        st.session_state.agents = st.session_state.generator.generate_population(len(df_raw), new_profiles)
                        st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
                        st.session_state.history = [] 
                        st.session_state.step_count = 0
                        
                        st.success(f"Model recalibrated! Population locked to {len(df_raw)}. Click 'Run' to simulate.")
                        st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("📝 Using mock data. Upload the 38-column CSV to calibrate with real CAC survey data.")

    # ==============================================================================
    # SIMULATION CONTROLS (Dropdown + Prominent Blue Run Button)
    # ==============================================================================
    st.markdown("---")
    st.header("1. Simulation Controls")
    
    col_select, col_run = st.columns([1, 3]) 
    with col_select:
        steps_to_run = st.selectbox("Steps to Simulate", [2, 3, 4, 5, 6, 7, 8, 9, 10], index=8)
    
    with col_run:
        # type="primary" makes this a highly visible BLUE button
        if st.button(f"▶️ Run {steps_to_run} Steps", use_container_width=True, type="primary"):
            for _ in range(steps_to_run):
                st.session_state.engine.step()
                st.session_state.step_count += 1
                # NOW we evaluate decisions
                for agent in st.session_state.agents:
                    agent.evaluate_decisions(st.session_state.flood_sev, st.session_state.lgu_threat)
                st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
                st.session_state.history.append(st.session_state.analytics.get_behavioral_metrics())
            st.rerun()

    if st.button("🔄 Reset Simulation (Back to 0%)", use_container_width=True):
        st.session_state.engine = SimulationEngine(nodes, edges, damping_factor=0.5)
        st.session_state.history = []
        st.session_state.step_count = 0
        for agent in st.session_state.agents:
            agent.has_relocated = False
            agent.will_evacuate = False
            agent.is_adapting_in_place = False
            agent.is_resisting_lgu = False
        st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
        st.rerun()

    st.markdown("---")
    st.header("2. Population Settings")
    
    # FIX: Bind the slider to the session state key so it updates dynamically
    pop_size = st.slider(
        "Total Residents to Simulate", 
        10, 5000, 
        key="pop_slider", 
        step=10,
        help="Defaults to your uploaded CSV size. You can manually increase this to simulate a larger population."
    )
    
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
        st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)
        st.session_state.history = []
        st.session_state.step_count = 0
        st.rerun()

if not st.session_state.agents:
    st.session_state.agents = st.session_state.generator.generate_population(st.session_state.uploaded_pop_size, st.session_state.cluster_profiles)
    st.session_state.analytics = CommunityAnalytics(st.session_state.agents, nodes)

# ==============================================================================
# STREAMLIT UI: MAIN DASHBOARD RENDERING
# ==============================================================================

metrics = st.session_state.analytics.get_behavioral_metrics()

st.subheader("Community Behavioral Outcomes")
st.caption("*Note: Behaviors start at 0%. Use the dropdown and blue 'Run' button in the sidebar to simulate resident decision-making over time.*")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Population", f"{metrics['Total Population']:,}")
col2.metric("Projected to Relocate", f"{metrics['Projected to Relocate (%)']:.1f}%")
col3.metric("Evacuating", f"{metrics['Evacuating (%)']:.1f}%")
col4.metric("Resisting LGU", f"{metrics['Resisting LGU (%)']:.1f}%")

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Socio-Psychological Network Graph")
    G = nx.DiGraph()
    for name in nodes.keys():
        G.add_node(name, score=nodes[name].current_score)
    for edge in edges:
        G.add_edge(edge.source_name, edge.target_name, weight=edge.coefficient)
    
    pos = nx.spring_layout(G, k=0.35, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node[0]}<br>Score: {node[1]['score']:.1f}")
        node_color.append(node[1]['score'])
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[n for n in G.nodes()], 
                            textposition="bottom center", hovertext=node_text, hoverinfo='text',
                            marker=dict(showscale=True, colorscale='Viridis', reversescale=True, color=node_color, 
                                        size=25, colorbar=dict(thickness=10, title='Node Score', xanchor='left')))
    
    fig_net = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(title='12 Nodes & 18 Causal Pathways', showlegend=False, hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig_net, use_container_width=True)
    st.caption("🎨 **Graph Legend:** Node colors represent psychological intensity (0-100). **Yellow** = Low Score, **Dark Purple** = High Score. Edge thickness represents the regression coefficient.")

with col_right:
    st.subheader("Behavioral Distribution by Psychological Cluster")
    cluster_df = st.session_state.analytics.get_cluster_breakdown()
    
    df_melted = cluster_df.melt(id_vars=['Cluster'], value_vars=['Projected to Relocate %', 'Evacuating %', 'Adapting %', 'Resisting %'], 
                                var_name='Behavior', value_name='Percentage')
    
    fig_cluster = px.bar(df_melted, x='Cluster', y='Percentage', color='Behavior', barmode='group',
                         title='Behavioral Distribution by Psychological Cluster', height=450)
    st.plotly_chart(fig_cluster, use_container_width=True)

st.markdown("---")

st.subheader("Regression Analysis & Causal Pathways")
st.info("The statistical backbone of the Digital Twin. These 18 unidirectional pathways dictate how interventions ripple through the community's psychology.")
reg_data = []
for edge in edges:
    impact = "Very High" if edge.coefficient > 0.9 else "High" if edge.coefficient > 0.7 else "Moderate"
    reg_data.append({
        "Source Construct": edge.source_name,
        "Target Construct": edge.target_name,
        "Regression Coeff.": edge.coefficient,
        "R-Square": edge.r_square,
        "Impact Strength": impact
    })
df_reg = pd.DataFrame(reg_data)
st.dataframe(df_reg, use_container_width=True, hide_index=True, height=400)

st.markdown("---")

st.subheader("Socio-Psychological Cluster Profiling (K-Means Results)")
if st.session_state.data_calibrated:
    st.success("✅ The following profiles were dynamically extracted from your uploaded survey data. Nomenclature is interpreted based on the cluster's dominant psychological driver.")
else:
    st.info("ℹ️ Showing default baseline profiles. Upload a CSV and click 'Recalibrate' to generate data-driven nomenclature.")

cluster_details = []
for name, profile in st.session_state.cluster_profiles.items():
    cluster_details.append({
        "Interpreted LGU Nomenclature": name,
        "Population Share": f"{profile.population_ratio * 100:.1f}%",
        # BULLETPROOF: Safely handles old session state objects
        "Dominant Psychological Driver": getattr(profile, 'dominant_driver', 'N/A (Clear Cache)'),
        "Behavioral Thresholds": f"Relocation: {getattr(profile, 'relocation_threshold', 35.0)} | Resistance: {getattr(profile, 'resistance_threshold', 35.0)}"
    })
st.dataframe(pd.DataFrame(cluster_details), use_container_width=True, hide_index=True)

st.markdown("---")

st.subheader("CAC Framework Breakdown (Challenge, Acceptance, Commitment)")
st.info("Visualizing the 12 constructs. X-axis: Challenge, Y-axis: Acceptance. Bubble size represents Commitment.")

cac_avgs = st.session_state.analytics.get_cac_averages(col_map)
scatter_data = []
for node, clean in col_map.items():
    scatter_data.append({
        "Construct": node,
        "Challenge": cac_avgs.get(f"{node} (Challenge)", 0),
        "Acceptance": cac_avgs.get(f"{node} (Acceptance)", 0),
        "Commitment": cac_avgs.get(f"{node} (Commitment)", 0)
    })
df_scatter = pd.DataFrame(scatter_data)

fig_scatter = px.scatter(df_scatter, x="Challenge", y="Acceptance", size="Commitment", color="Construct",
                         hover_name="Construct", size_max=60, 
                         title="Community CAC Profile (Bubble Size = Commitment)")
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

st.subheader("Policy Insights & Recommendations for LGU & Superintendents")
insights = []
if nodes["Assistance for relocation"].current_score < 40:
    insights.append("🔴 **Critical Intervention Needed:** 'Assistance for relocation' is perceived as significantly challenging. The LGU must prioritize increasing tangible support.")
if nodes["Fear of housing demolition"].current_score > 60:
    insights.append("🟠 **Warning - High Resistance Risk:** 'Fear of housing demolition' is elevated. The LGU should immediately initiate clear communication campaigns and issue housing security guarantees.")
if nodes["Viewpoints towards LGU"].current_score < 40:
    insights.append("🟡 **Trust Deficit:** 'Viewpoints towards LGU' are low. Policy transparency and community engagement must be improved.")
if not insights:
    insights.append("🟢 **Status Nominal:** The community shows a balanced psychological state. Continue current engagement strategies.")
for insight in insights:
    st.markdown(insight)

st.markdown("---")

st.subheader("Simulation Timeline")
if len(st.session_state.history) > 1:
    hist_df = pd.DataFrame(st.session_state.history)
    hist_df['Step'] = range(len(hist_df))
    cols_to_plot = ['Projected to Relocate (%)', 'Evacuating (%)', 'Resisting LGU (%)']
    hist_df_melted = hist_df.melt(id_vars=['Step'], value_vars=cols_to_plot, var_name='Metric', value_name='Percentage')
    fig_time = px.line(hist_df_melted, x='Step', y='Percentage', color='Metric', title='Macro-Metrics Over Time Steps', markers=True)
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info(f"Select a number of steps from the dropdown and click the blue **'▶️ Run'** button in the sidebar to see how interventions ripple through the community over time.")

st.markdown("---")
st.caption("*Note: Current simulation parameters, regression weights, and socio-psychological clusters are calibrated using baseline data from Sitio Dal-og (N=140). Projections for other barangays assume similar socio-psychological dynamics unless localized data is uploaded.*")
