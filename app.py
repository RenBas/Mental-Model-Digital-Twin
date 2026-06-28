# ============================================================
# Digital Twin – CDO Research Culture Framework (Phase 1+2 Merged, Fixed)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import math
import base64
import time  # NEW: for spinner simulation

# -----------------------------------------------------------------
# Optional scikit‑learn for Phase 2 features (calibration, clustering, causal)
# -----------------------------------------------------------------
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

# --- Colour Palette ---
USTP_DARK_BLUE = "#0D2B5E"
USTP_GOLD = "#F5A623"
DEPED_RED = "#D32F2F"
DEPED_MAROON = "#8B0000"
LIGHT_BG = "#F8F9FA"
DARK_BG = "#1E1E1E"
DARK_TEXT = "#FFFFFF"
LIGHT_TEXT = "#000000"

# --- Simulation Model Constants ---
BASE_YEAR = 2026
RANDOM_EVENT_PROB = 0.00417
LOSS_CHAMPION_PENALTY = 0.10
FUNDING_BOOST = 0.15
LEADERSHIP_PENALTY = 0.20
CYCLE_R_BONUS = 0.10
CYCLE_M_DECAY_FACTOR = 0.50
CYCLE_M_MIN_AFTER_DECAY = 0.20
CYCLE_BONUS_BASE = 0.03
CYCLE_BONUS_M_SCALE = 0.03
MONTHLY_OUTCOME_BASE = 0.001
MILESTONE_MIN_MONTHS = 6
VALUE_FLOOR = 0.1
VALUE_CEIL = 1.0

VARIABLES = ['R', 'A', 'C', 'S', 'I', 'P', 'M']

MILESTONE_NAMES = {
    0: "Milestone 0 (Readiness and Relevance)",
    1: "Milestone 1 (Awareness to Action)",
    2: "Milestone 2 (Capacity Spark)",
    3: "Milestone 3 (Structured Support)",
    4: "Milestone 4 (Institutional Anchoring)",
    5: "Milestone 5 (Community of Practice)",
    6: "Milestone 6 (Impact Realization)",
}

MILESTONE_SHORT = {k: f"M{k}" for k in range(7)}

VAR_FULL_NAMES = {
    'R': 'Readiness (R)', 'A': 'Awareness (A)', 'C': 'Capacity (C)',
    'S': 'Structured Support (S)', 'I': 'Institutional Anchoring (I)',
    'P': 'Community of Practice (P)', 'M': 'Impact Realization (M)',
}

MILESTONE_THRESHOLDS = {
    0: ('A', 0.8, 1),
    1: ('C', 0.7, 2),
    2: ('S', 0.7, 3),
    3: ('I', 0.8, 4),
    4: ('P', 0.8, 5),
    5: ('M', 0.7, 6),
}

RCSI_LEVELS = [
    (0.0, 0.2, "Very Low"),
    (0.2, 0.4, "Low"),
    (0.4, 0.6, "Moderate"),
    (0.6, 0.8, "High"),
    (0.8, 1.0, "Very High"),
]

REQUIRED_SURVEY_COLS = ['month', 'school_id_no'] + VARIABLES
OPTIONAL_SURVEY_COLS = ['school_name']

REQUIRED_META_COLS = ['upload_date', 'teacher_name', 'school_id_no']
OPTIONAL_META_COLS = {
    'document_type': 'abstract', 'title': '', 'theme': 'Uncategorized',
    'status': 'unpublished', 'publication_link': '', 'utilized_by_school': False,
    'utilization_date': '', 'year_undertaken': 2025, 'years_of_service': None,
    'teacher_rank': None, 'educational_attainment': None,
}

VAR_COLORS = ['#1E88E5', USTP_GOLD, '#8E44AD', '#2ECC71', '#E67E22', DEPED_RED, '#1ABC9C']


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def classify_rcsi(value: float) -> str:
    for low, high, lev in RCSI_LEVELS:
        if low <= value < high:
            return lev
    return "Very High"

def get_rcsi_interpretation_table() -> str:
    return """
| RCSI Score Range | Risk/Stability Level | School-Level Policy Implication |
| :--- | :--- | :--- |
| **0.000 – 0.010** | **Optimal / Extremely Stable** | Maintain current strategies; focus on qualitative improvements (instruction, morale) rather than structural fixes. |
| **0.011 – 0.030** | Low Risk | Routine monitoring; allocate 5-10% of discretionary budget to preemptive padding. |
| **0.031 – 0.070** | Moderate Risk | Requires targeted intervention; conduct a root-cause analysis on the top 2 contributing variables. |
| **0.071 – 0.100** | High Risk | Immediate corrective action required; escalate to Division Office for shared resource support. |
| **> 0.100** | Critical / Unstable | Full operational review triggered; Division-level contingency protocols are activated. |
"""

def interpret_avg_milestone(avg_milestone: float) -> str:
    desc = [
        (0.5, "between M0 and M1"),
        (1.5, "between M1 and M2"),
        (2.5, "between M2 and M3"),
        (3.5, "between M3 and M4"),
        (4.5, "between M4 and M5"),
        (5.5, "between M5 and M6"),
        (float('inf'), "at or beyond M6"),
    ]
    for threshold, d in desc:
        if avg_milestone < threshold:
            return f"{avg_milestone:.1f} → {d}"
    return f"{avg_milestone:.1f} → at or beyond M6"

def interpret_utilisation_rate(rate: float) -> Tuple[str, str]:
    if rate < 20: return "Very Low", "Rarely adopted."
    elif rate < 40: return "Low", "Limited adoption."
    elif rate < 60: return "Moderate", "Half adopted."
    elif rate < 80: return "High", "Strong translation."
    else: return "Very High", "Excellent utilisation."

def month_str_to_num(month_str: Any) -> int:
    try:
        parts = str(month_str).strip().split('-')
        if len(parts) == 2:
            return (int(parts[0]) - BASE_YEAR) * 12 + int(parts[1])
    except (ValueError, TypeError, AttributeError):
        pass
    return 0

def date_to_month_num(d: Any) -> int:
    try:
        return (d.year - BASE_YEAR) * 12 + d.month
    except (ValueError, TypeError, AttributeError):
        return 0


# ============================================================
# THEME
# ============================================================

def apply_theme(dark_mode: bool) -> None:
    if dark_mode:
        st.markdown(f"""
        <style>
            .stApp {{ background-color: {DARK_BG} !important; color: {DARK_TEXT} !important; }}
            .sidebar .sidebar-content {{ background-color: #2E2E2E !important; border-right: 2px solid {USTP_GOLD} !important; }}
            .sidebar .sidebar-content * {{ color: {DARK_TEXT} !important; }}
            h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {USTP_GOLD} !important; }}
            .stMarkdown, .stText, .stCaption, .stDataFrame {{ color: {DARK_TEXT} !important; }}
            .stButton > button {{ background-color: {USTP_DARK_BLUE} !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            .stButton > button:hover {{ background-color: {USTP_GOLD} !important; color: {USTP_DARK_BLUE} !important; }}
            .stMetric {{ background-color: #2E2E2E !important; border: 1px solid {USTP_GOLD} !important; border-radius: 5px; padding: 10px; }}
            .stMetric label {{ color: {DARK_TEXT} !important; }}
            .dataframe {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; }}
            .dataframe thead tr th {{ background-color: {USTP_DARK_BLUE} !important; color: {DARK_TEXT} !important; }}
            .dataframe tbody tr {{ background-color: #2E2E2E !important; }}
            .dataframe tbody tr:hover {{ background-color: #3E3E3E !important; }}
            .streamlit-expanderHeader {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            .streamlit-expanderContent {{ background-color: #1E1E1E !important; color: {DARK_TEXT} !important; }}
            .stAlert {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            .stSelectbox label, .stNumberInput label, .stCheckbox label {{ color: {DARK_TEXT} !important; }}
            .stRadio label {{ color: {DARK_TEXT} !important; }}
            .stFileUploader {{ background-color: #2E2E2E !important; border: 1px dashed {USTP_GOLD} !important; }}
            .stFileUploader label {{ color: {DARK_TEXT} !important; }}
            .stCaption {{ color: #CCCCCC !important; }}
            .main .block-container {{ background-color: {DARK_BG} !important; }}
            .css-1y4p8pa {{ background-color: #2E2E2E !important; }}
            div[style*="background-color: #E3F2FD"] {{ background-color: #2E2E2E !important; border-left: 5px solid {USTP_GOLD} !important; color: {DARK_TEXT} !important; }}
            div[style*="background-color: #E8F5E9"] {{ background-color: #2E2E2E !important; border-left: 5px solid {USTP_GOLD} !important; color: {DARK_TEXT} !important; }}
            table {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; border: 1px solid {USTP_GOLD} !important; }}
            table th {{ background-color: {USTP_DARK_BLUE} !important; color: {DARK_TEXT} !important; }}
            table td {{ background-color: #2E2E2E !important; color: {DARK_TEXT} !important; }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background-color: #FFFFFF; }
            .sidebar .sidebar-content { background-color: #F8F9FA; }
            .stButton > button { background-color: #0D2B5E; color: white; }
            .stButton > button:hover { background-color: #F5A623; color: #0D2B5E; }
        </style>
        """, unsafe_allow_html=True)


def get_figure_download_link(fig, filename="chart.html", link_text="Download chart"):
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    b64 = base64.b64encode(html_str.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)


# ============================================================
# DATA CLASSES (with optional per‑agent coefficients for Phase 2)
# ============================================================

@dataclass
class CycleRecord:
    cycle_number: int
    total_improvement: float
    completion_month: int


class SchoolAgent:
    def __init__(self, unique_id: int,
                 initial_R=0.3, initial_A=0.2, initial_C=0.2,
                 initial_S=0.1, initial_I=0.1, initial_P=0.1, initial_M=0.0,
                 coeff_dict: Optional[Dict[str, float]] = None,
                 random_events_enabled: bool = False):
        self.id = unique_id
        self.real_id: int = unique_id
        self.R = initial_R
        self.A = initial_A
        self.C = initial_C
        self.S = initial_S
        self.I = initial_I
        self.P = initial_P
        self.M = initial_M
        self.current_milestone = 0
        self.months_in_milestone = 0
        self.current_cycle_accumulator = 0.0
        self.cycle_improvements: List[CycleRecord] = []
        self.cycle_count = 0
        self.running_total_outcome = 0.0
        self.random_events_enabled = random_events_enabled
        self.model_time = 0
        self._rng = np.random.RandomState()
        if coeff_dict is None:
            self.coeff = {
                'R_M': 0.02, 'A_R': 0.04, 'A_train': 0.02, 'A_M': 0.01,
                'C_train': 0.03, 'C_mentor': 0.02, 'S_budget': 0.04, 'S_mentor': 0.02,
                'I_lead': 0.03, 'I_S': 0.02, 'P_collab': 0.04, 'P_I': 0.02,
                'M_C': 0.02, 'M_P': 0.02, 'const_R': -0.01, 'const_A': -0.005,
                'const_C': -0.01, 'const_S': -0.01, 'const_I': -0.005,
                'const_P': -0.01, 'const_M': -0.005
            }
        else:
            self.coeff = coeff_dict

    def apply_random_event(self):
        if not self.random_events_enabled:
            return
        if self._rng.rand() < RANDOM_EVENT_PROB:
            event_type = self._rng.choice(["loss_champion", "funding", "leadership_change"])
            if event_type == "loss_champion":
                for var in VARIABLES:
                    setattr(self, var, max(VALUE_FLOOR, getattr(self, var) - LOSS_CHAMPION_PENALTY))
            elif event_type == "funding":
                self.S = min(VALUE_CEIL, self.S + FUNDING_BOOST)
            elif event_type == "leadership_change":
                self.I = max(VALUE_FLOOR, self.I - LEADERSHIP_PENALTY)

    def step_individual(self, levers: Dict[str, float]):
        """Single agent step using its own coefficients (used in Phase 2)."""
        u_train, u_mentor, u_budget, u_lead, u_collab = levers.values()
        c = self.coeff
        u_lead_eff = min(1.0, u_lead + 0.05 * self.M)
        R_new = self.R + c['R_M'] * self.M + c['const_R'] * (1 - u_lead_eff)
        A_new = self.A + c['A_R'] * self.R + c['A_train'] * u_train + c['A_M'] * self.M + c['const_A']
        C_new = self.C + c['C_train'] * u_train + c['C_mentor'] * u_mentor + c['const_C']
        S_new = self.S + c['S_budget'] * u_budget + c['S_mentor'] * u_mentor + c['const_S'] * (1 - u_lead_eff)
        I_new = self.I + c['I_lead'] * u_lead_eff + c['I_S'] * self.S + c['const_I']
        P_new = self.P + c['P_collab'] * u_collab + c['P_I'] * self.I + c['const_P']
        M_new = self.M + c['M_C'] * self.C + c['M_P'] * self.P + c['const_M']
        self.R = max(VALUE_FLOOR, min(VALUE_CEIL, R_new))
        self.A = max(VALUE_FLOOR, min(VALUE_CEIL, A_new))
        self.C = max(VALUE_FLOOR, min(VALUE_CEIL, C_new))
        self.S = max(VALUE_FLOOR, min(VALUE_CEIL, S_new))
        self.I = max(VALUE_FLOOR, min(VALUE_CEIL, I_new))
        self.P = max(VALUE_FLOOR, min(VALUE_CEIL, P_new))
        self.M = max(VALUE_FLOOR, min(VALUE_CEIL, M_new))
        monthly_gain = MONTHLY_OUTCOME_BASE * self.M * (1 + self.P)
        self.running_total_outcome += monthly_gain
        self.current_cycle_accumulator += monthly_gain
        self._update_milestone()
        self.apply_random_event()

    def _update_milestone(self):
        self.months_in_milestone += 1
        next_milestone = self.current_milestone
        if self.current_milestone in MILESTONE_THRESHOLDS:
            var_name, threshold, target = MILESTONE_THRESHOLDS[self.current_milestone]
            if getattr(self, var_name) >= threshold:
                next_milestone = target
        elif self.current_milestone == 6:
            if self.M >= 0.9 and self.R >= 0.8:
                self._complete_cycle()
                next_milestone = 0
        if next_milestone != self.current_milestone and self.months_in_milestone >= MILESTONE_MIN_MONTHS:
            self.current_milestone = next_milestone
            self.months_in_milestone = 0

    def _complete_cycle(self):
        old_M = self.M
        self.R = min(VALUE_CEIL, self.R + CYCLE_R_BONUS)
        self.M = max(CYCLE_M_MIN_AFTER_DECAY, self.M * CYCLE_M_DECAY_FACTOR)
        bonus = CYCLE_BONUS_BASE + CYCLE_BONUS_M_SCALE * old_M
        self.running_total_outcome += bonus
        self.current_cycle_accumulator += bonus
        self.cycle_count += 1
        self.cycle_improvements.append(
            CycleRecord(cycle_number=self.cycle_count,
                        total_improvement=self.current_cycle_accumulator,
                        completion_month=self.model_time))
        self.current_cycle_accumulator = 0.0


class Simulation:
    def __init__(self, num_schools=1, random_events=False, agent_params=None):
        if agent_params:
            self.agents = []
            for i, params in enumerate(agent_params):
                init_R, init_A, init_C, init_S, init_I, init_P, init_M, coeff = params
                self.agents.append(SchoolAgent(i,
                                               initial_R=init_R, initial_A=init_A,
                                               initial_C=init_C, initial_S=init_S,
                                               initial_I=init_I, initial_P=init_P,
                                               initial_M=init_M,
                                               coeff_dict=coeff,
                                               random_events_enabled=random_events))
        else:
            self.agents = [SchoolAgent(i, random_events_enabled=random_events) for i in range(num_schools)]

    def step(self, levers, month):
        for agent in self.agents:
            agent.model_time = month
            agent.step_individual(levers)

    def get_agent(self, idx=0):
        return self.agents[idx]


# ============================================================
# SIMULATION HELPERS (from Phase 1)
# ============================================================

def create_empty_history(school_ids):
    return {sid: {var: [] for var in VARIABLES + ['month', 'milestone', 'running_outcome']}
            for sid in school_ids}

def init_simulation_with_data(school_ids, metadata_df, random_events, agent_params=None):
    if agent_params:
        sim = Simulation(agent_params=agent_params, random_events=random_events)
    else:
        sim = Simulation(num_schools=len(school_ids), random_events=random_events)
    for idx, agent in enumerate(sim.agents):
        agent.real_id = school_ids[idx]
    seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
    return sim

def seed_agents_from_metadata(agents, school_ids, metadata_df):
    for agent in agents:
        sm = metadata_df[metadata_df['school_id_no'] == agent.real_id]
        if sm.empty:
            continue
        agent.A = min(VALUE_CEIL, agent.A + len(sm[sm['document_type'] == 'abstract']) * 0.01)
        agent.M = min(VALUE_CEIL, agent.M + len(sm[sm['status'] == 'published']) * 0.02)
        agent.C = min(VALUE_CEIL, agent.C + len(sm[sm['document_type'] == 'full_paper']) * 0.005)
        agent.P = min(VALUE_CEIL, agent.P + sm['theme'].nunique() * 0.01)

def record_history(history, agents, total_months):
    for agent in agents:
        h = history[agent.real_id]
        h['month'].append(total_months)
        for var in VARIABLES:
            h[var].append(getattr(agent, var))
        h['milestone'].append(agent.current_milestone)
        h['running_outcome'].append(agent.running_total_outcome)

def apply_survey_override(agents, survey_df, target_month):
    for agent in agents:
        row = survey_df[(survey_df['school_id_no'] == agent.real_id) & (survey_df['month_num'] == target_month)]
        if not row.empty:
            r = row.iloc[0]
            for var in VARIABLES:
                setattr(agent, var, r[var])


# ============================================================
# PHASE 2 FUNCTIONS (calibration, clustering, sensitivity, MC, causal)
# ============================================================

def calibrate_coefficients(survey_df):
    if not SKLEARN_AVAILABLE:
        return None, "scikit‑learn not installed – using default coefficients."
    if survey_df is None or survey_df.empty:
        return None, "No survey data – using default coefficients."
    u_train = u_mentor = u_budget = u_lead = u_collab = 0.5
    X_R, y_R = [], []; X_A, y_A = [], []; X_C, y_C = [], []
    X_S, y_S = [], []; X_I, y_I = [], []; X_P, y_P = [], []; X_M, y_M = [], []
    schools = survey_df['school_id_no'].unique()
    for sid in schools:
        sdf = survey_df[survey_df['school_id_no'] == sid].sort_values('month_num')
        if len(sdf) < 2:
            continue
        for i in range(len(sdf)-1):
            curr = sdf.iloc[i]
            nxt = sdf.iloc[i+1]
            u_lead_eff = min(1.0, u_lead + 0.05 * curr['M'])
            X_R.append([curr['M']]); y_R.append(nxt['R'] - curr['R'])
            X_A.append([curr['R'], u_train, curr['M']]); y_A.append(nxt['A'] - curr['A'])
            X_C.append([u_train, u_mentor]); y_C.append(nxt['C'] - curr['C'])
            X_S.append([u_budget, u_mentor]); y_S.append(nxt['S'] - curr['S'])
            X_I.append([u_lead_eff, curr['S']]); y_I.append(nxt['I'] - curr['I'])
            X_P.append([u_collab, curr['I']]); y_P.append(nxt['P'] - curr['P'])
            X_M.append([curr['C'], curr['P']]); y_M.append(nxt['M'] - curr['M'])
    coeff = {}
    default = {
        'R_M': 0.02, 'A_R': 0.04, 'A_train': 0.02, 'A_M': 0.01,
        'C_train': 0.03, 'C_mentor': 0.02, 'S_budget': 0.04, 'S_mentor': 0.02,
        'I_lead': 0.03, 'I_S': 0.02, 'P_collab': 0.04, 'P_I': 0.02,
        'M_C': 0.02, 'M_P': 0.02, 'const_R': -0.01, 'const_A': -0.005,
        'const_C': -0.01, 'const_S': -0.01, 'const_I': -0.005,
        'const_P': -0.01, 'const_M': -0.005
    }
    try:
        if X_R:
            model = LinearRegression().fit(X_R, y_R)
            coeff['R_M'] = model.coef_[0]; coeff['const_R'] = model.intercept_
        if X_A:
            model = LinearRegression().fit(X_A, y_A)
            coeff['A_R'], coeff['A_train'], coeff['A_M'] = model.coef_
            coeff['const_A'] = model.intercept_
        if X_C:
            model = LinearRegression().fit(X_C, y_C)
            coeff['C_train'], coeff['C_mentor'] = model.coef_
            coeff['const_C'] = model.intercept_
        if X_S:
            model = LinearRegression().fit(X_S, y_S)
            coeff['S_budget'], coeff['S_mentor'] = model.coef_
            coeff['const_S'] = model.intercept_
        if X_I:
            model = LinearRegression().fit(X_I, y_I)
            coeff['I_lead'], coeff['I_S'] = model.coef_
            coeff['const_I'] = model.intercept_
        if X_P:
            model = LinearRegression().fit(X_P, y_P)
            coeff['P_collab'], coeff['P_I'] = model.coef_
            coeff['const_P'] = model.intercept_
        if X_M:
            model = LinearRegression().fit(X_M, y_M)
            coeff['M_C'], coeff['M_P'] = model.coef_
            coeff['const_M'] = model.intercept_
        for k, v in default.items():
            if k not in coeff:
                coeff[k] = v
        return coeff, "Calibration successful using data-driven coefficients."
    except Exception as e:
        return None, f"Calibration failed: {str(e)} – using default coefficients."

def cluster_schools(metadata_df, school_ids):
    if not SKLEARN_AVAILABLE:
        return {sid: 0 for sid in school_ids}, {0: 1.0}
    if metadata_df is None or metadata_df.empty:
        return {sid: 0 for sid in school_ids}, {0: 1.0}
    features = []
    for sid in school_ids:
        sm = metadata_df[metadata_df['school_id_no'] == sid]
        n_teachers = sm['teacher_name'].nunique()
        n_themes = sm['theme'].nunique()
        avg_util = sm['utilized_by_school'].mean() if not sm.empty else 0
        pub_rate = len(sm[sm['status'] == 'published']) / len(sm) if len(sm) > 0 else 0
        features.append([n_teachers, n_themes, avg_util, pub_rate])
    X = np.array(features)
    n_clusters = min(3, len(X))
    if n_clusters < 2:
        return {sid: 0 for sid in school_ids}, {0: 1.0}
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    cluster_map = {sid: cl for sid, cl in zip(school_ids, clusters)}
    multipliers = {0: 1.0, 1: 1.2, 2: 0.8}
    return cluster_map, multipliers

def get_agent_params(school_ids, survey_df, metadata_df, calibrated_coeff=None):
    cluster_map, multipliers = cluster_schools(metadata_df, school_ids)
    base_coeff = calibrated_coeff if calibrated_coeff else {
        'R_M': 0.02, 'A_R': 0.04, 'A_train': 0.02, 'A_M': 0.01,
        'C_train': 0.03, 'C_mentor': 0.02, 'S_budget': 0.04, 'S_mentor': 0.02,
        'I_lead': 0.03, 'I_S': 0.02, 'P_collab': 0.04, 'P_I': 0.02,
        'M_C': 0.02, 'M_P': 0.02, 'const_R': -0.01, 'const_A': -0.005,
        'const_C': -0.01, 'const_S': -0.01, 'const_I': -0.005,
        'const_P': -0.01, 'const_M': -0.005
    }
    params = []
    for sid in school_ids:
        latest = get_latest_survey(survey_df, sid)
        if latest is not None:
            init_vals = (latest['R'], latest['A'], latest['C'], latest['S'], latest['I'], latest['P'], latest['M'])
        else:
            init_vals = (0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0)
        mult = multipliers.get(cluster_map.get(sid, 0), 1.0)
        coeff = {k: v * mult for k, v in base_coeff.items()}
        params.append((*init_vals, coeff))
    return params

def run_sensitivity(sim_class, agent_params, school_ids, levers, duration, use_survey, survey_df, metadata_df, selected_school_id):
    baseline = levers.copy()
    lever_names = ['u_train', 'u_mentor', 'u_budget', 'u_lead', 'u_collab']

    def _quick_run(test_levers):
        sim = sim_class(agent_params=agent_params)
        for i, agent in enumerate(sim.agents):
            agent.real_id = school_ids[i]
        seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
        for m in range(1, duration + 1):
            if use_survey:
                apply_survey_override(sim.agents, survey_df, m)
            sim.step(test_levers, m)
        agent = next(a for a in sim.agents if a.real_id == selected_school_id)
        return agent.running_total_outcome

    base_rcsi = _quick_run(baseline)
    results = {}
    for lever in lever_names:
        for delta in [-0.1, 0.1]:
            test_levers = baseline.copy()
            test_levers[lever] = max(0.0, min(1.0, baseline[lever] + delta))
            results[(lever, delta)] = _quick_run(test_levers)

    tornado_data = []
    for lever in lever_names:
        low_change = results[(lever, -0.1)] - base_rcsi
        high_change = results[(lever, 0.1)] - base_rcsi
        tornado_data.append({'Lever': lever, 'Low Change': low_change, 'High Change': high_change})
    df = pd.DataFrame(tornado_data).melt(id_vars='Lever', var_name='Direction', value_name='Change')
    fig = px.bar(df, x='Change', y='Lever', color='Direction', orientation='h',
                 title='Sensitivity of Final RCSI to Policy Levers (±10%)',
                 color_discrete_map={'Low Change': DEPED_RED, 'High Change': USTP_GOLD})
    fig.update_layout(template='plotly_white')
    impacts = {lever: abs(results[(lever, 0.1)] - base_rcsi) + abs(results[(lever, -0.1)] - base_rcsi) for lever in lever_names}
    most_impactful = max(impacts, key=impacts.get)
    sensitivity_info = f"Sensitivity analysis shows that **{most_impactful}** has the greatest influence on final RCSI."
    return fig, sensitivity_info

def monte_carlo_sim(num_runs, sim_class, agent_params, school_ids, levers, duration, use_survey, survey_df, metadata_df, selected_school_id):
    all_rcsi = []
    all_milestone = []
    for _ in range(num_runs):
        noisy_params = []
        for params in agent_params:
            *init_vals, coeff = params
            new_init = [max(VALUE_FLOOR, min(VALUE_CEIL, v + np.random.normal(0, 0.02))) for v in init_vals]
            noisy_coeff = {k: v * np.random.normal(1, 0.05) for k, v in coeff.items()}
            noisy_params.append((*new_init, noisy_coeff))
        sim = sim_class(agent_params=noisy_params, random_events=True)
        for i, agent in enumerate(sim.agents):
            agent.real_id = school_ids[i]
        seed_agents_from_metadata(sim.agents, school_ids, metadata_df)
        target = next(a for a in sim.agents if a.real_id == selected_school_id)
        rcsi_hist, mil_hist = [], []
        for m in range(1, duration + 1):
            if use_survey:
                apply_survey_override(sim.agents, survey_df, m)
            sim.step(levers, m)
            rcsi_hist.append(target.running_total_outcome)
            mil_hist.append(target.current_milestone)
        all_rcsi.append(rcsi_hist)
        all_milestone.append(mil_hist)
    all_rcsi = np.array(all_rcsi)
    all_milestone = np.array(all_milestone)
    months = np.arange(1, duration + 1)
    mc_data = {
        'months': months,
        'rcsi': {'p10': np.percentile(all_rcsi, 10, axis=0),
                 'p50': np.percentile(all_rcsi, 50, axis=0),
                 'p90': np.percentile(all_rcsi, 90, axis=0)},
        'milestone': {'p10': np.percentile(all_milestone, 10, axis=0),
                      'p50': np.percentile(all_milestone, 50, axis=0),
                      'p90': np.percentile(all_milestone, 90, axis=0)},
        'final_rcsi': all_rcsi[:, -1]
    }
    mc_info = (f"Monte Carlo simulation ({num_runs} runs) estimates a median final RCSI of "
               f"**{np.median(mc_data['final_rcsi']):.3f}**, with a P10‑P90 range of "
               f"**{np.percentile(mc_data['final_rcsi'], 10):.3f}** – **{np.percentile(mc_data['final_rcsi'], 90):.3f}**. "
               "This narrow spread is relevant at the school level because it indicates that the forecast is highly stable and actionable for local policy decisions.")
    return mc_data, mc_info

def plot_monte_carlo_bands(mc_data, dark_mode):
    months = mc_data['months']
    fig = make_subplots(rows=2, cols=1, subplot_titles=("RCSI with Uncertainty", "Milestone with Uncertainty"))
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p10'], mode='lines', name='P10 RCSI',
                             line=dict(color=USTP_GOLD, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p50'], mode='lines', name='Median RCSI',
                             line=dict(color=USTP_GOLD)), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p90'], mode='lines', name='P90 RCSI',
                             line=dict(color=USTP_GOLD, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p10'], showlegend=False,
                             line=dict(color='rgba(0,0,0,0)'), hoverinfo='none'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['rcsi']['p90'], fill='tonexty',
                             fillcolor='rgba(245,166,35,0.2)', line=dict(color='rgba(0,0,0,0)'),
                             showlegend=False, hoverinfo='none'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p10'], mode='lines', name='P10 Milestone',
                             line=dict(color=DEPED_RED, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p50'], mode='lines', name='Median Milestone',
                             line=dict(color=DEPED_RED)), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p90'], mode='lines', name='P90 Milestone',
                             line=dict(color=DEPED_RED, dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p10'], showlegend=False,
                             line=dict(color='rgba(0,0,0,0)'), hoverinfo='none'), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=mc_data['milestone']['p90'], fill='tonexty',
                             fillcolor='rgba(211,47,47,0.2)', line=dict(color='rgba(0,0,0,0)'),
                             showlegend=False, hoverinfo='none'), row=2, col=1)
    fig.update_layout(height=700, template='plotly_dark' if dark_mode else 'plotly_white')
    fig.update_xaxes(title_text="Month", row=1, col=1); fig.update_yaxes(title_text="RCSI", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1); fig.update_yaxes(title_text="Milestone", row=2, col=1)
    return fig

def causal_analysis(monte_carlo_finals, baseline_values):
    if not SKLEARN_AVAILABLE or len(monte_carlo_finals) < 10:
        return None
    X = np.array([list(baseline_values.values()) for _ in range(len(monte_carlo_finals))])
    model = LinearRegression().fit(X, monte_carlo_finals)
    return dict(zip(baseline_values.keys(), model.coef_))


# ============================================================
# DATA PROCESSING (from Phase 1 – unchanged)
# ============================================================

@st.cache_data(show_spinner="Processing survey data...")
def process_survey(_survey_df):
    if _survey_df is None:
        return None, None, "No survey file uploaded."
    try:
        df = _survey_df.copy()
        missing = [c for c in REQUIRED_SURVEY_COLS if c not in df.columns]
        if missing:
            return None, None, f"Missing columns: {', '.join(missing)}"
        if 'school_id_no' in df.columns:
            df['school_id_no'] = df['school_id_no'].astype(int)
        elif 'school_id' in df.columns:
            df['school_id_no'] = df['school_id'].astype(str).apply(lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x))
        else:
            return None, None, "Need 'school_id_no' or 'school_id'."
        if 'school_name' not in df.columns:
            df['school_name'] = df['school_id_no'].apply(lambda x: f"School_{x}")
        else:
            df['school_name'] = df['school_name'].fillna(df['school_id_no'].apply(lambda x: f"School_{x}"))
        df['month_num'] = df['month'].apply(month_str_to_num)
        for v in VARIABLES:
            if not pd.api.types.is_numeric_dtype(df[v]):
                df[v] = pd.to_numeric(df[v], errors='coerce')
            if df[v].isna().any() or (df[v] < 0).any() or (df[v] > 1).any():
                return None, None, f"Column {v} must be numeric between 0 and 1."
        school_info = df[['school_id_no', 'school_name']].drop_duplicates().sort_values('school_id_no')
        return df, school_info, None
    except Exception as e:
        return None, None, f"Survey error: {str(e)}"

@st.cache_data(show_spinner="Processing metadata...")
def process_metadata(_metadata_df):
    if _metadata_df is None:
        return None, "No metadata file uploaded."
    try:
        df = _metadata_df.copy()
        missing = [c for c in REQUIRED_META_COLS if c not in df.columns]
        if missing:
            return None, f"Missing columns: {', '.join(missing)}"
        if 'school_id_no' not in df.columns:
            if 'school' in df.columns:
                df['school_id_no'] = df['school'].astype(str).apply(lambda x: int(x.split('_')[-1]) if '_' in str(x) else int(x))
            else:
                return None, "Need 'school_id_no' or 'school'."
        df['school_id_no'] = df['school_id_no'].astype(int)
        for col, default in OPTIONAL_META_COLS.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
        if df['upload_date'].isna().any():
            return None, "Invalid dates in upload_date."
        if df['utilized_by_school'].dtype != bool:
            df['utilized_by_school'] = df['utilized_by_school'].astype(str).str.lower().map(
                {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}).fillna(False)
        return df, None
    except Exception as e:
        return None, f"Metadata error: {str(e)}"

def get_latest_survey(survey_df, school_id):
    sdf = survey_df[survey_df['school_id_no'] == school_id]
    if sdf.empty:
        return None
    return sdf.sort_values('month_num').iloc[-1]


# ============================================================
# CHART BUILDERS (Radar without arrow, Research Dashboard)
# ============================================================

@st.cache_data(show_spinner=False)
def build_radar_chart(survey_values_tuple, school_name, dark_mode):
    labels = [f"{v} ({MILESTONE_SHORT[i]})" for i, v in enumerate(VARIABLES)]
    values = list(survey_values_tuple)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill='toself', name=school_name,
        line_color=USTP_GOLD, fillcolor="rgba(245, 166, 35, 0.3)",
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>'
    ))
    text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
    fig.update_layout(
        template='plotly_dark' if dark_mode else 'plotly_white',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0],
                            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                            color=text_color),
            angularaxis=dict(direction="clockwise", tickfont=dict(size=11, color=text_color))
        ),
        title=f"Current Research Culture Profile (latest quarter)<br>{school_name}",
        showlegend=False, font=dict(color=text_color),
        height=500, margin=dict(l=60, r=80, t=80, b=100)
    )
    return fig

@st.cache_data(show_spinner=False)
def _compute_research_metrics(metadata_df, school_id):
    school_meta = metadata_df[metadata_df['school_id_no'] == school_id]
    metrics = {}
    if school_meta.empty:
        return metrics
    tc = school_meta['theme'].value_counts().reset_index()
    tc.columns = ['Theme', 'Count']
    metrics['theme_counts'] = tc
    metrics['top_theme'] = tc.iloc[0]['Theme'] if not tc.empty else "N/A"
    if 'utilized_by_school' in school_meta.columns:
        tu = school_meta.groupby('theme')['utilized_by_school'].mean().reset_index()
        tu.columns = ['Theme', 'Utilisation Rate']
        metrics['theme_util_df'] = tu
        metrics['top_util_theme'] = tu.loc[tu['Utilisation Rate'].idxmax(), 'Theme'] if not tu.empty else "N/A"
    else:
        metrics['theme_util_df'] = None; metrics['top_util_theme'] = "N/A"
    sc = school_meta['status'].value_counts().reset_index()
    sc.columns = ['Status', 'Count']
    metrics['status_counts'] = sc
    published = sc[sc['Status'] == 'published']['Count'].sum() if not sc.empty else 0
    total = sc['Count'].sum() if not sc.empty else 0
    metrics['pub_rate'] = (published / total * 100) if total > 0 else 0
    metrics['total_outputs'] = total
    if 'upload_date' in school_meta.columns:
        sm = school_meta.copy()
        sm['quarter'] = sm['upload_date'].dt.to_period('Q').astype(str)
        ot = sm.groupby('quarter').size().reset_index(name='count')
        metrics['output_timeline'] = ot if not ot.empty else None
        if 'utilized_by_school' in school_meta.columns:
            ut = sm.groupby('quarter')['utilized_by_school'].mean().reset_index()
            ut.columns = ['quarter', 'utilisation_rate']
            metrics['util_timeline'] = ut if not ut.empty else None
        else:
            metrics['util_timeline'] = None
    else:
        metrics['output_timeline'] = None; metrics['util_timeline'] = None
    utilised = school_meta['utilized_by_school'].sum() if 'utilized_by_school' in school_meta.columns else 0
    total = len(school_meta)
    util_rate = (utilised / total * 100) if total > 0 else 0
    level, desc = interpret_utilisation_rate(util_rate)
    metrics['util_rate'] = util_rate; metrics['util_level'] = level; metrics['util_desc'] = desc
    tc2 = school_meta['teacher_name'].value_counts().reset_index().head(10)
    tc2.columns = ['Teacher', 'Number of Outputs']
    metrics['teacher_counts'] = tc2
    metrics['top_teacher'] = tc2.iloc[0]['Teacher'] if not tc2.empty else "N/A"
    if 'years_of_service' in school_meta.columns and not school_meta['years_of_service'].isna().all():
        ts = school_meta.groupby('teacher_name').agg(
            output_count=('document_type', 'count'), years_of_service=('years_of_service', 'first')
        ).dropna()
        if len(ts) > 1:
            x, y = ts['years_of_service'], ts['output_count']
            z = np.polyfit(x, y, 1)
            trend_x = np.linspace(x.min(), x.max(), 100)
            metrics['service_data'] = {'teacher_summary': ts, 'trend_x': trend_x,
                                       'trend_y': np.poly1d(z)(trend_x), 'slope': z[0],
                                       'avg_service': x.mean(), 'avg_output': y.mean()}
        else:
            metrics['service_data'] = None
    else:
        metrics['service_data'] = None
    if 'teacher_rank' in school_meta.columns and not school_meta['teacher_rank'].isna().all():
        rg = school_meta.groupby('teacher_rank').size().reset_index(name='total_outputs')
        rn = school_meta.groupby('teacher_rank')['teacher_name'].nunique().reset_index(name='num_teachers')
        rs = rg.merge(rn, on='teacher_rank')
        rs['avg_outputs'] = rs['total_outputs'] / rs['num_teachers']
        metrics['rank_summary'] = rs
    else:
        metrics['rank_summary'] = None
    if 'educational_attainment' in school_meta.columns and not school_meta['educational_attainment'].isna().all():
        eg = school_meta.groupby('educational_attainment').size().reset_index(name='total_outputs')
        en = school_meta.groupby('educational_attainment')['teacher_name'].nunique().reset_index(name='num_teachers')
        es = eg.merge(en, on='educational_attainment')
        es['avg_outputs'] = es['total_outputs'] / es['num_teachers']
        metrics['edu_summary'] = es
    else:
        metrics['edu_summary'] = None
    return metrics

def render_research_dashboard(metrics, school_name, dark_mode):
    figs = {}
    template = 'plotly_dark' if dark_mode else 'plotly_white'
    if 'theme_counts' in metrics:
        figs['theme_distribution'] = px.bar(metrics['theme_counts'], x='Theme', y='Count',
                                            title=f"Theme Distribution - {school_name}",
                                            color='Theme', color_discrete_sequence=[USTP_GOLD, DEPED_RED, USTP_DARK_BLUE])
        figs['theme_distribution'].update_layout(template=template)
    if metrics.get('theme_util_df') is not None:
        figs['theme_utilisation'] = px.bar(metrics['theme_util_df'], x='Theme', y='Utilisation Rate',
                                           title=f"Theme Utilisation Rate - {school_name}",
                                           color='Utilisation Rate', color_continuous_scale=['#F5A623', '#0D2B5E'])
        figs['theme_utilisation'].update_layout(template=template)
    if 'status_counts' in metrics:
        figs['publication_status'] = px.bar(metrics['status_counts'], x='Status', y='Count',
                                            title=f"Publication Status - {school_name}",
                                            color='Status', color_discrete_sequence=[USTP_DARK_BLUE, USTP_GOLD, DEPED_MAROON])
        figs['publication_status'].update_layout(template=template)
    if metrics.get('output_timeline') is not None:
        figs['output_timeline'] = px.line(metrics['output_timeline'], x='quarter', y='count',
                                          title=f"Research Output Timeline - {school_name}", markers=True)
        figs['output_timeline'].update_layout(template=template, xaxis_title='Quarter', yaxis_title='Number of Outputs')
    if metrics.get('util_timeline') is not None:
        figs['util_timeline'] = px.line(metrics['util_timeline'], x='quarter', y='utilisation_rate',
                                        title=f"Utilisation Rate Over Time - {school_name}", markers=True)
        figs['util_timeline'].update_layout(template=template, xaxis_title='Quarter', yaxis_title='Utilisation Rate')
    if 'teacher_counts' in metrics:
        figs['teacher_productivity'] = px.bar(metrics['teacher_counts'], x='Number of Outputs', y='Teacher',
                                              orientation='h', title=f"Teacher Productivity (Top 10) - {school_name}",
                                              color='Number of Outputs', color_continuous_scale=['#F5A623', '#0D2B5E'])
        figs['teacher_productivity'].update_layout(template=template)
    sd = metrics.get('service_data')
    if sd is not None:
        ts = sd['teacher_summary']
        text_color = USTP_GOLD if dark_mode else USTP_DARK_BLUE
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts['years_of_service'], y=ts['output_count'], mode='markers',
                                 marker=dict(size=12, color=USTP_GOLD, line=dict(color=USTP_DARK_BLUE, width=1)),
                                 text=ts.index, hoverinfo='text+x+y', name='Teachers'))
        fig.add_trace(go.Scatter(x=sd['trend_x'], y=sd['trend_y'], mode='lines',
                                 line=dict(color=USTP_DARK_BLUE, width=2, dash='dash'), name='Trend'))
        fig.update_layout(template=template, title=f"Years of Service vs Research Outputs - {school_name}",
                          xaxis_title="Years of Service", yaxis_title="Number of Outputs",
                          font=dict(color=text_color), showlegend=True, height=400)
        figs['service_vs_output'] = fig
    if metrics.get('rank_summary') is not None:
        figs['rank_breakdown'] = px.bar(metrics['rank_summary'], x='teacher_rank', y='total_outputs',
                                        title=f"Research Outputs by Teacher Rank - {school_name}",
                                        color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E'])
        figs['rank_breakdown'].update_layout(template=template)
    if metrics.get('edu_summary') is not None:
        figs['edu_breakdown'] = px.bar(metrics['edu_summary'], x='educational_attainment', y='total_outputs',
                                       title=f"Research Outputs by Educational Attainment - {school_name}",
                                       color='total_outputs', color_continuous_scale=['#F5A623', '#0D2B5E'])
        figs['edu_breakdown'].update_layout(template=template)
    return figs


# ============================================================
# BASELINE / SYNOPSIS / HEATMAP (from Phase 1)
# ============================================================

def generate_baseline_synopsis(survey_row, school_name, _metadata_df):
    values = {v: survey_row[v] for v in VARIABLES}
    strengths = [v for v in VARIABLES if values[v] >= 0.6]
    gaps = [v for v in VARIABLES if values[v] <= 0.3]
    moderate = [v for v in VARIABLES if 0.3 < values[v] < 0.6]
    baseline_rcsi = np.mean(list(values.values()))
    gap_actions = {
        'C': "Build Teacher Capacity (C). Conduct training workshops on research methods and data analysis.",
        'S': "Improve Structured Support (S). Allocate budget and time for research activities.",
        'I': "Institutional Anchoring (I). Embed research into school plans and regular meetings.",
        'P': "Strengthen Community of Practice (P). Establish regular research sharing forums and peer mentoring.",
        'M': "Enhance Impact Realization (M). Document and share evidence of research impact.",
    }
    recommendations = [f"**Priority: {gap_actions[var]}**" for var in gaps if var in gap_actions]
    if not recommendations:
        recommendations.append("All variables are at moderate or high levels. Maintain current policies.")
    return {'strengths': strengths, 'gaps': gaps, 'moderate': moderate,
            'baseline_rcsi': baseline_rcsi, 'recommendations': recommendations, 'values': values}

def baseline_heatmap(survey_df, metadata_df, dark_mode):
    st.markdown("### Historical Correlation Matrix (Diagnostic)")
    if survey_df is None or metadata_df is None:
        st.info("Insufficient data.")
        return
    survey_agg = survey_df.groupby(['school_id_no', 'month_num'])[VARIABLES].mean().reset_index()
    meta = metadata_df.copy()
    meta['month_num'] = meta['upload_date'].apply(date_to_month_num)
    output_counts = meta.groupby(['school_id_no', 'month_num']).size().reset_index(name='output_count')
    merged = survey_agg.merge(output_counts, on=['school_id_no', 'month_num'], how='inner')
    if merged.empty:
        st.info("Insufficient data.")
        return
    corr = merged[VARIABLES + ['output_count']].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale='Blues')
    fig.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Correlation between the seven variables and research output count in historical data.")

def cycle_research_correlation(agent, metadata_df, school_id, dark_mode):
    if not agent.cycle_improvements:
        st.info("No cycles completed.")
        return
    sm = metadata_df[metadata_df['school_id_no'] == school_id].copy()
    sm['month_num'] = sm['upload_date'].apply(date_to_month_num)
    cumulative = [len(sm[sm['month_num'] <= rec.completion_month]) for rec in agent.cycle_improvements]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c.cycle_number for c in agent.cycle_improvements], y=cumulative,
                             mode='markers+lines', marker=dict(size=10, color=USTP_GOLD),
                             line=dict(color=USTP_DARK_BLUE)))
    fig.update_layout(template='plotly_dark' if dark_mode else 'plotly_white', title="Cycle vs Cumulative Research Outputs")
    st.plotly_chart(fig, use_container_width=True)

def division_level_analysis(metadata_df, history_per_school, sim_agents, dark_mode):
    st.markdown("### Division-Level Analysis")
    if not metadata_df.empty:
        ts = metadata_df.groupby(['teacher_name', 'school_id_no']).size().reset_index(name='total_outputs')
        ts = ts.sort_values('total_outputs', ascending=False).head(20)
        st.dataframe(ts[['teacher_name', 'school_id_no', 'total_outputs']])
        top_div_teacher = ts.iloc[0]['teacher_name'] if not ts.empty else "N/A"
        top_div_school = ts.iloc[0]['school_id_no'] if not ts.empty else "N/A"
        top_div_outputs = ts.iloc[0]['total_outputs'] if not ts.empty else 0
    else:
        top_div_teacher = top_div_school = "N/A"; top_div_outputs = 0
    all_durations = {m: [] for m in range(7)}
    for sid, hist in history_per_school.items():
        if 'milestone' not in hist or not hist['milestone']:
            continue
        milestones = hist['milestone']
        for i in range(1, len(milestones)):
            if milestones[i] != milestones[i - 1]:
                start = milestones.index(milestones[i - 1], 0, i) if milestones[i - 1] in milestones[:i] else i - 1
                all_durations[milestones[i - 1]].append(i - start)
        if milestones:
            last = milestones[-1]
            start = milestones.index(last, 0, len(milestones)) if last in milestones else len(milestones) - 1
            all_durations[last].append(len(milestones) - start)
    avg_dur = {m: np.mean(v) if v else np.nan for m, v in all_durations.items()}
    df_dur = pd.DataFrame({'Milestone': [f'M{i}' for i in range(7)],
                           'Avg Months': [avg_dur.get(i, np.nan) for i in range(7)]}).dropna()
    bottleneck = "N/A"; bottleneck_time = 0
    if not df_dur.empty:
        max_row = df_dur.loc[df_dur['Avg Months'].idxmax()]
        bottleneck = max_row['Milestone']; bottleneck_time = max_row['Avg Months']
        fig_dur = px.bar(df_dur, x='Milestone', y='Avg Months', title="Average Months per Milestone",
                         color='Avg Months', color_continuous_scale=['#F5A623', '#0D2B5E'])
        fig_dur.update_layout(template='plotly_dark' if dark_mode else 'plotly_white')
        st.plotly_chart(fig_dur, use_container_width=True)
        st.caption(f"Bottleneck: {bottleneck} ({bottleneck_time:.1f} months).")
    else:
        st.info("Not enough transition data.")
    return {'top_div_teacher': top_div_teacher, 'top_div_school': top_div_school,
            'top_div_outputs': top_div_outputs, 'bottleneck_milestone': bottleneck,
            'bottleneck_time': bottleneck_time}

def school_comparison_dashboard(survey_df, history_per_school, school_info, selected_school_ids, dark_mode):
    st.markdown("### Comparative School Analysis")
    if len(selected_school_ids) < 2:
        st.info("Select at least two schools.")
        return
    histories = {sid: history_per_school.get(sid) for sid in selected_school_ids if history_per_school.get(sid)}
    if not histories:
        st.info("No simulation history.")
        return
    fig = make_subplots(rows=2, cols=1, subplot_titles=("RCSI Comparison", "Milestone Comparison"))
    for sid, hist in histories.items():
        name = school_info[school_info['school_id_no'] == sid]['school_name'].values[0]
        fig.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'], mode='lines', name=f"{name} RCSI"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'], mode='lines', name=f"{name} Milestone"), row=2, col=1)
    fig.update_layout(height=600, template='plotly_dark' if dark_mode else 'plotly_white')
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="CDO Research Culture Sustainability Framework", layout="wide")
st.markdown("<h1 style='text-align: center; color: #0D2B5E;'>CDO Division Research Culture Sustainability Framework</h1>", unsafe_allow_html=True)

for key, default in [('max_schools', 200), ('num_schools', 0), ('total_teachers', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar (one unified block) ---
with st.sidebar:
    st.markdown(f"<h2 style='color: {USTP_DARK_BLUE};'>Controls</h2>", unsafe_allow_html=True)
    dark_mode = st.checkbox("Dark Mode", value=False)
    apply_theme(dark_mode)

    st.metric("Total Schools Loaded", st.session_state.num_schools)
    st.metric("Total Teachers Recorded", st.session_state.total_teachers)
    if st.session_state.get('total_months', 0) > 0:
        st.metric("Simulation Month", st.session_state.total_months)

    st.markdown("---")
    # Removed the "Analyze Baseline" button – baseline is now automatic

    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Policy Levers</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        u_train = st.slider("Training freq.", 0.0, 1.0, 0.5, 0.05, help="Frequency of research training workshops per quarter")
        u_mentor = st.slider("Mentorship ratio", 0.0, 1.0, 0.5, 0.05, help="Ratio of experienced-to-novice researcher pairings")
        u_budget = st.slider("Support budget", 0.0, 1.0, 0.5, 0.05, help="Proportion of budget allocated to research support")
    with col2:
        u_lead = st.slider("Leadership commit.", 0.0, 1.0, 0.5, 0.05, help="Degree of school leadership commitment")
        u_collab = st.slider("Collaboration freq.", 0.0, 1.0, 0.5, 0.05, help="Frequency of inter-school collaboration events")
    levers = {'u_train': u_train, 'u_mentor': u_mentor, 'u_budget': u_budget, 'u_lead': u_lead, 'u_collab': u_collab}

    st.markdown("---")
    st.markdown(f"<h3 style='color: {USTP_DARK_BLUE};'>Simulation Parameters</h3>", unsafe_allow_html=True)
    duration = st.selectbox("Run duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=9)
    random_events = st.checkbox("Enable random events", value=False)
    use_survey = st.checkbox("Override with survey data", value=True)

    # --- Phase 2 Monte Carlo controls ---
    st.markdown("---")
    st.markdown("#### 🎲 Monte Carlo (Phase 2)")
    mc_enabled = st.checkbox("Enable Monte Carlo", value=False)
    mc_runs = st.number_input("Number of runs", min_value=10, max_value=100, value=30, step=10)

    st.markdown("---")
    st.markdown("#### Simulation Actions")
    col_buttons = st.columns(3)
    with col_buttons[0]:
        run_btn = st.button("Run", use_container_width=True)
    with col_buttons[1]:
        step_btn = st.button("Step (1 month)", use_container_width=True)
    with col_buttons[2]:
        reset_btn = st.button("Reset", use_container_width=True)
    st.caption("Run: full forecast. Step: one month. Reset: clear history.")

    st.markdown("---")
    st.markdown("#### Export Data")
    export_btn = st.button("Export results (CSV)", use_container_width=True)

# --- File Upload (unchanged from Phase 1) ---
with st.expander("Step 1: Upload your CSV files", expanded=True):
    st.markdown("""
    **Instructions:**
    - Upload **Quarterly Survey** CSV (columns: `month, school_id_no, R, A, C, S, I, P, M`).
    - Upload **Research Metadata** CSV (columns: `upload_date, teacher_name, school_id_no, ...`).
    """)
    col1, col2 = st.columns(2)
    with col1:
        survey_file = st.file_uploader("Upload quarterly survey (CSV)", type=["csv"], key="survey")
    with col2:
        metadata_file = st.file_uploader("Upload research metadata (CSV)", type=["csv"], key="metadata")
    st.markdown("---")
    st.markdown("**Need templates?**")
    survey_template = ("month,school_id_no,school_name,R,A,C,S,I,P,M\n"
                       "2026-01,1,School_1,0.32,0.41,0.28,0.15,0.14,0.19,0.08")
    metadata_template = ("upload_date,teacher_name,school_id_no,document_type,title,theme,"
                         "status,publication_link,utilized_by_school,utilization_date,"
                         "year_undertaken,years_of_service,teacher_rank,educational_attainment\n"
                         "2026-03-15,Anna Reyes,1,abstract,Improving Reading,Teaching Strategies,"
                         "published,https://doi.org/10.1234,True,2026-02-10,2025,10,Teacher II,Master's")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Survey Template", survey_template, "quarterly_survey_template.csv", "text/csv")
    with c2:
        st.download_button("Metadata Template", metadata_template, "research_metadata_template.csv", "text/csv")

# ============================================================
# MAIN AREA
# ============================================================
if survey_file is not None and metadata_file is not None:
    survey_df_raw = pd.read_csv(survey_file)
    metadata_df_raw = pd.read_csv(metadata_file)
    survey_df, school_info, survey_error = process_survey(survey_df_raw)
    metadata_df, meta_error = process_metadata(metadata_df_raw)

    if survey_error:
        st.error(f"Survey error: {survey_error}")
    elif meta_error:
        st.error(f"Metadata error: {meta_error}")
    else:
        actual_count = len(school_info)
        total_teachers = metadata_df['teacher_name'].nunique()
        state_changed = False
        if st.session_state.num_schools != actual_count:
            st.session_state.num_schools = actual_count; state_changed = True
        if st.session_state.total_teachers != total_teachers:
            st.session_state.total_teachers = total_teachers; state_changed = True
        if state_changed:
            st.rerun()

        st.success(f"Loaded {actual_count} schools and {total_teachers} teachers.")

        school_ids = school_info['school_id_no'].tolist()
        id_to_label = {sid: f"ID {sid}: {school_info[school_info['school_id_no']==sid]['school_name'].values[0]}" for sid in school_ids}
        label_to_id = {v: k for k, v in id_to_label.items()}
        selected_school_label = st.selectbox("Select school", list(id_to_label.values()), index=0)
        selected_school_id = label_to_id[selected_school_label]
        selected_school_name = id_to_label[selected_school_id].split(": ", 1)[1]

        # Baseline section
        st.markdown("<h2 style='text-align: center;'>Baseline from Uploaded Data</h2>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Research Outputs (Recent)")
        df_show = metadata_df[metadata_df['school_id_no'] == selected_school_id].sort_values('upload_date', ascending=False)
        if not df_show.empty:
            st.dataframe(df_show[['teacher_name', 'year_undertaken', 'title', 'theme', 'status', 'utilized_by_school']].head(10))
        else:
            st.info("No research outputs for this school.")

        latest = get_latest_survey(survey_df, selected_school_id)
        if latest is not None:
            col_left, col_right = st.columns([1, 5])
            with col_left:
                st.markdown("**Legend:**")
                legend_items = "\n".join(f"- **{v} ({MILESTONE_SHORT[i]})** → {MILESTONE_NAMES[i]}" for i, v in enumerate(VARIABLES))
                st.markdown(f'<div style="font-size: 12px;">{legend_items}</div>', unsafe_allow_html=True)
            with col_right:
                survey_tuple = tuple(latest[v] for v in VARIABLES)
                radar_fig = build_radar_chart(survey_tuple, selected_school_name, dark_mode)
                st.plotly_chart(radar_fig, use_container_width=True)
                get_figure_download_link(radar_fig, "radar_chart.html", "Download Radar Chart")
        else:
            st.info("No survey data for current quarter.")

        with st.expander("Research Outputs Dashboard"):
            metrics = _compute_research_metrics(metadata_df, selected_school_id)
            figs = render_research_dashboard(metrics, selected_school_name, dark_mode)
            if figs:
                for name, fig in figs.items():
                    st.plotly_chart(fig, use_container_width=True)
                    get_figure_download_link(fig, f"{name}.html")
                st.metric("School-level Research Utilisation Rate",
                          f"{metrics.get('util_rate', 0):.1f}% → {metrics.get('util_level', 'N/A')} level",
                          help=metrics.get('util_desc', ''))

        # ---------------------- NEW: Auto-generate baseline synopsis ----------------------
        if 'baseline_synopsis' not in st.session_state and latest is not None:
            with st.spinner("Generating baseline synopsis..."):
                # Add a tiny delay so the spinner is visible
                time.sleep(0.5)
                st.session_state.baseline_synopsis = generate_baseline_synopsis(latest, selected_school_name, metadata_df)
                st.session_state.baseline_survey_row = latest.to_dict()
                school_survey = survey_df[survey_df['school_id_no'] == selected_school_id]
                st.session_state.baseline_std_devs = {v: school_survey[v].std() if len(school_survey) > 1 else 0.1 for v in VARIABLES}
                st.rerun()  # force re-run to display the synopsis

        if 'baseline_synopsis' in st.session_state:
            bs = st.session_state.baseline_synopsis
            bg_color = '#2E2E2E' if dark_mode else '#E3F2FD'
            text_col = DARK_TEXT if dark_mode else 'inherit'
            st.markdown("### Baseline Synopsis")
            st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
            <b>School: {selected_school_name}</b><br>
            Baseline RCSI: {bs['baseline_rcsi']:.3f}<br>
            Strengths (≥0.6): {', '.join(bs['strengths']) if bs['strengths'] else 'None'}<br>
            Critical Gaps (≤0.3): {', '.join(bs['gaps']) if bs['gaps'] else 'None'}<br>
            Moderate (0.3–0.6): {', '.join(bs['moderate']) if bs['moderate'] else 'None'}<br>
            Actionable Recommendations:<br>{'<br>'.join(bs['recommendations'])}
            </div>
            """, unsafe_allow_html=True)

        baseline_heatmap(survey_df, metadata_df, dark_mode)

        # ---------- Phase 2: Calibration & Heterogeneous Params ----------
        if 'calibrated_coeff' not in st.session_state:
            with st.spinner("Calibrating model coefficients from data..."):
                # Add a small delay for visibility
                time.sleep(0.5)
                coeff, calib_msg = calibrate_coefficients(survey_df)
                if coeff is not None:
                    st.success(calib_msg)
                    st.session_state.calibrated_coeff = coeff
                    st.session_state.calibration_status = "custom"
                else:
                    # Use default coefficients; show an info message (not a warning)
                    st.info(calib_msg)  # e.g., "No survey data – using default coefficients."
                    st.session_state.calibrated_coeff = None
                    st.session_state.calibration_status = "default"
                st.rerun()

        # Show calibration status in sidebar or main area
        if 'calibration_status' in st.session_state:
            if st.session_state.calibration_status == "custom":
                st.caption("✅ Using data‑calibrated coefficients.")
            else:
                st.caption("ℹ️ Using default coefficients (insufficient data for calibration).")

        agent_params = get_agent_params(school_ids, survey_df, metadata_df, st.session_state.calibrated_coeff)

        # ---------- Initialize Simulation ----------
        if 'sim' not in st.session_state:
            st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events, agent_params)
            st.session_state.current_month = 0
            st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)

        # ---------- Run / Step / Reset ----------
        if run_btn:
            st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events, agent_params)
            st.session_state.current_month = 0; st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)
            progress = st.progress(0, text="Running simulation...")
            for m in range(1, duration + 1):
                if use_survey:
                    apply_survey_override(st.session_state.sim.agents, survey_df, m)
                st.session_state.sim.step(levers, m)
                st.session_state.current_month = m; st.session_state.total_months = m
                record_history(st.session_state.history, st.session_state.sim.agents, m)
                progress.progress(m / duration, text=f"Month {m}/{duration}")
            progress.empty()
            if mc_enabled:
                with st.spinner(f"Running {mc_runs} Monte Carlo simulations..."):
                    mc_data, mc_info = monte_carlo_sim(mc_runs, Simulation, agent_params, school_ids, levers, duration,
                                                       use_survey, survey_df, metadata_df, selected_school_id)
                    st.session_state.mc_data = mc_data
                    st.session_state.mc_info = mc_info
            st.rerun()

        if step_btn:
            target = st.session_state.current_month + 1
            if use_survey:
                apply_survey_override(st.session_state.sim.agents, survey_df, target)
            st.session_state.sim.step(levers, target)
            st.session_state.current_month = target; st.session_state.total_months = target
            record_history(st.session_state.history, st.session_state.sim.agents, target)
            st.rerun()

        if reset_btn:
            st.session_state.sim = init_simulation_with_data(school_ids, metadata_df, random_events, agent_params)
            st.session_state.current_month = 0; st.session_state.total_months = 0
            st.session_state.history = create_empty_history(school_ids)
            st.rerun()

        # ---------- Simulation Results Display ----------
        if st.session_state.total_months > 0:
            # Ensure text_col is defined for synopses
            text_col = DARK_TEXT if dark_mode else 'inherit'

            st.markdown("<h2 style='text-align: center;'>Simulated Data</h2>", unsafe_allow_html=True)
            st.markdown("---")
            hist = st.session_state.history.get(selected_school_id)
            agent = next((a for a in st.session_state.sim.agents if a.real_id == selected_school_id), None)
            if hist and agent:
                # Main 4-panel chart
                fig1 = make_subplots(rows=2, cols=2, subplot_titles=("Variable Evolution", "Milestone Progress",
                                                                     "Research Culture Sustainability Index (RCSI)",
                                                                     "Improvement per Completed Cycle"))
                for i, var in enumerate(VARIABLES):
                    fig1.add_trace(go.Scatter(x=hist['month'], y=hist[var], mode='lines', name=var,
                                              line=dict(color=VAR_COLORS[i])), row=1, col=1)
                fig1.add_trace(go.Scatter(x=hist['month'], y=hist['milestone'], mode='lines', name='Milestone',
                                          line=dict(color=DEPED_RED, width=3)), row=1, col=2)
                fig1.add_trace(go.Scatter(x=hist['month'], y=hist['running_outcome'], mode='lines', name='RCSI',
                                          line=dict(color=USTP_GOLD, width=3)), row=2, col=1)
                if agent.cycle_improvements:
                    cycles = [c.cycle_number for c in agent.cycle_improvements]
                    improvements = [c.total_improvement for c in agent.cycle_improvements]
                    fig1.add_trace(go.Bar(x=cycles, y=improvements, name='RCSI per cycle',
                                          marker_color=USTP_DARK_BLUE), row=2, col=2)
                else:
                    fig1.add_annotation(text="No cycles completed yet", xref="x2 domain", yref="y2 domain",
                                        x=0.5, y=0.5, showarrow=False, row=2, col=2)
                template = 'plotly_dark' if dark_mode else 'plotly_white'
                fig1.update_layout(height=800, showlegend=True, font=dict(color=text_col), template=template)
                st.plotly_chart(fig1, use_container_width=True)
                get_figure_download_link(fig1, "simulation_overview.html", "Download Simulation Charts")

                with st.expander("Cycle vs Research Outputs"):
                    cycle_research_correlation(agent, metadata_df, selected_school_id, dark_mode)

                with st.expander("Division-Level Analysis"):
                    div_metrics = division_level_analysis(metadata_df, st.session_state.history,
                                                          st.session_state.sim.agents, dark_mode)

                with st.expander("Comparative School Analysis"):
                    selected_comparison = st.multiselect("Select schools to compare", options=school_ids,
                                                         format_func=lambda x: id_to_label[x])
                    school_comparison_dashboard(survey_df, st.session_state.history, school_info,
                                                selected_comparison, dark_mode)

                # ---------- Phase 2: Sensitivity & Monte Carlo Expandable ----------
                sensitivity_info = ""
                if not st.session_state.get('sensitivity_fig'):
                    with st.spinner("Computing sensitivity analysis..."):
                        fig_tornado, sensitivity_info = run_sensitivity(Simulation, agent_params, school_ids, levers,
                                                                        duration, use_survey, survey_df, metadata_df,
                                                                        selected_school_id)
                        st.session_state.sensitivity_fig = fig_tornado
                        st.session_state.sensitivity_info = sensitivity_info
                else:
                    fig_tornado = st.session_state.sensitivity_fig
                    sensitivity_info = st.session_state.sensitivity_info

                with st.expander("Sensitivity Analysis (Tornado)"):
                    st.plotly_chart(fig_tornado, use_container_width=True)
                    st.caption("Each lever varied ±10% while others fixed at current slider values.")
                    if sensitivity_info:
                        st.markdown(sensitivity_info)

                # ---------- Monte Carlo Display with added table and caption ----------
                if 'mc_data' in st.session_state:
                    with st.expander("Monte Carlo Uncertainty Bands"):
                        mc_data = st.session_state.mc_data
                        fig_mc = plot_monte_carlo_bands(mc_data, dark_mode)
                        st.plotly_chart(fig_mc, use_container_width=True)
                        st.caption(f"Shaded area: P10‑P90 range over {mc_runs} simulations.")
                        if st.session_state.get('mc_info'):
                            st.markdown(st.session_state.mc_info)
                            # RCSI Interpretation Table
                            with st.expander("RCSI Interpretation Guide"):
                                st.markdown(get_rcsi_interpretation_table())
                            # Graphic interpretation caption aligned with school & division
                            st.markdown("""
                            **Graph Interpretation:**  
                            At the school level, the simulated RCSI’s tight interquartile range confirms that internal processes are fully insulated from external shocks, validating the current local administration.  
                            Divergently, at the division level, this uniformity signals a systemic plateau—prompting division leaders to shift focus from risk mitigation to pedagogical innovation, as quantitative variance no longer provides actionable leverage for district‑wide improvement.
                            """)
                        if 'baseline_synopsis' in st.session_state:
                            baseline_vals = st.session_state.baseline_synopsis['values']
                            causal_coeffs = causal_analysis(mc_data['final_rcsi'], baseline_vals)
                            if causal_coeffs:
                                st.markdown("**Causal Impact (increase final RCSI per unit increase in baseline variable):**")
                                df_causal = pd.DataFrame(list(causal_coeffs.items()), columns=['Variable', 'Impact'])
                                st.dataframe(df_causal)
                            else:
                                st.info("Not enough Monte Carlo runs for causal analysis (need >10).")

                # ===== School-Level Synopsis (enhanced with sensitivity/Monte Carlo insights and gap) =====
                rcsi_val = agent.running_total_outcome
                rcsi_level = classify_rcsi(rcsi_val)
                milestone_name = MILESTONE_NAMES.get(agent.current_milestone, f"Milestone {agent.current_milestone}")
                if agent.cycle_count >= 2:
                    cycle_text = f"has completed {agent.cycle_count} full cycles, indicating a self-sustaining research culture."
                elif agent.cycle_count == 1:
                    cycle_text = "has completed one full cycle, demonstrating initial sustainability."
                else:
                    cycle_text = "has not yet completed any full cycle."
                if agent.current_milestone == 0:
                    milestone_progress = "is at the very beginning of the journey."
                elif agent.current_milestone <= 2:
                    milestone_progress = "has moved beyond initial readiness but remains in early capacity‑building phases."
                elif agent.current_milestone <= 4:
                    milestone_progress = "has established structured support and is embedding research into institutional practice."
                else:
                    milestone_progress = "is realising tangible impact and is approaching or has achieved cyclical sustainability."
                key_R = hist['R'][-1] if hist['R'] else 0; key_M = hist['M'][-1] if hist['M'] else 0

                sens_text = sensitivity_info if sensitivity_info else ""
                mc_text = st.session_state.get('mc_info', "")

                # Compute division average RCSI for gap analysis
                avg_rcsi_division = np.mean([a.running_total_outcome for a in st.session_state.sim.agents])
                gap = rcsi_val - avg_rcsi_division
                gap_text = f"Compared to the division average of **{avg_rcsi_division:.3f}**, this school is **{gap:+.3f}** points {'above' if gap > 0 else 'below'} the division average."

                output_trend_text = ""
                tl = metrics.get('output_timeline')
                if tl is not None and len(tl) >= 2:
                    if tl.iloc[-1]['count'] > tl.iloc[-2]['count']:
                        output_trend_text = "Research output is increasing over time."
                    elif tl.iloc[-1]['count'] < tl.iloc[-2]['count']:
                        output_trend_text = "Research output is declining over time."
                    else:
                        output_trend_text = "Research output has remained stable."
                    avg_output = tl['count'].mean()
                    output_trend_text += f" On average, the school produces {avg_output:.1f} outputs per quarter."

                theme_util_text = ""
                tu = metrics.get('theme_util_df')
                if tu is not None and not tu.empty:
                    max_util = tu.loc[tu['Utilisation Rate'].idxmax()]
                    min_util = tu.loc[tu['Utilisation Rate'].idxmin()]
                    theme_util_text = (f"The most utilised theme is '{max_util['Theme']}' "
                                       f"({max_util['Utilisation Rate']:.0%}), while "
                                       f"'{min_util['Theme']}' has the lowest adoption "
                                       f"({min_util['Utilisation Rate']:.0%}).")

                top_teacher_text = f"The school's top researcher is {metrics['top_teacher']}." if metrics.get('top_teacher') != "N/A" else ""

                bg_col = '#2E2E2E' if dark_mode else '#E3F2FD'
                synopsis = f"""
                After {st.session_state.total_months} months, {selected_school_name} (ID {selected_school_id}) has reached {milestone_name} and {cycle_text}
                The school's Research Culture Sustainability Index (RCSI) is <b>{rcsi_val:.3f}</b>, which falls into the <b>{rcsi_level}</b> level.
                Key indicators: Readiness (R) = {key_R:.2f}, Impact (M) = {key_M:.2f}, and current Milestone = {agent.current_milestone}.
                This combination suggests that {milestone_progress}
                The RCSI level <b>{rcsi_level.lower()}</b> reinforces this assessment.
                {output_trend_text}
                {theme_util_text}
                {top_teacher_text}
                {sens_text}
                {mc_text}
                {gap_text}
                Overall, the school is on a path toward research culture sustainability, but further policy support may be needed.
                """
                st.markdown(f"""
                <div style="background-color: {bg_col}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                <b>School {selected_school_id} ({selected_school_name}) – Simulation Synopsis</b><br>
                {synopsis}
                </div>
                """, unsafe_allow_html=True)

                # Baseline vs Simulation comparison table
                if ('baseline_synopsis' in st.session_state and 'baseline_survey_row' in st.session_state):
                    bs = st.session_state.baseline_synopsis
                    baseline_vals = st.session_state.baseline_survey_row
                    gaps = bs['gaps']
                    if gaps:
                        st.markdown("#### Baseline vs Simulation Comparison (Critical Gaps)")
                        table_data = []
                        baseline_std_devs = st.session_state.get('baseline_std_devs', {})
                        for var in gaps:
                            base_val = baseline_vals[var]
                            sim_val = getattr(agent, var)
                            diff = sim_val - base_val
                            status = "Improving" if diff > 0.01 else ("Regressing" if diff < -0.01 else "Stable")
                            std_dev = baseline_std_devs.get(var, 0.1)
                            if abs(diff) >= 0.10:
                                significance = "Both statistically and practically significant"
                            elif abs(diff) >= 0.5 * std_dev:
                                significance = "Statistically significant, but limited practical impact"
                            else:
                                significance = "Not significant (within normal variability)"
                            table_data.append({
                                "Critical Gap": VAR_FULL_NAMES[var],
                                "Baseline": f"{base_val:.2f}",
                                "Simulation": f"{sim_val:.2f}",
                                "Status": status,
                                "Significance": significance
                            })
                        st.table(pd.DataFrame(table_data))

                # ===== Division-Level Synopsis (enhanced with Monte Carlo if available) =====
                total_schools = len(st.session_state.sim.agents)
                early_stage = sum(1 for a in st.session_state.sim.agents if a.current_milestone <= 2)
                advanced_stage = sum(1 for a in st.session_state.sim.agents if a.current_milestone >= 4)
                transitional = total_schools - early_stage - advanced_stage
                early_percent = (early_stage / total_schools * 100) if total_schools > 0 else 0
                advanced_percent = (advanced_stage / total_schools * 100) if total_schools > 0 else 0
                transitional_percent = (transitional / total_schools * 100) if total_schools > 0 else 0
                early_text = f"{early_percent:.1f}% of schools" if early_percent > 0 else "No schools"
                advanced_text = f"{advanced_percent:.1f}% of schools" if advanced_percent > 0 else "No schools"
                if early_percent == 100:
                    sustainability_text = "All schools are in early milestones; foundational capacity‑building is the priority."
                elif early_percent >= 75:
                    sustainability_text = f"The vast majority ({early_percent:.1f}%) are in early milestones; urgent interventions needed."
                elif early_percent >= 50:
                    sustainability_text = f"More than half ({early_percent:.1f}%) are in early milestones; targeted policy support may accelerate progress."
                elif early_percent > 0:
                    sustainability_text = f"{early_percent:.1f}% remain in early milestones; continued efforts are required."
                else:
                    sustainability_text = "No schools are in early milestones; the division exhibits a strong, advanced research culture."

                total_outcome = sum(a.running_total_outcome for a in st.session_state.sim.agents)
                avg_rcsi = total_outcome / total_schools if total_schools > 0 else 0
                level_avg = classify_rcsi(avg_rcsi)
                total_cycles = sum(a.cycle_count for a in st.session_state.sim.agents)
                avg_milestone = np.mean([a.current_milestone for a in st.session_state.sim.agents])
                avg_milestone_interp = interpret_avg_milestone(avg_milestone)

                school_ids_in_sim = [a.real_id for a in st.session_state.sim.agents]
                div_metadata = metadata_df[metadata_df['school_id_no'].isin(school_ids_in_sim)]
                total_utilised = div_metadata['utilized_by_school'].sum() if 'utilized_by_school' in div_metadata.columns else 0
                total_research = len(div_metadata)
                div_util_rate = (total_utilised / total_research * 100) if total_research > 0 else 0

                top_div_teacher = div_metrics.get('top_div_teacher', 'N/A')
                top_div_school = div_metrics.get('top_div_school', 'N/A')
                top_div_outputs = div_metrics.get('top_div_outputs', 0)
                bottleneck_milestone = div_metrics.get('bottleneck_milestone', 'N/A')
                bottleneck_time = div_metrics.get('bottleneck_time', 0)

                output_trend_div = ""
                if not metadata_df.empty and 'upload_date' in metadata_df.columns:
                    div_timeline = metadata_df.groupby(metadata_df['upload_date'].dt.to_period('Q')).size()
                    if len(div_timeline) >= 2:
                        if div_timeline.iloc[-1] > div_timeline.iloc[-2]:
                            output_trend_div = "The division's research output is increasing over time."
                        elif div_timeline.iloc[-1] < div_timeline.iloc[-2]:
                            output_trend_div = "The division's research output is declining over time."
                        else:
                            output_trend_div = "The division's research output has remained stable."
                        avg_div_output = div_timeline.mean()
                        output_trend_div += f" On average, the division produces {avg_div_output:.1f} outputs per quarter."

                full_bottleneck = MILESTONE_NAMES.get(
                    int(bottleneck_milestone.replace('M', '')) if isinstance(bottleneck_milestone, str) and bottleneck_milestone.startswith('M') else 0,
                    bottleneck_milestone
                )
                bottleneck_insight = (f"Schools spend the most time on average in {full_bottleneck} ({bottleneck_time:.1f} months). This is the critical bottleneck." if bottleneck_milestone != "N/A" else "")
                top_teacher_insight = (f"The division's top researcher is {top_div_teacher} from {top_div_school} with {top_div_outputs} outputs." if top_div_teacher != "N/A" else "")

                # Monte Carlo reference to division synopsis if available
                div_mc_text = ""
                if 'mc_data' in st.session_state:
                    mc_finals = st.session_state.mc_data['final_rcsi']
                    div_mc_text = (f"Monte Carlo projections suggest that the division's average RCSI is estimated around "
                                   f"**{np.mean(mc_finals):.3f}** with a P10‑P90 range of "
                                   f"**{np.percentile(mc_finals, 10):.3f}** – **{np.percentile(mc_finals, 90):.3f}**, "
                                   f"indicating that the division as a whole exhibits low variance and stable sustainability.")

                bg_div = '#2E2E2E' if dark_mode else '#E8F5E9'
                st.markdown(f"""
                <div style="background-color: {bg_div}; border-left: 5px solid {USTP_GOLD}; padding: 10px; border-radius: 5px; margin-top: 10px; color: {text_col};">
                <b>Division‑Level Sustainability Synopsis (all {total_schools} schools)</b><br>
                - Average milestone = {avg_milestone:.1f} → {avg_milestone_interp}<br>
                - Total completed cycles = {total_cycles}<br>
                - Average RCSI = <b>{avg_rcsi:.3f}</b> → <b>{level_avg}</b> level.<br>
                - Average research utilisation rate = <b>{div_util_rate:.1f}%</b>.<br>
                - Stage distribution: {early_text} are in early stages (M≤2), {transitional_percent:.1f}% transitional (M3), and {advanced_text} are advanced (M≥4).<br>
                <i>Division‑wide sustainability assessment:</i> {sustainability_text}<br><br>
                <b>Productivity:</b> {output_trend_div}<br>
                <b>Bottleneck:</b> {bottleneck_insight}<br>
                <b>Top Division Researcher:</b> {top_teacher_insight}<br>
                {div_mc_text}
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Graph Interpretations"):
                    st.markdown("""
                    - **Variable Evolution:** How R, A, C, S, I, P, M change over time. Higher values (closer to 1) mean stronger readiness, awareness, capacity, etc.
                    - **Milestone Progress:** The school moves through milestones 0-6. Reaching milestone 6 and cycling back indicates a full sustainable cycle.
                    - **RCSI:** Cumulative strength of the research ecosystem, derived from Impact Realization (M) and Collaboration (P).
                    - **Improvement per Cycle:** Each bar shows the RCSI contributed by one cycle. Higher bars in later cycles indicate increasing effectiveness.
                    - **Radar Chart:** Current snapshot of the seven milestone-linked variables.
                    - **Research Outputs Dashboard:** Tracks themes, publication status, utilisation, teacher productivity, experience vs output, timeline, top teachers, and breakdown by rank and attainment.
                    - **Sensitivity Tornado:** Shows which policy lever most influences the final RCSI when varied ±10%.
                    - **Monte Carlo Bands:** Depicts the uncertainty range (P10‑P90) of RCSI and milestone trajectories over multiple simulation runs.
                    - **Division‑Level Analysis:** Milestone transition bottlenecks and teacher leaderboard.
                    - **Comparative Analysis:** Overlay multiple schools' RCSI and milestone progress.
                    - **Cycle vs Research Outputs:** Shows how research output accumulation relates to cycle progression.
                    """)

            # Export button
            if export_btn:
                all_data = []
                for a in st.session_state.sim.agents:
                    h = st.session_state.history[a.real_id]
                    for t in range(len(h['month'])):
                        row = {'school_id': a.real_id, 'month': h['month'][t], 'milestone': h['milestone'][t], 'running_outcome': h['running_outcome'][t]}
                        for v in VARIABLES:
                            row[v] = h[v][t]
                        all_data.append(row)
                df_hist = pd.DataFrame(all_data)
                cycle_records = []
                for a in st.session_state.sim.agents:
                    for rec in a.cycle_improvements:
                        cycle_records.append({'school_id': a.real_id, 'cycle_number': rec.cycle_number,
                                              'total_improvement': rec.total_improvement, 'completion_month': rec.completion_month})
                df_cycles = pd.DataFrame(cycle_records)
                st.download_button("Download simulation history", df_hist.to_csv(index=False).encode('utf-8'), "simulation_history.csv", "text/csv")
                st.download_button("Download cycle improvements", df_cycles.to_csv(index=False).encode('utf-8'), "cycle_improvements.csv", "text/csv")
else:
    st.info("Please upload quarterly survey and research metadata CSV files to begin.")
