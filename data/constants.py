# data/constants.py

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

# Baseline node data (used to initialise the 12 constructs)
NODE_DATA = {
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

# Edge data for the 18 causal pathways
EDGE_DATA = [
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
# data/constants.py (add at the bottom)
from models.node import MentalModelNode
from models.edge import MentalModelEdge

def build_base_nodes_and_edges():
    nodes = {}
    for name, vals in NODE_DATA.items():
        ch, ac, co, corr_ca, corr_ac, corr_cc = vals
        nodes[name] = MentalModelNode(name, ch, ac, co, corr_ca, corr_ac, corr_cc)

    edges = [MentalModelEdge(s, t, c, r2) for s, t, c, r2 in EDGE_DATA]
    return nodes, edges
