import streamlit as st
from data.constants import NODE_DATA, EDGE_DATA, col_map
from models.node import MentalModelNode
from models.edge import MentalModelEdge
from engine.twin import DigitalTwin
from data.calibration import run_calibration
import pandas as pd

# Build nodes and edges from constants
nodes = {name: MentalModelNode(name, *vals) for name, vals in NODE_DATA.items()}
edges = [MentalModelEdge(s, t, c, r2) for s, t, c, r2 in EDGE_DATA]

st.write("Modules loaded successfully!")

# Quick test: load a mock CSV (if available) and run calibration
uploaded = st.file_uploader("Upload CSV")
if uploaded:
    df = pd.read_csv(uploaded)
    profiles, k, labeled = run_calibration(df, "Auto (silhouette)")
    st.write(f"K = {k}, clusters: {list(profiles.keys())}")
