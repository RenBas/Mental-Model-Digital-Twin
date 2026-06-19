# Digital Twin: Socio-Psychological Dynamics of Flood-Prone Communities

## Overview
This repository contains the source code for a **Digital Twin** designed to simulate and visualize the socio-psychological dynamics of residents living in flood-prone areas. Initially calibrated using baseline data from Sitio Dal-og, Tagoloan, Misamis Oriental, this tool is scalable and intended for municipal-wide application across the Municipality of Tagoloan.

The Digital Twin bridges the physical and virtual worlds by combining **System Dynamics** (macro-level psychological networks) with **Agent-Based Modeling** (micro-level individual resident behaviors) to help Local Government Units (LGUs) and researchers test "What-If" intervention scenarios before deploying real-world resources.

## Key Features
* **12-Node Mental Model:** Maps the core socio-psychological constructs (e.g., *Desire for relocation, Fear of housing demolition, Viewpoints towards LGU*).
* **18 Causal Pathways:** Utilizes empirically derived regression coefficients to simulate how interventions ripple through the community's psychology over time.
* **Agent-Based Population:** Simulates a dynamic population of virtual residents (scalable from 140 to 5,000+) categorized by distinct socio-psychological profiles derived from K-Means cluster analysis.
* **Interactive "What-If" Engine:** Allows LGU planners to adjust environmental triggers (Flood Severity, Demolition Threats) and LGU interventions via a real-time dashboard.
* **Streamlit Dashboard:** A highly visual, interactive web interface featuring network graphs, KPI metrics, and time-series projections.

## Architecture
The application is built on a 3-Layer Architecture:
1. **Layer 1: Mathematical Engine (System Dynamics):** Handles the 12 nodes, 18 edges, and time-step propagation with a psychological damping factor.
2. **Layer 2: Agent Population (Agent-Based Modeling):** Generates the virtual residents, applies K-means archetypes, and calculates utility-based behavioral thresholds (Relocation, Evacuation, Adaptation, Resistance).
3. **Layer 3: Interactive UI (Streamlit):** The frontend dashboard that binds user inputs to the backend engines and renders real-time Plotly/NetworkX visualizations.

## Prerequisites
* Python 3.9 or higher
* pip (Python package manager)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[YOUR-USERNAME]/[YOUR-REPO-NAME].git
   cd [YOUR-REPO-NAME]
