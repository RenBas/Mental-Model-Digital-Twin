# analysis/sensitivity.py

import numpy as np
import pandas as pd
from engine.analytics import CommunityAnalytics

def run_sensitivity(twin, param_type, chosen_construct, component, start_val, end_val, n_steps):
    if param_type not in ("Flood Severity", "CAC Construct"):
        raise ValueError(f"Unknown param_type: {param_type}")
    if param_type == "CAC Construct":
        if not isinstance(chosen_construct, str) or not isinstance(component, str):
            raise ValueError("For CAC Construct, chosen_construct and component must be strings")
        if component not in ("Challenge", "Acceptance", "Commitment"):
            raise ValueError(f"Unknown component: {component}")
        if chosen_construct not in twin.nodes:
            raise ValueError(f"Construct not found: {chosen_construct}")

    baseline_scores = {name: node.current_score for name, node in twin.nodes.items()}
    if param_type == "CAC Construct":
        orig_agent_states = [agent.node_states.get(chosen_construct, 50.0) for agent in twin.agents]

    values = np.linspace(start_val, end_val, n_steps)
    results = []
    final_node_scores = None

    for val in values:
        if param_type == "Flood Severity":
            twin.flood_severity = val
        else:
            node = twin.nodes[chosen_construct]
            if component == "Challenge":
                node.update_cac(challenge=val)
            elif component == "Acceptance":
                node.update_cac(acceptance=val)
            else:
                node.update_cac(commitment=val)
            for agent in twin.agents:
                agent.node_states[chosen_construct] = node.current_score

        for agent in twin.agents:
            agent.reset_decisions()
        for agent in twin.agents:
            agent.evaluate_decisions(twin.flood_severity, twin.lgu_threat)

        twin.analytics = CommunityAnalytics(twin.agents, list(twin.nodes.keys()))
        metrics = twin.get_metrics()
        advanced = twin.get_advanced_metrics()
        results.append({
            'Parameter Value': val,
            'Relocate %': metrics['Projected to Relocate (%)'],
            'Evacuating %': metrics['Evacuating (%)'],
            'Resisting LGU %': metrics['Resisting LGU (%)'],
            'Proactive %': advanced['Proactive Preparedness (%)'],
            'LGU Trust %': advanced['LGU Trust & Cooperation (%)'],
            'Heritage Refusal %': advanced['Heritage-Based Refusal (%)'],
            'Demolition Anxiety %': advanced['Demolition Anxiety (%)'],
            'Relocation Readiness %': advanced['Relocation Readiness (%)']
        })

    final_node_scores = {name: node.current_score for name, node in twin.nodes.items()}
    return pd.DataFrame(results), baseline_scores, final_node_scores
