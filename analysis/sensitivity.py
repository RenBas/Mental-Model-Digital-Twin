# analysis/sensitivity.py

import numpy as np
import pandas as pd

def run_sensitivity(twin, param_type, chosen_construct, component, start_val, end_val, n_steps):
    """
    Fast parameter sweep – temporarily adjust one parameter, re‑evaluate decisions,
    and record behavioural/advanced metrics.  The twin is restored after the sweep.
    Returns a DataFrame with columns: Parameter Value, Relocate %, Evacuating %, ...
    """
    # Save original state
    orig_flood_severity = twin.flood_severity
    if param_type == "CAC Construct":
        node = twin.nodes[chosen_construct]
        orig_challenge = node.baseline_cac['Challenge']
        orig_acceptance = node.baseline_cac['Acceptance']
        orig_commitment = node.baseline_cac['Commitment']
        orig_score = node.current_score
        orig_agent_states = [agent.node_states.get(chosen_construct, 50.0) for agent in twin.agents]

    values = np.linspace(start_val, end_val, n_steps)
    results = []

    for val in values:
        if param_type == "Flood Severity":
            twin.flood_severity = val
        else:
            if component == "Challenge":
                node.update_cac(challenge=val)
            elif component == "Acceptance":
                node.update_cac(acceptance=val)
            else:
                node.update_cac(commitment=val)
            # Push the new score to all agents for this construct
            for agent in twin.agents:
                agent.node_states[chosen_construct] = node.current_score

        # Re‑evaluate decisions
        for agent in twin.agents:
            agent.evaluate_decisions(twin.flood_severity, twin.lgu_threat)

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

    # Restore original state
    twin.flood_severity = orig_flood_severity
    if param_type == "CAC Construct":
        node.update_cac(challenge=orig_challenge, acceptance=orig_acceptance, commitment=orig_commitment)
        node.current_score = orig_score
        for agent, orig_state in zip(twin.agents, orig_agent_states):
            agent.node_states[chosen_construct] = orig_state
        for agent in twin.agents:
            agent.evaluate_decisions(twin.flood_severity, twin.lgu_threat)

    return pd.DataFrame(results)
