import numpy as np
from models.agent import ResidentAgent

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
