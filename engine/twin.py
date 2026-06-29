# engine/twin.py

import numpy as np
from engine.generator import PopulationGenerator
from engine.analytics import CommunityAnalytics

class DigitalTwin:
    def __init__(self, nodes, edges, cluster_profiles, total_population, flood_severity, lgu_threat,
                 damping_factor=0.5, seed=None, col_map=None):
        self.nodes = nodes
        self.edges = edges
        self.damping_factor = damping_factor
        self.flood_severity = flood_severity
        self.lgu_threat = lgu_threat
        self.cluster_profiles = cluster_profiles if cluster_profiles else {}
        self.total_population = total_population
        self.step_count = 0
        self.history = []

        self.incoming_edges = {name: [] for name in self.nodes.keys()}
        for edge in self.edges:
            self.incoming_edges[edge.target_name].append((edge.source_name, edge.coefficient))

        # Initialise generator with the provided col_map
        if col_map is None:
            raise ValueError("col_map must be provided to DigitalTwin")
        self.generator = PopulationGenerator(col_map)

        self.agents = self.generator.generate_population(
            self.total_population, self.cluster_profiles, self.nodes, seed=seed
        )
        for agent in self.agents:
            agent.evaluate_decisions(self.flood_severity, self.lgu_threat)
        self.analytics = CommunityAnalytics(self.agents, list(self.nodes.keys()))

    def step(self):
        if self.total_population == 0:
            return
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
            node.current_score = node._compute_score()
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

    def get_advanced_metrics(self):
        return self.analytics.get_advanced_metrics()
