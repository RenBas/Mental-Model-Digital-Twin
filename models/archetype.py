class ClusterArchetype:
    def __init__(self, name, population_ratio, node_baseline_scores, dominant_driver,
                 relocation_threshold=35.0, resistance_threshold=35.0):
        self.name = name
        self.population_ratio = population_ratio
        self.node_baseline_scores = node_baseline_scores  # dict, may contain cleaned names or construct names
        self.dominant_driver = dominant_driver
        self.relocation_threshold = relocation_threshold
        self.resistance_threshold = resistance_threshold
