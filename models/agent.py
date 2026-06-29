class ResidentAgent:
    def __init__(self, agent_id, cluster_name, node_states, cac_states, archetype):
        self.agent_id = agent_id
        self.cluster_name = cluster_name
        self.node_states = node_states      # dict: construct name -> macro score (0-100)
        self.cac_states = cac_states        # dict: cleaned_name_Challenge etc.
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

        relocation_utility = (desire*0.35) + (feasibility*0.30) + (assistance*0.25) - (fear*0.05) - (family_ties*0.05)
        relocation_utility = max(0, min(100, relocation_utility))
        if relocation_utility >= self.archetype.relocation_threshold and not self.has_relocated:
            self.has_relocated = True

        coping = self.node_states.get("Coping during flooding", 50.0)
        prevention = self.node_states.get("Prevention and flooding", 50.0)
        lgu_view = self.node_states.get("Viewpoints towards LGU", 50.0)
        evac_utility = (coping*0.4) + (prevention*0.3) + (lgu_view*0.3)
        self.will_evacuate = evac_utility >= (50.0 + 20.0 * (1.0 - flood_severity))

        preference = self.node_states.get("Preference and adaptation", 50.0)
        self.is_adapting_in_place = (not self.has_relocated) and (preference >= 40.0)

        if lgu_demolition_threat:
            rights = self.node_states.get("Rights to live in the area", 50.0)
            resist_utility = (rights*0.5) + (family_ties*0.3) + (fear*0.2)
            self.is_resisting_lgu = resist_utility >= self.archetype.resistance_threshold
        else:
            self.is_resisting_lgu = False
