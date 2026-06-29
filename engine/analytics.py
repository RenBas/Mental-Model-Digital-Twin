import pandas as pd
import numpy as np

class CommunityAnalytics:
    def __init__(self, agents_list, node_names):
        self.agents = agents_list
        self.node_names = node_names
        self.total_population = len(agents_list)
        self._build_dataframe()

    def _build_dataframe(self):
        data = []
        for agent in self.agents:
            row = {
                'Agent_ID': agent.agent_id,
                'Cluster': agent.cluster_name,
                'Has_Relocated': agent.has_relocated,
                'Will_Evacuate': agent.will_evacuate,
                'Is_Adapting': agent.is_adapting_in_place,
                'Is_Resisting': agent.is_resisting_lgu
            }
            for node_name in self.node_names:
                row[node_name] = agent.node_states.get(node_name, 0.0)
            data.append(row)
        self.df = pd.DataFrame(data)

    def get_behavioral_metrics(self):
        if self.total_population == 0:
            return {
                "Total Population": 0,
                "Projected to Relocate (%)": 0.0,
                "Evacuating (%)": 0.0,
                "Adapting In-Place (%)": 0.0,
                "Resisting LGU (%)": 0.0
            }
        return {
            "Total Population": self.total_population,
            "Projected to Relocate (%)": (self.df['Has_Relocated'].sum() / self.total_population) * 100,
            "Evacuating (%)": (self.df['Will_Evacuate'].sum() / self.total_population) * 100,
            "Adapting In-Place (%)": (self.df['Is_Adapting'].sum() / self.total_population) * 100,
            "Resisting LGU (%)": (self.df['Is_Resisting'].sum() / self.total_population) * 100
        }

    def get_advanced_metrics(self):
        if self.total_population == 0:
            return {
                "Proactive Preparedness (%)": 0.0,
                "LGU Trust & Cooperation (%)": 0.0,
                "Heritage-Based Refusal (%)": 0.0,
                "Demolition Anxiety (%)": 0.0,
                "Relocation Readiness (%)": 0.0
            }
        proactive = self.df[
            (self.df["Prevention and flooding"] > 60) & 
            (self.df["Coping during flooding"] > 60)
        ].shape[0]
        proactive_pct = (proactive / self.total_population) * 100

        trust_coop = self.df[
            (self.df["Viewpoints towards LGU"] > 50) & 
            (self.df["Assistance for relocation"] > 30)
        ].shape[0]
        trust_pct = (trust_coop / self.total_population) * 100

        heritage_refusal = self.df[
            (self.df["Has_Relocated"] == False) & 
            (self.df["Family history and identity"] > 48)
        ].shape[0]
        heritage_pct = (heritage_refusal / self.total_population) * 100

        anxiety = self.df[self.df["Fear of housing demolition"] > 50].shape[0]
        anxiety_pct = (anxiety / self.total_population) * 100

        ready = self.df[
            (self.df["Has_Relocated"] == True) & 
            (self.df["Preference and adaptation"] > 60)
        ].shape[0]
        readiness_pct = (ready / self.total_population) * 100

        return {
            "Proactive Preparedness (%)": proactive_pct,
            "LGU Trust & Cooperation (%)": trust_pct,
            "Heritage-Based Refusal (%)": heritage_pct,
            "Demolition Anxiety (%)": anxiety_pct,
            "Relocation Readiness (%)": readiness_pct
        }

    def get_cluster_breakdown(self):
        if self.total_population == 0:
            return pd.DataFrame(columns=['Cluster', 'Population Count', 'Projected to Relocate %', 'Evacuating %', 'Adapting %', 'Resisting %'])
        cluster_stats = self.df.groupby('Cluster').agg(
            Count=('Agent_ID', 'count'),
            Relocated=('Has_Relocated', 'mean'),
            Evacuating=('Will_Evacuate', 'mean'),
            Adapting=('Is_Adapting', 'mean'),
            Resisting=('Is_Resisting', 'mean')
        )
        cluster_stats[['Relocated','Evacuating','Adapting','Resisting']] *= 100
        cluster_stats.columns = ['Population Count', 'Projected to Relocate %',
                                 'Evacuating %', 'Adapting %', 'Resisting %']
        return cluster_stats.reset_index()

    def get_cac_averages(self, col_map):
        cac_avgs = {}
        for node, clean in col_map.items():
            for cac in ['Challenge', 'Acceptance', 'Commitment']:
                key = f"{clean}_{cac}"
                vals = [getattr(agent, 'cac_states', {}).get(key, 0) for agent in self.agents]
                cac_avgs[f"{node} ({cac})"] = np.mean(vals) if vals else 0
        return cac_avgs
