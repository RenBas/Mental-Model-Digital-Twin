    # ---------- Glossary ----------
    st.markdown("---")
    with st.expander("📖 Glossary (click to open)", expanded=False):
        glossary = [
            ("Advanced Community Indicators", "Five derived metrics (Proactive Preparedness, LGU Trust & Cooperation, Heritage‑Based Refusal, Demolition Anxiety, Relocation Readiness) that capture deeper psychological dynamics beyond basic behaviours."),
            ("Auto‑mode (PAGASA)", "When enabled, the twin automatically uses the latest PAGASA advisory severity to drive the simulation, updating the evacuation metric instantly. The simulated flood severity slider is locked."),
            ("Baseline", "The initial state of the twin right after calibration – a frozen snapshot of the psychological network and all metrics. You can always return to it with the 'Restore Baseline' button."),
            ("CAC (Challenge–Acceptance–Commitment)", "The three components that make up each psychological construct. Challenge = perceived difficulty, Acceptance = readiness to embrace, Commitment = active engagement. They are the building blocks of the 12 constructs."),
            ("Calibration", "The process of running K‑Means clustering on your uploaded survey data to create psychological profiles (clusters) and generate the agent population. Triggered by the 'Recalibrate' button."),
            ("Community Behavioral Outcomes", "The four main percentages shown at the top of the dashboard: Projected to Relocate, Evacuating, Resisting LGU, and Adapting In‑Place. They summarise how the community is likely to behave under current conditions."),
            ("Constructs (12 Psychological Constructs)", "The twelve measured dimensions (e.g., 'Fear of housing demolition', 'Viewpoints towards LGU'). They form the nodes of the psychological network and determine agent decisions."),
            ("Evacuating (%)", "The percentage of agents willing to evacuate under the current flood severity. Depends on Coping during flooding, Prevention and flooding, and Viewpoints towards LGU."),
            ("Flood Severity (Simulated)", "A value from 0 to 1 that you can adjust manually or let the PAGASA advisory control. It directly influences the evacuation threshold – higher severity makes more agents willing to evacuate."),
            ("LGU Demolition Threat", "A toggle that simulates whether demolition threats are active. When ON, some agents may resist the LGU based on their psychological scores."),
            ("Network Graph (Socio‑Psychological)", "The 12‑node, 18‑edge diagram showing how psychological constructs influence each other. Hover over an edge to see the regression coefficient and R‑squared. Colours show current score intensity."),
            ("Network Impact View", "The right‑hand graph that appears after a sensitivity analysis. It shows how much each node’s score would change from the baseline if the selected intervention were applied. Red = increase, Blue = decrease."),
            ("PAGASA Advisory", "The daily flood outlook for the Tagoloan River Basin, scraped from the official DOST‑PAGASA page. It is displayed as a coloured gauge in the sidebar and can automatically set the flood severity."),
            ("Projected Population Size", "A slider that scales the agent population up or down while keeping the same cluster proportions. It answers 'what if our barangay had X residents?'."),
            ("Rainfall Intensity (mm)", "The flood severity expressed as millimetres of rain. Used in the sensitivity analysis for flood severity. The gauge colours follow PAGASA’s official rainfall classification."),
            ("Recalibrate", "The button that re‑runs the clustering and rebuilds the twin with newly uploaded survey data. It creates a fresh baseline."),
            ("Restore Baseline", "The button that returns the twin to the exact state it was in right after the last calibration. All sensitivity changes are undone, and auto‑mode is re‑enabled."),
            ("Sensitivity Analysis", "A tool that lets you vary one parameter (flood severity or any CAC component) across a range and instantly see how behavioural and advanced indicators respond. The twin stays in the last parameter state so the whole dashboard reflects the scenario."),
            ("Sensitivity Scenario Active", "A banner that appears when a sensitivity analysis has been run. It reminds you that the current dashboard shows a 'what‑if' scenario, not the original baseline."),
            ("Snapshot (Log)", "The 'Log Current Snapshot' button saves the current dashboard metrics (including barangay, timestamp, severity, and all indicators) to a table. You can download the entire log as a CSV for later trend analysis."),
            ("Station (Monitoring)", "The 10 interactive markers on the hazard map (raingauges, water level stations, weather stations). Click one to see its details, or hover for a quick tooltip."),
            ("Water Level Gauge (future)", "An inactive gauge reserved for future real‑time river water level data from PAGASA PREDICT or the MDRRMO. Currently shown as 'No Data'."),
            ("Zoom / Pan", "The interactive map supports dragging (pan) and scrolling (zoom) to explore different parts of the Tagoloan River Basin. A download button lets you save the map as an interactive HTML file.")
        ]
        # Sort alphabetically by term
        glossary.sort(key=lambda x: x[0].lower())
        for term, definition in glossary:
            st.markdown(f"**{term}** – {definition}")
