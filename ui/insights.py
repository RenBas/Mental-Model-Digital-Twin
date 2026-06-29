import streamlit as st

def render_policy_insights(twin, metrics, advanced, flood_sev, label, barangay_title):
    pop = metrics['Total Population']
    reloc_pct = metrics['Projected to Relocate (%)']
    evac_pct = metrics['Evacuating (%)']
    resist_pct = metrics['Resisting LGU (%)']

    insight_parts = []
    insight_parts.append(
        f"**{barangay_title}** analysis covers **{pop:,} residents** (uploaded survey data). "
        f"Under the current PAGASA advisory ({label}, severity {flood_sev:.2f}), "
        f"**{reloc_pct:.1f}%** are projected to relocate, **{evac_pct:.1f}%** are prepared to evacuate, "
        f"and **{resist_pct:.1f}%** show resistance to LGU initiatives. "
        f"Advanced indicators: proactive preparedness {advanced['Proactive Preparedness (%)']:.1f}%, "
        f"LGU trust {advanced['LGU Trust & Cooperation (%)']:.1f}%, "
        f"heritage refusal {advanced['Heritage-Based Refusal (%)']:.1f}%, "
        f"demolition anxiety {advanced['Demolition Anxiety (%)']:.1f}%, "
        f"relocation readiness {advanced['Relocation Readiness (%)']:.1f}%."
    )

    # Relocation insights
    if reloc_pct > 30:
        insight_parts.append(
            f"🔴 **High relocation pressure ({reloc_pct:.1f}%):** This indicates strong desire or feasibility for moving, "
            "likely driven by elevated 'Desire for relocation' and 'Feasibility of relocation' scores in the network graph. "
            "If unmanaged, this could lead to unplanned out‑migration or strain on resettlement programs."
        )
    elif reloc_pct > 10:
        insight_parts.append(
            f"🟡 **Moderate relocation interest ({reloc_pct:.1f}%):** A notable segment considers moving, "
            "but most residents prefer to stay. The cluster breakdown can pinpoint which groups are relocation‑ready "
            "so that assistance can be targeted without triggering unnecessary displacement."
        )
    else:
        insight_parts.append(
            f"🟢 **Low relocation intent ({reloc_pct:.1f}%):** The majority of residents are rooted in place. "
            "Policies should focus on in‑situ adaptation, livelihood support, and strengthening local capacities "
            "rather than resettlement."
        )

    # Evacuation insights
    if evac_pct > 80:
        insight_parts.append(
            f"🔵 **Very high evacuation readiness ({evac_pct:.1f}%):** Almost the entire community is willing to evacuate "
            "when warned. This reflects strong 'Coping during flooding' and 'Prevention and flooding' scores. "
            "Maintain early warning systems and conduct regular drills to sustain this readiness."
        )
    elif evac_pct > 50:
        insight_parts.append(
            f"🟡 **Moderate evacuation readiness ({evac_pct:.1f}%):** More than half would evacuate, but a significant "
            "minority remains hesitant. The cluster chart highlights which groups are least likely to evacuate; "
            "targeted awareness campaigns and incentives could raise this figure."
        )
    else:
        insight_parts.append(
            f"🔴 **Low evacuation readiness ({evac_pct:.1f}%):** A majority of residents may not respond to evacuation orders. "
            "This is a critical gap in disaster preparedness. Strengthen 'Coping during flooding' and 'Viewpoints towards LGU' "
            "through community engagement and trust‑building, as suggested by the network graph."
        )

    # Resistance insights
    if resist_pct > 20:
        insight_parts.append(
            f"🟠 **Significant LGU resistance ({resist_pct:.1f}%):** Friction is apparent, especially if a demolition threat "
            "is active. The CAC profile likely shows high 'Fear of housing demolition' and low 'Viewpoints towards LGU'. "
            "Immediate action: issue housing tenure guarantees and open a transparent dialogue with affected clusters."
        )
    elif resist_pct > 5:
        insight_parts.append(
            f"🟡 **Low‑to‑moderate resistance ({resist_pct:.1f}%):** A small fraction resists LGU efforts. "
            "Monitor the clusters that contribute to this resistance; even a few vocal opponents can escalate tensions."
        )
    else:
        insight_parts.append(
            f"🟢 **Negligible resistance ({resist_pct:.1f}%):** The community is largely cooperative, "
            "providing a favourable environment for new programs and policies."
        )

    # Advanced indicator triggers
    if advanced['Proactive Preparedness (%)'] > 60:
        insight_parts.append("🛠️ **High proactive preparedness** – community champions exist; engage them as partners.")
    if advanced['LGU Trust & Cooperation (%)'] < 40:
        insight_parts.append("🤝 **Low LGU trust** – major barrier to policy rollouts; invest in trust‑building before launching new programs.")
    if advanced['Heritage-Based Refusal (%)'] > 30:
        insight_parts.append("🏡 **High heritage‑based refusal** – relocation programs must include psychosocial and cultural components, not just financial aid.")
    if advanced['Demolition Anxiety (%)'] > 30:
        insight_parts.append("⚠️ **Elevated demolition anxiety** – deploy MHPSS and issue clear housing guarantees immediately.")
    if advanced['Relocation Readiness (%)'] > 20:
        insight_parts.append("🚀 **Relocation readiness high** – a pilot relocation program can succeed quickly if targeted at these early adopters.")

    # Network graph specific triggers
    if twin.nodes["Fear of housing demolition"].current_score > 60:
        insight_parts.append(
            "⚠️ **Elevated fear of demolition** – The network shows this node is particularly hot (high score). "
            "It will likely amplify resistance and reduce relocation willingness. Address it with clear communication "
            "about housing security before any infrastructure project."
        )
    if twin.nodes["Viewpoints towards LGU"].current_score < 40:
        insight_parts.append(
            "⚠️ **Low LGU trust** – Trust is a multiplier across all behaviours. "
            "Implement visible, participatory projects to improve this score; the network shows it directly influences "
            "'Assistance for relocation' and 'Fear of housing demolition'."
        )

    # Combined profiles
    if reloc_pct < 10 and evac_pct > 80 and resist_pct < 5:
        insight_parts.append(
            "✅ **Optimal profile:** This barangay exhibits high resilience with low resistance and relocation pressure. "
            "The primary strategy is to **maintain current interventions** and monitor for any shifts, especially if "
            "environmental conditions change."
        )
    elif reloc_pct > 30 and evac_pct < 50:
        insight_parts.append(
            "⚠️ **Dual vulnerability:** Many want to leave but few would evacuate in an emergency. "
            "This contradictory pattern suggests deep‑seated distrust or fear. Prioritise building evacuation capacity "
            "while simultaneously offering voluntary, dignified relocation options."
        )

    st.markdown(" ".join(insight_parts))
    st.caption(
        "🔗 **How to use these insights:** The network graph identifies which psychological drivers to adjust, "
        "the cluster breakdown shows exactly which groups drive each behaviour, and the CAC bubble chart reveals "
        "the underlying community profile. Adjust the intervention sliders or environmental triggers, then "
        "click 'Rebuild Population' or 'Run' to test policy scenarios."
    )
