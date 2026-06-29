# ui/charts.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

def render_network_graph(twin, barangay_title, node_scores=None):
    """
    Renders the 12‑node psychological network graph.
    If node_scores dict is provided, those scores are used instead of twin.nodes.current_score.
    """
    G = nx.DiGraph()
    if node_scores is not None:
        # Use provided scores (e.g., baseline frozen scores)
        for name, score in node_scores.items():
            G.add_node(name, score=score)
    else:
        for name, node in twin.nodes.items():
            G.add_node(name, score=node.current_score)
    for edge in twin.edges:
        G.add_edge(edge.source_name, edge.target_name, weight=edge.coefficient)

    pos = nx.spring_layout(G, k=0.35, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'),
                            hoverinfo='text',
                            text=[f"{edge.source_name} → {edge.target_name}<br>Regression coeff.: {edge.coefficient:+.3f}<br>R²: {edge.r_square:.3f}" for edge in twin.edges],
                            mode='lines')
    node_x, node_y, node_text, node_color = [], [], [], []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(f"{n}<br>Score: {d['score']:.1f}")
        node_color.append(d['score'])
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=list(G.nodes()), textposition="bottom center",
                            hovertext=node_text, hoverinfo='text',
                            marker=dict(showscale=True, colorscale='Viridis', reversescale=True,
                                        color=node_color, size=25,
                                        colorbar=dict(thickness=10, title='Score')))
    fig_net = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='12 Nodes & 18 Causal Pathways', showlegend=False,
                                         margin=dict(b=20,l=5,r=5,t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig_net, use_container_width=True)

    high_nodes = [n for n, d in G.nodes(data=True) if d['score'] > 60]
    low_nodes = [n for n, d in G.nodes(data=True) if d['score'] < 40]
    if high_nodes:
        st.caption(f"🔵 **Strong drivers in {barangay_title}:** {', '.join(high_nodes)} show high intensity, which can be leveraged for positive behavioral change.")
    if low_nodes:
        st.caption(f"🟡 **Weak areas in {barangay_title}:** {', '.join(low_nodes)} are psychologically fragile; interventions here may have the greatest impact.")
    if not high_nodes and not low_nodes:
        st.caption(f"⚪ **Balanced profile in {barangay_title}:** all nodes are in the moderate range, suggesting a stable psychological landscape.")

def render_cluster_breakdown(cluster_df, barangay_title):
    st.caption("Overall metrics are the weighted average of these per‑cluster percentages. Population counts shown in the table below.")
    fig_cluster = go.Figure()
    behaviors = ['Projected to Relocate %', 'Evacuating %', 'Adapting %', 'Resisting %']
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    for i, beh in enumerate(behaviors):
        fig_cluster.add_trace(go.Bar(
            x=cluster_df['Cluster'],
            y=cluster_df[beh],
            name=beh,
            text=[f"{v:.1f}%" for v in cluster_df[beh]],
            textposition='outside',
            marker_color=colors[i]
        ))
    fig_cluster.update_layout(barmode='group', title='Behavioral Distribution by Cluster',
                              yaxis_title='Percentage', height=450)
    st.plotly_chart(fig_cluster, use_container_width=True)

    dominating_clusters = []
    for _, row in cluster_df.iterrows():
        if row['Projected to Relocate %'] > 50 or row['Evacuating %'] > 60 or row['Resisting %'] > 30:
            dominating_clusters.append(row['Cluster'])
    if dominating_clusters:
        st.caption(f"🔍 **Clusters driving {barangay_title} outcomes:** {', '.join(dominating_clusters)} have notably high behavioral percentages. Targeting these groups can shift aggregate outcomes significantly.")
    else:
        st.caption(f"🔍 **Balanced cluster behavior in {barangay_title}:** no single cluster dominates the outcomes, indicating a mixed community profile.")

    st.markdown("**Cluster populations:**")
    pop_summary = cluster_df[['Cluster', 'Population Count']].set_index('Cluster')
    st.dataframe(pop_summary.T, use_container_width=True)
    st.caption("The sum of these counts equals the total population shown above.")

def render_cac_bubble(cac_avgs, col_map, barangay_title):
    scatter = [{"Construct": node, "Challenge": cac_avgs[f"{node} (Challenge)"],
                "Acceptance": cac_avgs[f"{node} (Acceptance)"],
                "Commitment": cac_avgs[f"{node} (Commitment)"]} for node in col_map]
    fig_cac = px.scatter(pd.DataFrame(scatter), x="Challenge", y="Acceptance",
                         size="Commitment", color="Construct", size_max=60,
                         title="Community CAC Profile")
    st.plotly_chart(fig_cac, use_container_width=True)

    high_challenge = [d['Construct'] for d in scatter if d['Challenge'] > 60]
    low_acceptance = [d['Construct'] for d in scatter if d['Acceptance'] < 40]
    if high_challenge:
        st.caption(f"🔴 **High Challenge in {barangay_title}:** {', '.join(high_challenge)} – residents perceive significant barriers, requiring policy support to lower perceived difficulty.")
    if low_acceptance:
        st.caption(f"🟠 **Low Acceptance in {barangay_title}:** {', '.join(low_acceptance)} – these areas lack community buy‑in; awareness campaigns or incentives may be needed.")
    if not high_challenge and not low_acceptance:
        st.caption(f"⚪ **Balanced CAC profile in {barangay_title}:** all constructs are within moderate range, indicating a generally stable psychological state.")
