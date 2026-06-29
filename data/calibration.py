# data/calibration.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from data.constants import col_map, generate_lgu_cluster_name
from models.archetype import ClusterArchetype

def run_calibration(df, k_mode="Auto (silhouette)", manual_k=None):
    """
    Given a filtered dataframe, run K‑Means on the 36 CAC variables,
    return (cluster_profiles, chosen_k, labeled_dataframe).
    """
    numeric_cols = [c for c in df.columns if c not in ['Respondent_Name', 'Barangay_Name']]
    df_num = df[numeric_cols].copy()
    scaler = MinMaxScaler(feature_range=(0, 100))
    X_scaled = scaler.fit_transform(df_num)

    if k_mode == "Auto (silhouette)":
        best_k = 3
        best_sil = -1
        for k in range(2, min(6, len(df))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        chosen_k = best_k
    elif k_mode == "Fixed (3 clusters)":
        chosen_k = 3
    else:
        chosen_k = manual_k if manual_k else 3

    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(X_scaled)

    df_labeled = df.copy()
    df_labeled['Cluster'] = final_labels

    df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
    df_scaled['Cluster'] = final_labels

    new_profiles = {}
    for i in range(chosen_k):
        cluster_data = df_scaled[df_scaled['Cluster'] == i]
        ratio = len(cluster_data) / len(df_scaled)
        centroids = cluster_data[numeric_cols].mean().to_dict()
        base_name, driver = generate_lgu_cluster_name(centroids, col_map)

        final_name = base_name
        counter = 1
        while final_name in new_profiles:
            final_name = f"{base_name} (Segment {counter})"
            counter += 1

        new_profiles[final_name] = ClusterArchetype(
            name=final_name,
            population_ratio=ratio,
            node_baseline_scores=centroids,
            dominant_driver=driver
        )

    return new_profiles, chosen_k, df_labeled
