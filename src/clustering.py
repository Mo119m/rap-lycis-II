"""Clustering module for bag-of-entities analysis.

Handles:
- Building artist × entity frequency matrix
- K-means clustering
- Cluster summarization
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def build_bag_of_entities(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot entity DataFrame into artist × entity frequency matrix."""
    if entity_df.empty:
        return pd.DataFrame()
    entity_counts = (
        entity_df.groupby(["artist", "entity"])
        .size()
        .reset_index(name="count")
        .pivot(index="artist", columns="entity", values="count")
        .fillna(0)
        .astype(int)
    )
    return entity_counts


def run_kmeans(
    entity_matrix: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run K-means on the entity matrix.

    Returns
    -------
    assignments : DataFrame with columns ['artist', 'cluster']
    centroids : DataFrame indexed by cluster name, columns = entities
    """
    from sklearn.cluster import KMeans

    if entity_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    n_clusters = min(n_clusters, len(entity_matrix))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = kmeans.fit_predict(entity_matrix)

    assignments = pd.DataFrame({"artist": entity_matrix.index, "cluster": clusters})
    centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=entity_matrix.columns,
        index=[f"cluster_{i}" for i in range(n_clusters)],
    )
    return assignments, centroids


def summarize_clusters(centroids: pd.DataFrame, top_k: int = 25) -> pd.DataFrame:
    """Extract top-k entities per cluster ranked by centroid weight."""
    summaries = []
    for cluster_name, values in centroids.iterrows():
        top_entities = values.sort_values(ascending=False).head(top_k)
        for entity, score in top_entities.items():
            summaries.append({
                "cluster": cluster_name,
                "entity": entity,
                "centroid_weight": round(float(score), 4),
            })
    return pd.DataFrame(summaries)
