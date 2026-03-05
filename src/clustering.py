"""Clustering module for bag-of-entities analysis.

Handles:
- Building artist × entity sparse frequency matrix
- K-means clustering
- Cluster summarization
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class EntityMatrix:
    """Sparse artist × entity frequency matrix.

    Wraps a scipy CSR matrix with artist/entity index metadata so the rest
    of the pipeline can stay simple while memory stays low.
    """

    def __init__(self, data: csr_matrix, artists: List[str], entities: List[str]):
        self.data = data
        self.artists = artists
        self.entities = entities

    @property
    def empty(self) -> bool:
        return self.data.shape[0] == 0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def __len__(self) -> int:
        return self.data.shape[0]

    def to_dense_df(self) -> pd.DataFrame:
        """Convert to a dense pandas DataFrame (for CSV export / inspection)."""
        return pd.DataFrame(
            self.data.toarray(),
            index=self.artists,
            columns=self.entities,
        )

    def save_npz(self, path: str) -> None:
        """Save in efficient sparse format (.npz + metadata csv)."""
        from pathlib import Path
        from scipy.sparse import save_npz
        p = Path(path)
        save_npz(str(p.with_suffix(".npz")), self.data)
        meta = pd.DataFrame({"artist": self.artists})
        meta.to_csv(str(p.with_suffix(".artists.csv")), index=False)
        pd.DataFrame({"entity": self.entities}).to_csv(
            str(p.with_suffix(".entities.csv")), index=False
        )

    @classmethod
    def load_npz(cls, path: str) -> "EntityMatrix":
        """Load from sparse format saved by save_npz."""
        from pathlib import Path
        from scipy.sparse import load_npz
        p = Path(path)
        data = load_npz(str(p.with_suffix(".npz")))
        artists = pd.read_csv(str(p.with_suffix(".artists.csv")))["artist"].tolist()
        entities = pd.read_csv(str(p.with_suffix(".entities.csv")))["entity"].tolist()
        return cls(data, artists, entities)


def build_bag_of_entities(entity_df: pd.DataFrame) -> EntityMatrix:
    """Build a sparse artist × entity frequency matrix from long-form entity data.

    Uses scipy CSR format — memory-efficient for the typical 99%+ sparsity
    in lyrics NER data. A 241 × 21k matrix at 99.1% sparsity uses ~200KB
    instead of ~40MB dense.
    """
    if entity_df.empty:
        return EntityMatrix(csr_matrix((0, 0)), [], [])

    counts = entity_df.groupby(["artist", "entity"]).size().reset_index(name="count")

    # Encode artists and entities as integer indices
    artist_cat = pd.Categorical(counts["artist"])
    entity_cat = pd.Categorical(counts["entity"])

    sparse = csr_matrix(
        (counts["count"].values, (artist_cat.codes, entity_cat.codes)),
        shape=(len(artist_cat.categories), len(entity_cat.categories)),
        dtype=np.float32,
    )

    return EntityMatrix(
        data=sparse,
        artists=list(artist_cat.categories),
        entities=list(entity_cat.categories),
    )


def run_kmeans(
    entity_matrix: EntityMatrix,
    n_clusters: int = 6,
    random_state: int = 42,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run K-means on the sparse entity matrix.

    Parameters
    ----------
    normalize : If True, apply L2-normalization per artist before clustering.
        This prevents artists with more lyrics from dominating.

    Returns
    -------
    assignments : DataFrame with columns ['artist', 'cluster']
    centroids : DataFrame indexed by cluster name, columns = entities
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize as sklearn_normalize

    if entity_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    n_clusters = min(n_clusters, len(entity_matrix))

    mat = entity_matrix.data.astype(np.float32)
    if normalize:
        mat = sklearn_normalize(mat, norm="l2")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state, batch_size=64
    )
    clusters = kmeans.fit_predict(mat)

    assignments = pd.DataFrame({
        "artist": entity_matrix.artists,
        "cluster": clusters,
    })
    return assignments, kmeans.cluster_centers_


def summarize_clusters(
    centers: np.ndarray, entities: List[str], top_k: int = 25
) -> pd.DataFrame:
    """Extract top-k entities per cluster ranked by centroid weight.

    Parameters
    ----------
    centers : numpy array of shape (n_clusters, n_entities)
    entities : list of entity names matching columns of centers
    top_k : number of top entities per cluster
    """
    summaries = []
    for i, row in enumerate(centers):
        top_idx = np.argsort(row)[::-1][:top_k]
        for j in top_idx:
            summaries.append({
                "cluster": f"cluster_{i}",
                "entity": entities[j],
                "centroid_weight": round(float(row[j]), 4),
            })
    return pd.DataFrame(summaries)
