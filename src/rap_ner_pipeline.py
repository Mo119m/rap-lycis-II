#!/usr/bin/env python3
"""Chinese rap NER + bag-of-entities + clustering pipeline.

This script is designed as a research starter pipeline for colloquial Chinese lyrics.
It supports:
1) Loading lyric chunks and grouping text by artist.
2) Running NER with spaCy + optional custom EntityRuler lexicon.
3) Building bag-of-entities matrices.
4) Running k-means clustering and saving interpretable summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import spacy
from spacy.language import Language


DEFAULT_STOP_LABELS = {"CARDINAL", "DATE", "TIME", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NER pipeline for Chinese rap lyrics")
    parser.add_argument("--input", default="lyrics_chunks_enriched.csv", help="Input CSV path")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--lexicon", default="configs/rap_lexicon_seed.jsonl", help="EntityRuler patterns jsonl")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for quick experiments (0 = all)")
    parser.add_argument("--min-entity-len", type=int, default=2, help="Minimum entity text length")
    parser.add_argument("--top-k", type=int, default=25, help="Top entities to show per cluster")
    parser.add_argument("--n-clusters", type=int, default=6, help="K in k-means")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data(path: str, max_rows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"artist", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()
    df = df.dropna(subset=["artist", "text"])
    return df


def combine_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    artist_lyrics = df.groupby("artist", as_index=False)["text"].apply(lambda x: "\n".join(x.astype(str)))
    artist_lyrics = artist_lyrics.rename(columns={"text": "combined_text"})
    return artist_lyrics


def build_nlp(lexicon_path: str) -> Language:
    """Try loading zh_core_web_lg; fallback to blank zh + EntityRuler."""
    try:
        nlp = spacy.load("zh_core_web_lg")
        print("[INFO] Loaded spaCy model: zh_core_web_lg")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to load zh_core_web_lg ({exc}); using blank Chinese pipeline.")
        nlp = spacy.blank("zh")

    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", last=True, config={"overwrite_ents": False})
    else:
        ruler = nlp.get_pipe("entity_ruler")

    lexicon_file = Path(lexicon_path)
    if lexicon_file.exists():
        patterns = []
        with lexicon_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                patterns.append(json.loads(line))
        if patterns:
            ruler.add_patterns(patterns)
            print(f"[INFO] Added {len(patterns)} lexicon patterns from {lexicon_file}")
    else:
        print(f"[WARN] Lexicon not found: {lexicon_file}")

    return nlp


def extract_entities(
    artist_lyrics: pd.DataFrame,
    nlp: Language,
    min_entity_len: int = 2,
    stop_labels: Iterable[str] = DEFAULT_STOP_LABELS,
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in artist_lyrics.iterrows():
        artist = row["artist"]
        doc = nlp(row["combined_text"])
        for ent in doc.ents:
            text = ent.text.strip()
            if len(text) < min_entity_len:
                continue
            if ent.label_ in stop_labels:
                continue
            rows.append({"artist": artist, "entity": text, "label": ent.label_})
    return pd.DataFrame(rows)


def build_bag_of_entities(entity_df: pd.DataFrame) -> pd.DataFrame:
    if entity_df.empty:
        return pd.DataFrame()
    entity_counts = (
        entity_df.groupby(["artist", "entity"]).size().reset_index(name="count")
        .pivot(index="artist", columns="entity", values="count")
        .fillna(0)
        .astype(int)
    )
    return entity_counts


def run_kmeans(entity_matrix: pd.DataFrame, n_clusters: int, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scikit-learn is required for clustering.") from exc

    if entity_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = kmeans.fit_predict(entity_matrix)
    assignments = pd.DataFrame({"artist": entity_matrix.index, "cluster": clusters})

    centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=entity_matrix.columns,
        index=[f"cluster_{i}" for i in range(n_clusters)],
    )
    return assignments, centroids


def summarize_clusters(centroids: pd.DataFrame, top_k: int) -> pd.DataFrame:
    summaries = []
    for cluster_name, values in centroids.iterrows():
        top_entities = values.sort_values(ascending=False).head(top_k)
        for entity, score in top_entities.items():
            summaries.append({"cluster": cluster_name, "entity": entity, "centroid_weight": float(score)})
    return pd.DataFrame(summaries)


def save_outputs(
    output_dir: Path,
    artist_lyrics: pd.DataFrame,
    entity_df: pd.DataFrame,
    entity_matrix: pd.DataFrame,
    assignments: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artist_lyrics.to_csv(output_dir / "artist_lyrics.csv", index=False)
    entity_df.to_csv(output_dir / "entities_long.csv", index=False)
    entity_matrix.to_csv(output_dir / "bag_of_entities.csv")
    if not assignments.empty:
        assignments.to_csv(output_dir / "artist_clusters.csv", index=False)
    if not summaries.empty:
        summaries.to_csv(output_dir / "cluster_entity_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    df = load_data(args.input, args.max_rows)
    artist_lyrics = combine_by_artist(df)
    nlp = build_nlp(args.lexicon)

    entity_df = extract_entities(artist_lyrics, nlp, min_entity_len=args.min_entity_len)
    entity_matrix = build_bag_of_entities(entity_df)

    assignments = pd.DataFrame()
    summaries = pd.DataFrame()
    if not entity_matrix.empty and len(entity_matrix) >= args.n_clusters:
        assignments, centroids = run_kmeans(entity_matrix, args.n_clusters, args.random_state)
        summaries = summarize_clusters(centroids, args.top_k)
    else:
        print("[WARN] Skipping clustering: empty entity matrix or fewer artists than n_clusters.")

    save_outputs(Path(args.output_dir), artist_lyrics, entity_df, entity_matrix, assignments, summaries)

    print("\n=== Pipeline finished ===")
    print(f"Rows processed: {len(df)}")
    print(f"Artists: {len(artist_lyrics)}")
    print(f"Entities extracted: {len(entity_df)}")
    if not entity_df.empty:
        counter = Counter(entity_df["entity"])
        print("Top entities:")
        for entity, cnt in counter.most_common(10):
            print(f"  - {entity}: {cnt}")


if __name__ == "__main__":
    main()
