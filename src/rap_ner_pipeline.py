#!/usr/bin/env python3
"""CLI entry point for the Chinese rap NER pipeline.

Delegates to modular components in src/.

Usage:
    python src/rap_ner_pipeline.py --input lyrics_chunks_enriched.csv --n-clusters 6
    python src/rap_ner_pipeline.py --max-rows 5000 --n-clusters 4   # quick test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_cleaning import load_and_clean, combine_by_artist
from src.ner import build_nlp, extract_entities, normalize_entities, entity_summary, apply_entity_corrections
from src.clustering import build_bag_of_entities, run_kmeans, summarize_clusters
from src.io_utils import save_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NER pipeline for Chinese rap lyrics")
    parser.add_argument("--input", default="lyrics_chunks_enriched.csv", help="Input CSV path")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--model", default="zh_core_web_trf",
                        help="spaCy model name (zh_core_web_trf or zh_core_web_lg)")
    parser.add_argument("--lexicon", default="configs/rap_lexicon_seed.jsonl",
                        help="EntityRuler patterns JSONL")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Limit rows for quick experiments (0 = all)")
    parser.add_argument("--min-entity-len", type=int, default=2,
                        help="Minimum entity text length")
    parser.add_argument("--top-k", type=int, default=25,
                        help="Top entities to show per cluster")
    parser.add_argument("--n-clusters", type=int, default=6, help="K in k-means")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--entity-review", default="configs/entity_review.csv",
                        help="Entity review CSV for manual corrections")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Auto-delete entities not in review with count < N (0=disable)")
    parser.add_argument("--global-min-count", type=int, default=0,
                        help="Also include entities with global freq >= N in review, "
                             "even if not in per-label top_n (0=disable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load & clean
    df = load_and_clean(args.input, max_rows=args.max_rows)
    artist_lyrics = combine_by_artist(df)

    # 2. NER
    nlp = build_nlp(args.lexicon, model_name=args.model)
    entity_df = extract_entities(artist_lyrics, nlp, min_entity_len=args.min_entity_len)

    # 2.5 Normalize entity variants (space artifacts, case, suffixes, substrings)
    entity_df = normalize_entities(entity_df)

    # 2.6 Apply manual entity corrections (if review file exists)
    entity_df = apply_entity_corrections(entity_df, review_path=args.entity_review,
                                          min_count=args.min_count)

    # 3. Clustering
    entity_matrix = build_bag_of_entities(entity_df)
    assignments = pd.DataFrame()
    summaries = pd.DataFrame()
    if not entity_matrix.empty and len(entity_matrix) >= args.n_clusters:
        assignments, centroids = run_kmeans(
            entity_matrix, args.n_clusters, args.random_state
        )
        summaries = summarize_clusters(centroids, args.top_k)
    else:
        print("[WARN] Skipping clustering: empty entity matrix or fewer artists than n_clusters.")

    # 4. Save
    save_outputs(args.output_dir, artist_lyrics, entity_df, entity_matrix, assignments, summaries)

    # 5. Summary
    print("\n=== Pipeline finished ===")
    print(f"Rows processed : {len(df)}")
    print(f"Artists         : {len(artist_lyrics)}")
    print(f"Entities found  : {len(entity_df)}")
    if not entity_df.empty:
        top = entity_summary(entity_df, top_n=10)
        print("\nTop 10 entities:")
        for _, r in top.iterrows():
            print(f"  {r['entity']} ({r['label']}): {r['count']}")


if __name__ == "__main__":
    main()
