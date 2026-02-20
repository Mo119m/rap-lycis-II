"""I/O utilities for saving and loading pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_outputs(
    output_dir: str | Path,
    artist_lyrics: pd.DataFrame,
    entity_df: pd.DataFrame,
    entity_matrix: pd.DataFrame,
    assignments: pd.DataFrame,
    summaries: pd.DataFrame,
) -> Path:
    """Save all pipeline outputs to the given directory.

    Returns the resolved output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artist_lyrics.to_csv(output_dir / "artist_lyrics.csv", index=False)
    entity_df.to_csv(output_dir / "entities_long.csv", index=False)
    entity_matrix.to_csv(output_dir / "bag_of_entities.csv")

    if not assignments.empty:
        assignments.to_csv(output_dir / "artist_clusters.csv", index=False)
    if not summaries.empty:
        summaries.to_csv(output_dir / "cluster_entity_summary.csv", index=False)

    print(f"[IO] Outputs saved to {output_dir}/")
    for f in sorted(output_dir.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")

    return output_dir
