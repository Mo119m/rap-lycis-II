"""Named Entity Recognition module for Chinese rap lyrics.

Handles:
- Building the spaCy NLP pipeline with EntityRuler
- Extracting entities from artist-grouped lyrics
- Entity filtering (stop labels, minimum length)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import spacy
from spacy.language import Language


DEFAULT_STOP_LABELS: Set[str] = {
    "CARDINAL", "DATE", "TIME", "ORDINAL",
    "QUANTITY", "MONEY", "PERCENT",
}


def build_nlp(lexicon_path: str = "configs/rap_lexicon_seed.jsonl") -> Language:
    """Build a spaCy NLP pipeline with optional EntityRuler lexicon.

    Tries to load zh_core_web_lg first; falls back to blank Chinese tokenizer.
    Only keeps the NER-related components to save memory.
    """
    try:
        # Only load NER-relevant components; skip parser/tagger to save memory
        nlp = spacy.load("zh_core_web_lg", exclude=["tagger", "parser", "lemmatizer",
                                                      "attribute_ruler"])
        print("[INFO] Loaded spaCy model: zh_core_web_lg (NER-only)")
    except Exception:
        print("[WARN] zh_core_web_lg not found; using blank Chinese pipeline + lexicon rules.")
        nlp = spacy.blank("zh")

    # Add EntityRuler
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


_CHUNK_SIZE = 5000  # characters per chunk fed to spaCy


def _split_text(text: str, chunk_size: int = _CHUNK_SIZE) -> List[str]:
    """Split text into chunks at newline boundaries to avoid cutting mid-sentence."""
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in text.split("\n"):
        line_len = len(line) + 1  # +1 for the newline
        if current_len + line_len > chunk_size and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def extract_entities(
    artist_lyrics: pd.DataFrame,
    nlp: Language,
    min_entity_len: int = 2,
    stop_labels: Iterable[str] = DEFAULT_STOP_LABELS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run NER on artist-grouped lyrics and return a long-form entity DataFrame.

    Text is processed in small chunks (~5000 chars) to avoid OOM crashes
    when using large spaCy models like zh_core_web_lg.

    Parameters
    ----------
    artist_lyrics : DataFrame with columns ['artist', 'combined_text']
    nlp : spaCy Language pipeline
    min_entity_len : skip entities shorter than this
    stop_labels : entity labels to filter out
    verbose : print progress

    Returns
    -------
    DataFrame with columns ['artist', 'entity', 'label']
    """
    stop_labels = set(stop_labels)
    rows: List[dict] = []
    total = len(artist_lyrics)

    for idx, (_, row) in enumerate(artist_lyrics.iterrows()):
        artist = row["artist"]
        chunks = _split_text(row["combined_text"])
        for chunk in chunks:
            doc = nlp(chunk)
            for ent in doc.ents:
                text = ent.text.strip()
                if len(text) < min_entity_len:
                    continue
                if ent.label_ in stop_labels:
                    continue
                rows.append({"artist": artist, "entity": text, "label": ent.label_})
        if verbose and (idx + 1) % 20 == 0:
            print(f"  [NER] Processed {idx + 1}/{total} artists...")

    entity_df = pd.DataFrame(rows)
    if verbose:
        print(f"[NER] Extracted {len(entity_df)} entity mentions from {total} artists")
        if not entity_df.empty:
            label_counts = entity_df["label"].value_counts()
            print(f"[NER] Label distribution:\n{label_counts.to_string()}")
    return entity_df


def entity_summary(entity_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return top-N most frequent entities across all artists."""
    if entity_df.empty:
        return pd.DataFrame(columns=["entity", "label", "count"])
    counts = (
        entity_df.groupby(["entity", "label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return counts
