"""Named Entity Recognition module for Chinese rap lyrics.

Handles:
- Building the spaCy NLP pipeline with EntityRuler
- Extracting entities from artist-grouped lyrics
- Entity filtering (stop labels, minimum length)
"""

from __future__ import annotations

import gc
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
            del doc
        # Periodically free accumulated memory
        if (idx + 1) % 20 == 0:
            gc.collect()
            if verbose:
                print(f"  [NER] Processed {idx + 1}/{total} artists...")

    entity_df = pd.DataFrame(rows)
    if verbose:
        print(f"[NER] Extracted {len(entity_df)} entity mentions from {total} artists")
        if not entity_df.empty:
            label_counts = entity_df["label"].value_counts()
            print(f"[NER] Label distribution:\n{label_counts.to_string()}")
    return entity_df


def generate_entity_review(
    entity_df: pd.DataFrame,
    output_path: str = "configs/entity_review.csv",
    top_n: int = 100,
) -> Path:
    """Generate a CSV for manual entity review.

    Outputs one row per unique (entity, label) pair, sorted by frequency,
    with an empty ``action`` column for the user to fill in:
    - leave blank → keep as-is
    - ``delete``  → remove this entity from the dataset
    - a label name (e.g. ``CITY``) → relabel to that type

    Parameters
    ----------
    entity_df : long-form entity DataFrame
    output_path : where to write the review CSV
    top_n : max entities per label (0 = all)
    """
    if entity_df.empty:
        return Path(output_path)

    parts = []
    for label in entity_df["label"].value_counts().index:
        sub = (
            entity_df[entity_df["label"] == label]
            .groupby("entity").size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        if top_n > 0:
            sub = sub.head(top_n)
        sub["label"] = label
        parts.append(sub)

    review = pd.concat(parts, ignore_index=True)[["entity", "label", "count", ]]
    review["action"] = ""
    review = review.sort_values(["label", "count"], ascending=[True, False]).reset_index(drop=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[REVIEW] Wrote {len(review)} entities to {out}")
    print(f"[REVIEW] Fill the 'action' column: blank=keep, delete=remove, LABEL_NAME=relabel")
    return out


def apply_entity_corrections(
    entity_df: pd.DataFrame,
    review_path: str = "configs/entity_review.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply manual corrections from the review CSV to the entity DataFrame.

    Returns a new DataFrame with deletions and relabels applied.
    """
    review_file = Path(review_path)
    if not review_file.exists():
        if verbose:
            print(f"[REVIEW] No review file at {review_file}, skipping corrections")
        return entity_df

    review = pd.read_csv(review_file, encoding="utf-8-sig")
    if "action" not in review.columns:
        if verbose:
            print(f"[REVIEW] Review file has no 'action' column, skipping")
        return entity_df

    # Build lookup: (entity, label) → action
    corrections = {}
    for _, r in review.iterrows():
        raw = r.get("action", "")
        if pd.isna(raw):
            continue
        action = str(raw).strip()
        if action:
            corrections[(str(r["entity"]).strip(), str(r["label"]).strip())] = action

    if not corrections:
        if verbose:
            print(f"[REVIEW] No corrections found in {review_file}")
        return entity_df

    n_before = len(entity_df)
    delete_count = 0
    relabel_count = 0

    # Apply corrections
    mask_delete = pd.Series(False, index=entity_df.index)
    new_labels = entity_df["label"].copy()

    for (ent, label), action in corrections.items():
        match = (entity_df["entity"] == ent) & (entity_df["label"] == label)
        if action.lower() == "delete":
            mask_delete |= match
            delete_count += match.sum()
        else:
            # Treat action as a new label
            new_labels[match] = action
            relabel_count += match.sum()

    entity_df = entity_df[~mask_delete].copy()
    entity_df["label"] = new_labels[~mask_delete]

    if verbose:
        print(f"[REVIEW] Applied corrections from {review_file}")
        print(f"  Deleted:   {delete_count} mentions")
        print(f"  Relabeled: {relabel_count} mentions")
        print(f"  Remaining: {len(entity_df)}/{n_before} mentions")

    return entity_df.reset_index(drop=True)


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
