"""Named Entity Recognition module for Chinese rap lyrics.

Handles:
- Building the spaCy NLP pipeline with EntityRuler
- Extracting entities from artist-grouped lyrics
- Entity filtering (stop labels, minimum length)
"""

from __future__ import annotations

import gc
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd
import spacy
from spacy.language import Language


DEFAULT_STOP_LABELS: Set[str] = {
    "CARDINAL", "DATE", "TIME", "ORDINAL",
    "QUANTITY", "MONEY", "PERCENT",
}


def build_nlp(
    lexicon_path: str = "configs/rap_lexicon_seed.jsonl",
    model_name: str = "zh_core_web_trf",
) -> Language:
    """Build a spaCy NLP pipeline with optional EntityRuler lexicon.

    Parameters
    ----------
    lexicon_path : path to EntityRuler patterns JSONL
    model_name : spaCy model to load. Supported:
        - ``zh_core_web_trf``  (transformer, most accurate, needs torch)
        - ``zh_core_web_lg``   (CNN, faster, no torch needed)
        Falls back to blank Chinese tokenizer if the model is not installed.
    """
    # Components to exclude vary by model architecture
    _EXCLUDE_STAT = ["tagger", "parser", "lemmatizer", "attribute_ruler"]
    _EXCLUDE_TRF = ["tagger", "parser", "lemmatizer", "attribute_ruler"]

    exclude = _EXCLUDE_TRF if "trf" in model_name else _EXCLUDE_STAT
    try:
        nlp = spacy.load(model_name, exclude=exclude)
        print(f"[INFO] Loaded spaCy model: {model_name}")
    except Exception:
        print(f"[WARN] {model_name} not found; using blank Chinese pipeline + lexicon rules.")
        print(f"[HINT] Install with: python -m spacy download {model_name}")
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


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_CHINESE_SUFFIXES = re.compile(r"(们|的|了|着|过|啊|呢|吧|呀|哦|嘛|啦)$")


def _is_cjk(text: str) -> bool:
    """True if the text is primarily CJK characters."""
    cjk_count = len(_CJK_RE.findall(text))
    return cjk_count > len(text) * 0.5


def _normalize_key(text: str) -> str:
    """Produce a matching key: lowercase, strip CJK-internal spaces."""
    # Remove spaces between CJK characters (tokenization artifacts like "长 沙")
    result = re.sub(
        r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])",
        r"\1\2",
        text,
    )
    # Apply repeatedly for chains like "说 唱 歌 手"
    while re.search(r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])", result):
        result = re.sub(
            r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])",
            r"\1\2",
            result,
        )
    return result.lower().strip()


def normalize_entities(
    entity_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Normalize entity text to merge trivial variants.

    Three-pass normalization:

    1. **Space & case**: collapse CJK-internal spaces ("长 沙" → "长沙"),
       case-fold English ("God"/"god" → "god"), merge English space-artifacts
       when the no-space form exists ("re al" → "real").
    2. **Chinese suffix stripping**: "老子们" → "老子" if the base form already
       exists in the dataset with the same label.
    3. **Substring containment**: "奥运会" → "奥运" if the shorter form exists
       with the same label and is more frequent.
    """
    if entity_df.empty:
        return entity_df

    # -- Build frequency table per (entity, label) --
    freq: Counter = Counter()
    for ent, label in zip(entity_df["entity"], entity_df["label"]):
        freq[(ent, label)] += 1

    # -- Pass 1: space + case normalization --
    # Group entities that share the same normalized key + label
    key_to_entities: Dict[tuple, List[str]] = {}
    for (ent, label) in freq:
        key = (_normalize_key(ent), label)
        key_to_entities.setdefault(key, []).append(ent)

    mapping: Dict[tuple, str] = {}  # (old_entity, label) → canonical_entity
    for (norm_key, label), forms in key_to_entities.items():
        if len(forms) <= 1:
            continue
        # Pick the most frequent form as canonical
        canonical = max(forms, key=lambda f: freq[(f, label)])
        for form in forms:
            if form != canonical:
                mapping[(form, label)] = canonical

    # Also handle pure English space artifacts: "re al" → "real"
    # by checking if removing ALL spaces produces an existing entity
    for (ent, label), count in list(freq.items()):
        if " " in ent and not _is_cjk(ent):
            no_space = ent.replace(" ", "")
            # Check if the no-space version (case-insensitive) exists
            for (other_ent, other_label) in freq:
                if other_label == label and other_ent.lower() == no_space.lower() and other_ent != ent:
                    if (ent, label) not in mapping:
                        mapping[(ent, label)] = other_ent
                    break

    # -- Pass 2: Chinese suffix stripping --
    # After pass 1, rebuild frequency with merged forms
    freq2: Counter = Counter()
    for (ent, label), count in freq.items():
        canonical = mapping.get((ent, label), ent)
        freq2[(canonical, label)] += count

    existing = set(freq2.keys())
    for (ent, label) in list(existing):
        if _is_cjk(ent) and len(ent) >= 3:
            stripped = _CHINESE_SUFFIXES.sub("", ent)
            if stripped != ent and (stripped, label) in existing:
                if (ent, label) not in mapping:
                    mapping[(ent, label)] = stripped

    # -- Pass 3: substring containment (same label) --
    # Rebuild frequency again with suffix merges applied
    freq3: Counter = Counter()
    for (ent, label), count in freq.items():
        canonical = mapping.get((ent, label), ent)
        freq3[(canonical, label)] += count

    by_label: Dict[str, List[tuple]] = {}
    for (ent, label), count in freq3.items():
        by_label.setdefault(label, []).append((ent, count))

    for label, entries in by_label.items():
        # Sort by frequency descending, then length ascending
        entries.sort(key=lambda x: (-x[1], len(x[0])))
        # For each entity, check if a more frequent entity is a substring
        canonicals = []  # (entity, count) pairs that are already canonical
        for ent, count in entries:
            merged = False
            for canon, canon_count in canonicals:
                # Only merge if one strictly contains the other
                if len(ent) != len(canon) and (canon in ent or ent in canon):
                    shorter = canon if len(canon) < len(ent) else ent
                    longer = ent if shorter == canon else canon
                    # Merge longer → shorter (keep the more concise form)
                    if shorter == canon:
                        # ent is longer, merge ent → canon
                        mapping[(ent, label)] = canon
                    else:
                        # canon is longer, merge canon → ent
                        mapping[(canon, label)] = ent
                    merged = True
                    break
            if not merged:
                canonicals.append((ent, count))

    # -- Apply mapping --
    if not mapping:
        if verbose:
            print("[NORM] No entities to normalize")
        return entity_df

    new_entities = entity_df["entity"].copy()
    labels = entity_df["label"]
    merge_count = 0
    for i in range(len(entity_df)):
        key = (new_entities.iloc[i], labels.iloc[i])
        if key in mapping:
            new_entities.iloc[i] = mapping[key]
            merge_count += 1

    result = entity_df.copy()
    result["entity"] = new_entities

    if verbose:
        print(f"[NORM] Normalized {merge_count} mentions across {len(mapping)} entity variants")
        # Show the merges
        shown = 0
        for (old, label), new in sorted(mapping.items(), key=lambda x: -freq.get(x[0], 0)):
            print(f"  \"{old}\" → \"{new}\" [{label}]")
            shown += 1
            if shown >= 20:
                remaining = len(mapping) - shown
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break

    return result


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
