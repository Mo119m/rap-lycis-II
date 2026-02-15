"""Data loading and cleaning utilities for Chinese rap lyrics.

Handles:
- Loading raw CSV data
- Filtering out non-lyric songs (Live, instrumental, etc.)
- Stripping production credit lines from lyric text
- Grouping cleaned lyrics by artist
"""

from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Song-level filters (applied to song_title)
# ---------------------------------------------------------------------------

# Regex patterns that mark a song as non-original / non-studio
_TITLE_EXCLUDE_PATTERNS: List[str] = [
    r"[\(（]Live[\)）]",
    r"[\(（]live[\)）]",
    r"[\(（]伴奏[\)）]",
    r"伴奏$",
    r"[\(（]Instrumental[\)）]",
    r"[\(（]instrumental[\)）]",
]

_TITLE_EXCLUDE_RE = re.compile("|".join(_TITLE_EXCLUDE_PATTERNS), re.IGNORECASE)


def _is_excluded_title(title: str) -> bool:
    """Return True if the song title indicates a Live / instrumental version."""
    if pd.isna(title):
        return False
    return bool(_TITLE_EXCLUDE_RE.search(str(title)))


# ---------------------------------------------------------------------------
# Line-level filters (applied to individual lines within text)
# ---------------------------------------------------------------------------

# Lines that are purely production credits / metadata
_CREDIT_LINE_PATTERNS: List[str] = [
    r"^出品",                        # 出品公司 AFSC MUSIC ...
    r"^Prod[\.\s]",                  # Prod. by ...
    r"^prod[\.\s]",                  # prod. by ...
    r"^Beat\s*by",                   # Beat by ...
    r"^Mixed\s*by",                  # Mixed by ...
    r"^Mastered\s*by",               # Mastered by ...
    r"^录音",                        # 录音: / 录音室: ...
    r"^混音",                        # 混音: ...
    r"^母带",                        # 母带: ...
    r"^编曲",                        # 编曲: ...
    r"^作词",                        # 作词: ...
    r"^作曲",                        # 作曲: ...
    r"^演唱",                        # 演唱: ...
    r"^制作人",                      # 制作人: ...
    r"^Recording",                   # Recording Engineer: ...
    r"^Mixing",                      # Mixing Engineer: ...
    r"^Mastering",                   # Mastering Engineer: ...
    r"^OP\s*[:：]",                  # OP: ...
    r"^SP\s*[:：]",                  # SP: ...
]

_CREDIT_LINE_RE = re.compile("|".join(_CREDIT_LINE_PATTERNS), re.IGNORECASE)

# Copyright / legal disclaimers that got scraped with lyrics
_COPYRIGHT_RE = re.compile(r"未经.*书面许可|未经.*权利人", re.UNICODE)

# Structure markers that are not actual lyrics (standalone labels only)
_STRUCTURE_LINE_RE = re.compile(
    r"^\s*[\[\(（【]*\s*"
    r"(Verse|Hook|Chorus|Bridge|Intro|Outro|Pre-?Chorus|Refrain|Interlude)"
    r"[\s\d]*[\]\)）】]*\s*[:：]?\s*$",
    re.IGNORECASE,
)


# Speaker labels in collaborative tracks: "小老虎：别信这一套" → "别信这一套"
# Matches patterns like "艺人名：" or "ArtistName:" at the start of a line.
# We strip the label but keep the lyric content after it.
_SPEAKER_LABEL_RE = re.compile(
    r"^[\w\s\.\-]{1,20}[：:]\s*",
    re.UNICODE,
)


def _strip_speaker_label(line: str) -> str:
    """Remove speaker labels like '小老虎：' from the start of a line."""
    return _SPEAKER_LABEL_RE.sub("", line)


def _clean_text(text: str) -> str:
    """Remove production credit lines, structure markers, and speaker labels.

    Keeps actual lyric lines, including lines that *mention* Verse/Hook inline
    (e.g., "我的verse比你强" is kept).
    """
    if pd.isna(text):
        return ""
    lines = str(text).split("\n")
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _CREDIT_LINE_RE.match(stripped):
            continue
        if _STRUCTURE_LINE_RE.match(stripped):
            continue
        if _COPYRIGHT_RE.search(stripped):
            continue
        # Strip speaker labels (e.g., "小老虎：" at line start)
        stripped = _strip_speaker_label(stripped)
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(path: str, max_rows: int = 0) -> pd.DataFrame:
    """Load the raw lyrics CSV and validate required columns."""
    df = pd.read_csv(path)
    required_cols = {"artist", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()
    df = df.dropna(subset=["artist", "text"])
    return df


def clean_songs(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove non-lyric songs (Live, instrumental, etc.) from the dataset.

    Returns a filtered copy of the DataFrame.
    """
    n_before = len(df)
    songs_before = df["song_id"].nunique() if "song_id" in df.columns else None

    # Filter by song title
    if "song_title" in df.columns:
        mask = df["song_title"].apply(_is_excluded_title)
        df = df[~mask].copy()

    n_after = len(df)
    if verbose:
        removed = n_before - n_after
        print(f"[CLEAN] Removed {removed} rows from excluded song types "
              f"(Live/伴奏/instrumental)")
        if songs_before is not None:
            songs_after = df["song_id"].nunique()
            print(f"[CLEAN] Songs: {songs_before} → {songs_after} "
                  f"(removed {songs_before - songs_after})")
    return df


def clean_text(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Strip production credits and structure markers from lyric text.

    Modifies the 'text' column in-place on a copy.
    """
    df = df.copy()
    df["text"] = df["text"].apply(_clean_text)

    # Drop rows where text became empty after cleaning
    empty_mask = df["text"].str.strip() == ""
    if verbose and empty_mask.any():
        print(f"[CLEAN] Dropped {empty_mask.sum()} rows that became empty after text cleaning")
    df = df[~empty_mask]
    return df


def deduplicate(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove duplicate text chunks within the same artist."""
    n_before = len(df)
    df = df.drop_duplicates(subset=["artist", "text"]).copy()
    n_after = len(df)
    if verbose and n_before != n_after:
        print(f"[CLEAN] Deduplicated: {n_before} → {n_after} rows "
              f"(removed {n_before - n_after} same-artist duplicate chunks)")
    return df


def combine_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """Group all lyric chunks by artist into a single combined text."""
    artist_lyrics = (
        df.groupby("artist", as_index=False)["text"]
        .apply(lambda x: "\n".join(x.astype(str)))
    )
    artist_lyrics = artist_lyrics.rename(columns={"text": "combined_text"})
    return artist_lyrics


def load_and_clean(
    path: str,
    max_rows: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convenience function: load → clean songs → clean text → return."""
    df = load_data(path, max_rows=max_rows)
    if verbose:
        print(f"[LOAD] Loaded {len(df)} rows, {df['artist'].nunique()} artists")
    df = clean_songs(df, verbose=verbose)
    df = clean_text(df, verbose=verbose)
    df = deduplicate(df, verbose=verbose)
    if verbose:
        print(f"[CLEAN] Final: {len(df)} rows, {df['artist'].nunique()} artists, "
              f"{df['song_id'].nunique() if 'song_id' in df.columns else '?'} songs")
    return df
