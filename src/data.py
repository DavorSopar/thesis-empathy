"""Data loading utilities for the WASSA 2024 CONV-Turn (Track 2) dataset.

The raw CSVs use **backslash-escaped double quotes** inside the ``text`` field
(e.g. ``a \\"celebrity\\" like that``). Pandas' default C parser treats the
double-double-quote (``""``) convention as the only escape, so it raises a
tokenising error on those rows unless ``escapechar='\\'`` is passed. Loading
the file through this helper keeps that detail in one place.

Splits
------
- ``train`` — 7788 turns, 13 columns (all four targets labeled).
- ``dev``   — 990 turns, 13 columns. Empathy fully labeled; SelfDisclosure
              column present but entirely null (as released by WASSA).
- ``test``  — 2316 turns, 11 columns (no ``id`` or ``SelfDisclosure``).
              Empathy is null on 255 rows (rows without gold labels); the
              loader drops these by default so downstream evaluation is on
              2061 fully-labeled turns.

The four regression targets (train and dev only; test has three, no
SelfDisclosure) are:

    Emotion            emotion intensity           (0-5, averaged annotators)
    EmotionalPolarity  valence: negative->positive (0-~2.67, averaged annotators)
    Empathy            expressed empathy           (0-5, averaged annotators)
    SelfDisclosure     personal self-disclosure    (1-4, averaged annotators)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo root = parent of the directory that holds this file (src/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"

TARGETS = ["Emotion", "EmotionalPolarity", "Empathy", "SelfDisclosure"]

_SPLIT_FILES = {
    "train": "trac2_CONVT_train.csv",
    "dev":   "trac2_CONVT_dev.csv",
    "test":  "goldstandard_CONVT.csv",
}


def load_convt(
    split: str = "train",
    path: str | Path | None = None,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """Load one split of the CONV-Turn dataset.

    Parameters
    ----------
    split : {"train", "dev", "test"}
        Which split to load. Ignored if ``path`` is provided.
    path : str | Path | None
        Explicit path to a CSV. If given, ``split`` is ignored. Defaults to
        ``data/raw/<split-file>`` relative to the repository root.
    drop_unlabeled : bool
        If True (default), rows with a null Empathy value are dropped after
        loading. This mainly affects ``test`` (255 rows without gold labels).
        Set False if you need to preserve the original row count for e.g.
        submitting predictions in the WASSA leaderboard format.

    Returns
    -------
    pandas.DataFrame
    """
    if path is None:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}. Expected one of "
                f"{sorted(_SPLIT_FILES)}."
            )
        path = DATA_DIR / _SPLIT_FILES[split]
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "The WASSA data is research-use-only and is not committed to this "
            "repository. See data/README.md for how to obtain it and where to "
            "place it."
        )

    df = pd.read_csv(path, escapechar="\\")

    if drop_unlabeled and "Empathy" in df.columns:
        df = df.dropna(subset=["Empathy"]).reset_index(drop=True)

    return df


def impute_selfdisclosure(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing SelfDisclosure with the speaker's own mean.

    SelfDisclosure is a property of the person speaking, so a missing value is
    filled with that person's average SelfDisclosure across their other turns.
    If a person has no observed value anywhere, the global mean is used as a
    fallback. Returns a copy; the input is not modified.

    Note the speaker labels in this dataset are the strings ``"Person 1"`` and
    ``"Person 2"`` (with a space) — matching them without the space silently
    fails and sends every row to the global-mean fallback.

    This function is retained from the original all-four-targets pipeline and
    is not called in the Empathy-only workflow.
    """
    df = df.copy()

    def speaker_person_id(row: pd.Series):
        if row["speaker"] == "Person 1":
            return row["person_id_1"]
        if row["speaker"] == "Person 2":
            return row["person_id_2"]
        return None

    person_id = df.apply(speaker_person_id, axis=1)
    observed = df["SelfDisclosure"].notna()
    person_mean = df.loc[observed, "SelfDisclosure"].groupby(person_id[observed]).mean()
    global_mean = df["SelfDisclosure"].mean()
    fill = person_id.map(person_mean).fillna(global_mean)
    df["SelfDisclosure"] = df["SelfDisclosure"].fillna(fill)
    return df
