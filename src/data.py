"""Data loading utilities for the WASSA CONV-turn (TRAC Track 2) dataset.

The raw CSV uses **backslash-escaped double quotes** inside the ``text`` field
(e.g. ``a \\"celebrity\\" like that``). Pandas' default C parser treats the
double-double-quote (``""``) convention as the only escape, so it raises a
tokenising error on those rows unless ``escapechar='\\'`` is passed. Loading
the file through this helper keeps that detail in one place.

The four regression targets are:

    Emotion            emotion intensity          (0-5, averaged annotators)
    EmotionalPolarity  valence: negative->positive (0-~2.67, averaged annotators)
    Empathy            expressed empathy          (0-5, averaged annotators)
    SelfDisclosure     personal self-disclosure   (1-4, averaged annotators)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo root = parent of the directory that holds this file (src/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

TARGETS = ["Emotion", "EmotionalPolarity", "Empathy", "SelfDisclosure"]


def load_convt(path: str | Path | None = None) -> pd.DataFrame:
    """Load the CONV-turn CSV with the correct quote-escaping.

    Parameters
    ----------
    path : str | Path | None
        Path to the CSV. Defaults to ``data/trac2_CONVT_train.csv`` relative
        to the repository root.

    Returns
    -------
    pandas.DataFrame
    """
    if path is None:
        path = DATA_DIR / "trac2_CONVT_train.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "The WASSA data is research-use-only and is not committed to this "
            "repository. See data/README.md for how to obtain it and where to "
            "place it."
        )
    return pd.read_csv(path, escapechar="\\")


def impute_selfdisclosure(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing SelfDisclosure with the speaker's own mean.

    SelfDisclosure is a property of the person speaking, so a missing value is
    filled with that person's average SelfDisclosure across their other turns.
    If a person has no observed value anywhere, the global mean is used as a
    fallback. Returns a copy; the input is not modified.

    Note the speaker labels in this dataset are the strings ``"Person 1"`` and
    ``"Person 2"`` (with a space) -- matching them without the space silently
    fails and sends every row to the global-mean fallback.
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
