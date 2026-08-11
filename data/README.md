# Data

This project uses the **WASSA CONV-turn** dataset (Track 2 / TRAC: turn-level
empathy, emotion polarity, emotion intensity, and self-disclosure in dyadic
conversations about news articles), an extension of the dataset introduced by
Omitaomu et al. (2022).

## Why the CSV is not in the repository

The WASSA shared-task data is released **for scientific/research purposes only**;
other use is explicitly prohibited by the task's terms and conditions. Because
redistribution is restricted, the raw CSV is **git-ignored** and is not
committed here. Obtain your own copy from the official source and place it in
this folder.

## What to download and where to put it

Expected file (referenced by `src/data.py` and both notebooks):

```
data/trac2_CONVT_train.csv
```

Obtain it from the WASSA shared-task distribution (CodaLab competition pages,
linked from the WASSA workshop site: <https://wassa-workshop.github.io/>).
Registering for the shared task grants access to the train / dev / test splits.
Only the **train** split is required to reproduce the notebooks as written; if
you also download `dev` and `test`, drop them in beside the train file and point
`src.data.load_convt(path=...)` at them.

## Expected format

Comma-separated, one row per conversational turn, with columns:

| Column | Meaning |
|---|---|
| `id` | Row identifier |
| `article_id` | Article that seeded the conversation |
| `conversation_id` | Conversation identifier |
| `turn_id` | Position of the turn within its conversation |
| `speaker` | `"Person 1"` or `"Person 2"` (note the space) |
| `text` | The utterance |
| `person_id_1`, `person_id_2` | Stable IDs of the two interlocutors |
| `Emotion` | Emotion intensity, 0–5 (averaged annotators) |
| `EmotionalPolarity` | Valence, negative→positive (averaged annotators) |
| `Empathy` | Expressed empathy, 0–5 (averaged annotators) |
| `SelfDisclosure` | Self-disclosure, 1–4 (averaged annotators) |

> The `text` field contains **backslash-escaped** double quotes, so the file
> must be read with `escapechar='\\'`. `src/data.py` handles this.

## Citing

If you use this data, cite:

- Omitaomu, D., Tafreshi, S., Liu, T., Buechel, S., Callison-Burch, C.,
  Eichstaedt, J., Ungar, L., & Sedoc, J. (2022). *Empathic Conversations: A
  Multi-level Dataset of Contextualized Conversations.* arXiv:2205.12698.
- Giorgi, S., Sedoc, J., Barriere, V., & Tafreshi, S. (2024). *Findings of WASSA
  2024 Shared Task on Empathy and Personality Detection in Interactions.* In
  Proceedings of WASSA 2024, 369–379.
