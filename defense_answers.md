Task and scope

Q: Why the WASSA 2024 CONV-Turn dataset?
Established benchmark for turn-level empathy prediction. Enables direct comparison with 14 published team submissions. Has three annotated affective targets on a single-utterance basis — appropriate for the specific question of what turn-level lexical features carry.

Q: Why only Empathy, not all four targets?
Scope decision. Empathy is central to prosocial dialogue modeling and is WASSA's primary evaluated dimension. Single-target focus enables deeper analysis (error patterns, coefficient interpretation) than a broader multi-target scan would allow in a 4-week thesis timeline. Multi-task learning across the four targets is named as future work.

Q: The dataset mixes human-human and human-AI conversations. Why not separate them?
Following convention established by all 14 WASSA 2024 teams — none distinguished bot vs. human turns. The task specification treats turns identically regardless of speaker origin, since the annotation is third-party ("how empathetic does this turn sound") not first-party. Separating them would answer a different research question about cross-domain generalization.

Data splits and integrity

Q: What splits did you use?
Official WASSA 2024 CONV-Turn splits: train (9,330 turns after filtering, from 405 conversations), dev (990 turns, 33 conversations), test (2,061 turns, 63 conversations after dropping 255 unlabeled rows).

Q: Wait — the WASSA train file has 11,166 turns. Why did you filter?
The released "train" file contains all conversations in the underlying corpus, including 33 conversations that appear in dev and 49 in test. Training on the unfiltered file would produce direct data leakage. We filter to a conversation-disjoint partition before training.

Q: Is speaker overlap a concern?
No. All three splits are conversation-disjoint. Our models see only turn text; speaker identity is not a feature. Even where speakers happen to overlap across splits, the model cannot exploit this. WASSA's official splits are additionally speaker-disjoint, which is a stronger guarantee than we strictly need.

Q: Why were your pre-rework numbers higher?
Prior experiments used an internal 70/15/15 conversation-grouped split. That split was conversation-disjoint but not speaker-disjoint, allowing the model to exploit speaker-specific writing patterns. Moving to WASSA official splits removed this leakage, revealing the model's true generalization performance to unseen speakers. Reported numbers are the honest post-filter results.

Baseline model (Ridge + TF-IDF)

Q: Why Ridge and not plain linear regression?
L2 regularization prevents overfitting in the high-dimensional sparse feature space that TF-IDF produces. Alpha=0 collapses to OLS and overfits; too-large alpha underfits. We sweep alpha on dev.

Q: Why Ridge and not Lasso?
Ridge shrinks coefficients toward zero; Lasso can drive them to exactly zero. For interpretability of top-coefficient words, Ridge's continuous shrinkage produces a more informative ranking. Sparsity isn't our objective — feature selection is handled implicitly by TF-IDF's max_features cap.

Q: How did you pick alpha = 3.0?
Development-set sweep across α ∈ {0.1, 1, 3, 10, 100}. α=3 produced the highest dev Pearson correlation, with clearly worse performance at both smaller values (overfitting) and larger values (underfitting) — the classical bias-variance U-shape.

Q: Why max_features = 5000?
Sweep across {1000, 5000, 10000, 20000} on dev. Performance is essentially flat at and above 5000 on our data because the training vocabulary contains only ~7,500 unique unigrams. Selected the smallest value at which performance saturates.

Q: Why unigrams only?
Sweep of (1,1) vs (1,2) showed essentially identical performance. Retained unigrams for baseline simplicity; adding bigrams provided no meaningful signal.

Q: Why not report R² instead of Pearson?
Pearson r is WASSA's official evaluation metric, enabling direct comparison with published shared task submissions. Report MAE, RMSE, Pearson r together since each captures different error properties: MAE for average absolute error, RMSE for outlier sensitivity, Pearson for ranking quality.

RoBERTa fine-tuning

Q: Why RoBERTa specifically?
Well-established transformer baseline for text regression tasks. Pretrained on 160GB of English text; matched in complexity to what fine-tuning on ~9k turns can meaningfully leverage. Larger models (e.g. RoBERTa-large, DeBERTa) would likely overfit on this training set size.

Q: Why fine-tuning rather than a specialized architecture?
Following standard practice for BERT-family models. Task-specific architectures like the CombinedLoss + FGM adversarial training of Yang et al. (2024) yield marginal improvements over standard fine-tuning; our simpler approach achieves competitive performance (test Pearson 0.558 vs. their 0.544 on Empathy).

Q: How did you pick hyperparameters?
Standard defaults for BERT-family fine-tuning: learning rate 2e-5, batch size 16, 2-3 epochs, AdamW optimizer. We validated by comparing lr=2e-5 vs. lr=5e-5 on dev; the smaller value performed better and was retained.

Q: Why 2 (or 3) epochs and not more?
Development-loss trajectory showed early saturation: validation loss ceased improving after epoch 2 while training loss continued dropping, indicating overfitting. Selecting the epoch with highest dev Pearson via load_best_model_at_end prevents training beyond the optimum.

Q: Did you freeze any layers?
No. All 125M parameters were updated during fine-tuning, using a small learning rate (2e-5) to preserve pretrained representations while adapting to the task.

Evaluation and results

Q: Your Pearson r is around 0.55 on test. Is that good?
Competitive with the 4th-place WASSA 2024 team submission on the same test set (Yang et al. 2024, Empathy test Pearson 0.544). Our RoBERTa result of 0.558 slightly exceeds it, achieved with standard fine-tuning without their specialized CombinedLoss or FGM adversarial training. The ceiling on this task appears to be near 0.6-0.65 on test, likely reflecting the inherent gap between turn-level input and context-informed annotations.

Q: Why is dev Pearson higher than test?
Common property of the WASSA CONV-Turn task, observed in other published submissions (e.g. Yang et al. achieved dev Empathy 0.633 vs. test 0.544, a gap of 0.09). Our dev-test gap of 0.064 is smaller. The gap likely reflects distributional differences between the specific conversations sampled into dev vs. test.

Q: RoBERTa only beats Ridge by 0.01 on test. Is the added complexity justified?
Both models are near the task's apparent ceiling. RoBERTa's larger improvement on dev (0.076) partially transfers to test. More importantly, the models fail on different types of examples (see error analysis), meaning they could plausibly complement each other. The comparison itself is informative: bag-of-words features capture most of the extractable signal from single turns, with contextualized representations adding a modest additional gain.

Error analysis

Q: Where does your model fail most?
Two systematic patterns: (1) high-empathy turns using minimal vocabulary ("me too", "yeah, 30 years, imagine that") are under-predicted because they lack emotional word content; (2) low-empathy turns using emotional vocabulary ("that would be traumatizing", "so distressing") are over-predicted because Ridge coefficients weight negative-affect words heavily. Both patterns show the model relies on lexical cues rather than pragmatic function.

Q: Isn't this just a limitation of your model?
Partly, but more fundamentally a task-inherent property. Annotators had full conversation context when rating; models predict from single turns without it. The gap between what turn text encodes and what context-informed annotations reflect is unrecoverable from turn-only input. Named as future work: incorporating dialogue history via concatenated context or hierarchical models.

Q: Are the annotators reliable?
Three third-party crowdworkers per turn, ratings averaged. Some turns show clear inter-annotator disagreement. However, our concern is not annotation quality per se but the information asymmetry between annotators (with context) and models (without context). Both surface readers and turn-level models systematically misalign with context-informed empathy judgments in the same ways.

Methodology and choices we deliberately didn't make

Q: Why not train a joint multi-target model?
Named as future work. Multi-task learning across all four targets could exploit cross-target correlations (e.g. Empathy correlates with Emotion). Beyond thesis timeline scope.

Q: Why not use a larger model like RoBERTa-large or DeBERTa?
Fine-tuning larger models on this training set size (9,330 turns) risks overfitting. Yang et al. (2024) used DeBERTa with additional regularization (CombinedLoss + FGM) and achieved test Empathy 0.544 — slightly below our RoBERTa-base result of 0.558.

Q: Why not add conversation context?
Named as future work in Discussion. The CONV-Turn task specifically requires turn-level prediction; adding context would change the task and invalidate direct comparison with published WASSA benchmarks. A follow-up study could concatenate previous turns or use hierarchical models.

Q: Why not train an LLM (GPT-4, Llama) for comparison?
Scope decision. Zero-shot LLM performance on annotation-style regression tasks is often inconsistent and requires substantial prompt engineering. Comparison between a strong linear baseline and a fine-tuned transformer covers the main question about representation types (bag-of-words vs. contextualized) for this task.

Reproducibility

Q: Can your work be reproduced?
Yes. Code and all decisions are documented in the GitHub repository (link). The WASSA 2024 CONV-Turn dataset is publicly available. Trained Ridge model weights are checked in; RoBERTa fine-tuned weights are ~500MB and are excluded from the repo but regenerable via the notebook (30 min on Colab T4 GPU).

Q: What are the main sources of variance in your results?
For Ridge, results are deterministic given the data. For RoBERTa, random initialization of the regression head introduces run-to-run variance; we report a single run. Ideally, results should be averaged across 3-5 seeds — named as a methodological limitation.
