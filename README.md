# School Nonattendance in Nigeria: MICS 2021

This project classifies whether a child aged 5-17 reported attending school during the current school year, using the Nigeria MICS 2021 household survey. It compares ten classical and neural experiments (logistic regression, random forests, gradient boosting and Keras neural networks) on identical community-level data splits.

The task is to classify reported attendance. It does not predict future dropout or establish the causes of nonattendance.

**Mission:** Promote inclusive education by understanding patterns associated with nonattendance and informing where further investigation and educational support may be needed.

## Project links

- GitHub repository: **Should be found on project report**
- Presentation video: **Should be found on project report**
- Dataset file used: **Should be found on project report**

## Results

The best validation accuracy was **82.82%** (E06, neural network), compared with a 73.39% majority-class baseline. The exploratory 87% target was not met. E10 (gradient boosting) led on average precision (AP, 0.7163, narrowly ahead of E09 at 0.7162) and ROC-AUC (0.8594).

Accuracy hides a weakness: at threshold 0.50, E06 misses 495 of the 1,183 nonattenders in the validation set.

All scores below are validation results at threshold 0.50. Final test evaluation was skipped, so none of them is test accuracy.

![Validation comparison of all ten experiments](figures/validation_metric_comparison.png)

| ID | Model | Features | Settings | Accuracy | Recall | AP | ROC-AUC |
|---|---|---|---|---|---|---|---|
| E01 | Logistic regression | Baseline | C=1 | 0.8178 | 0.5435 | 0.6932 | 0.8363 |
| E02 | Logistic regression | Baseline | C in {0.1, 1, 10}, grouped-CV AP search | 0.8178 | 0.5435 | 0.6932 | 0.8363 |
| E03 | Random forest | Baseline | 150 trees, leaf size 1 | 0.7998 | 0.4945 | 0.6369 | 0.8101 |
| E04 | Random forest | Enhanced | 150 trees, leaf {2, 5}, depth {12, None}, grouped-CV AP search | 0.8207 | 0.5097 | 0.6971 | 0.8496 |
| E05 | Keras MLP | Baseline | 64/32 units, Adam 0.001 | 0.8273 | **0.5850** | 0.7102 | 0.8558 |
| E06 | Keras MLP | Enhanced | 64/32 units, Adam 0.001 | **0.8282** | 0.5816 | 0.7127 | 0.8553 |
| E07 | Keras MLP | Enhanced | Dropout 0.25, L2 0.0001, Adam 0.0003 | 0.8275 | 0.5427 | 0.7124 | 0.8554 |
| E08 | Random forest | Enhanced | 200 trees, depth {10, 20, None}, leaf {2, 8}, max features {sqrt, 0.8}, grouped-CV accuracy search | 0.8189 | 0.5114 | 0.6973 | 0.8496 |
| E09 | Histogram boosting | Baseline | 200 iterations, rate {0.05, 0.1}, leaves {15, 31}, L2 {1, 10}, grouped-CV accuracy search | 0.8257 | 0.5461 | 0.7162 | 0.8593 |
| E10 | Histogram boosting | Enhanced | Same search as E09 | 0.8252 | 0.5410 | **0.7163** | **0.8594** |

E02 reproduced E01's scores exactly. The neural networks use a sigmoid output, binary cross-entropy, at most 40 epochs and early stopping on validation loss (patience 5, best weights restored). E05 and E06 ran 10 epochs and restored epoch 5. E07 ran 38 and restored epoch 33.

### Model selection

The final models are the highest validation AP in each family across all ten runs: **E10** (classical) and **E06** (neural). AP suits the minority outcome better than accuracy. A separate search on validation F1 set decision thresholds of **0.35** for E10 and **0.38** for E06. Scores at 0.50 must not be reported as scores at these thresholds.

### Where the models struggle

At its tuned threshold, E10 recalls about 74% of rural nonattenders but only 42% of urban ones. It also recalls about 81% of children whose label was reconstructed from never-attended answers, against 33% for directly reported nonattendance. Permutation importance ranks mother's education, state and wealth highest. These are validation descriptions, not causal findings or a fairness assessment.

## Run in Google Colab

1. Open the notebook through the badge above, or upload it via **File > Upload notebook**. Use a Python **CPU runtime**.
2. Run **Install dependencies**, restart the session once, then continue from **Load the input from Google Drive**. Do not rerun the install cell after training.
3. In the input cell, `DRIVE_LINK` points to the project dataset. To use your own copy, set it to the sharing link of a ZIP or `fs.sav` **file** (not a folder) and sign in when prompted. No API key is needed.
4. Run the remaining cells in order. Before training, check the source hash, raw shape and split counts printed in sections 3 and 7.
5. Run E01 to E10. Each experiment is its own function with a fresh model and run directory. Neural training stops early and need not reach 40 epochs.
6. Keep `RUN_FINAL_TEST = False` to reproduce the reported results.
7. Colab's `/content` folder is temporary. To keep the trained models and logs, set `DOWNLOAD_EVIDENCE = True` in section 20 before the runtime ends.
8. Run the TensorBoard cell at the end to compare all runs.

Use a fresh session for a complete rerun. If preprocessing or splits change, rerun every experiment.

The notebook has 21 sections: data audit and cohort (3-6), splits and preprocessing (7-10), experiment setup (11-12), experiments E01-E10 (13-15), comparison and selection (16-17), diagnostics (18), optional test evaluation (19) and packaging (20-21).

## Data and target

**Source:** [World Bank Microdata Library, Nigeria MICS 2021](https://microdata.worldbank.org/catalog/5959), produced by UNICEF and conducted by Nigeria's National Bureau of Statistics. Obtain the data through the official access process and follow its terms. This repository does not grant redistribution rights, so do not commit raw survey files.

The notebook reads only `fs.sav`, directly or from the nested survey ZIP, so no table joins are needed. The raw file has 22,706 rows and 427 columns. After excluding 263 incomplete interviews and 11 unresolved targets, **22,432 records** remain: 16,428 attenders and 6,004 nonattenders.

**Target:** `0` = attended, `1` = did not attend school or an early childhood programme during the current school year. Direct `CB7` answers define the label. For 4,435 records, `CB4 = 2` (never attended) with a system-missing `CB7` is treated as nonattendance. A missing `CB7` alone is not enough.

The expected SHA-256 of the uncompressed `fs.sav` bytes (not the ZIP) is:

```text
6e5283706128255d4393db19ef397d41a9116d20ad94db1bc42351ca7f566ae4
```

**Predictors:** `CB3` (age), `HH52` (household children aged 5-17), `HL4` (sex), `HH6` (area), `HH7` (state), `MELEVEL` (mother's education) and `WINDEX5` (wealth quintile). The enhanced representation adds age bands and `log1p(HH52)`. Attendance answers, identifiers and survey weights are never model inputs.

## Splits and preprocessing

All experiments use seed 42 and the same partitions. `StratifiedGroupKFold` groups by `PSU` (survey cluster), so no community appears in more than one partition. A five-fold split reserves the test set, then a four-fold split of the remainder reserves validation. The result is roughly 60/20/20.

| Split | Rows | PSUs | Nonattenders |
|---|---|---|---|
| Training | 13,444 | 1,046 | 3,536 |
| Validation | 4,446 | 351 | 1,183 |
| Test | 4,542 | 358 | 1,285 |

Imputation, missingness indicators, scaling and one-hot encoding are fitted on training data only, giving 54 baseline and 58 enhanced features. Classical pipelines refit preprocessing inside three grouped CV folds. Survey weights are used for descriptive checks only. All model metrics are unweighted.

## TensorBoard and saved outputs

TensorBoard runs locally with no account, API key or upload:

```python
%load_ext tensorboard
%tensorboard --logdir {TENSORBOARD_ROOT}
```

Select runs `E01` to `E10`. **Scalars** shows validation metrics, classical accuracy against training size and neural metrics against epoch (the horizontal axes differ). **Images** shows confusion matrices and curves. **HParams** compares settings. The saved figures also stay visible in the executed notebook. If no curves appear, confirm that training finished in the current session and restart TensorBoard on the current `TENSORBOARD_ROOT`.

Each run writes to `/content/mics_project/<SESSION_ID>/`:

```text
preprocessing_artifacts/       # dictionary, audit, split summaries, EDA
results/experiments.csv        # full metrics and settings with run IDs
results/runs/E01..E10/<run-id>/
results/locked_selection.json
results/runtime_environment.json, environment_freeze.txt
models/best_classical.joblib
models/best_neural.keras, best_neural.weights.h5
models/best_neural_preprocessor.joblib
models/model_manifest.json
tensorboard/                   # event files for all runs
experiment_evidence_tensorboard.zip
```

The ZIP holds models, metrics and logs, not the raw dataset. Keep `EXPORT_PRIVATE_ARRAYS = False`.

To reuse a saved model, apply the notebook's `clean_feature` and `add_features` to your input, keep the feature order from the manifest, and do not pass raw SPSS codes to a model. Classical models include their fitted preprocessing. The neural model also needs `best_neural_preprocessor.joblib`.

## Limitations

- Validation data was reused for model selection, neural early stopping and threshold tuning, so validation scores are optimistic.
- One seed and one outer partition were studied. Classical learning curves are diagnostics at the selected settings, not nested estimates of the whole selection process.
- A test partition was inspected in earlier project work. Any later evaluation on it must disclose that history, and it is not an untouched holdout.
- 4,435 labels are reconstructed from never-attended answers, and the models perform very differently on them than on directly reported nonattendance.
- The models describe patterns in survey data. They are an educational research prototype and are not validated for decisions about individual children.

## Reproducibility

Seeds, pinned installs and deterministic TensorFlow settings improve reproducibility but do not guarantee identical results across runtimes or hardware. The saved run used numpy 2.1.3, pandas 2.3.3, pyreadstat 1.3.6, scikit-learn 1.6.1, matplotlib 3.10.0 and joblib 1.6.0. Its saved outputs do not show the TensorFlow, Keras or Python versions, so rerun the notebook and keep the generated `runtime_environment.json` and `environment_freeze.txt`. This is a Colab workflow, not a standalone local install.

## Repository contents

- `Nigeria_MICS_Summative.ipynb`: the Colab notebook with data audit, ten experiments, selection and TensorBoard dashboard.
- `requirements.txt`: pinned direct dependencies, matching the notebook's install cell.
- `results_snapshot/`: aggregate result tables, source fingerprint and TensorBoard run IDs, at displayed precision.
- `figures/`: PNG outputs from the reported run, including learning curves, confusion matrices and ROC and precision-recall plots.
- `.gitignore`: excludes raw data, trained models, event logs, evidence ZIPs and credentials.

## References

- [Nigeria MICS 2021 official catalog](https://microdata.worldbank.org/catalog/5959)
- [NBS and UNICEF 2021 MICS report](https://www.unicef.org/nigeria/reports/2021-multiple-indicator-cluster-survey-national-immunization-coverage-survey-report)
- [TensorBoard with Keras](https://www.tensorflow.org/tensorboard/scalars_and_keras)
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

The report carries the full IEEE-style references. No software or data licence is asserted by this repository. Respect the original dataset's access and use conditions.
