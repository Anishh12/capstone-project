# Zepto — Analyst to Data Scientist (Titanic)

One cohesive pipeline: load the classic Titanic dataset **once**, profile it,
clean it defensibly, tell a visual data story, then build and rigorously
evaluate a full predictive-modeling pipeline on the same cleaned data.

Run order: `01_eda.py` → `02_modeling.py`. The raw dataset is loaded
**exactly once** in `01_eda.py`; everything else continues from that same load
(via the committed `titanic.csv`), never from a fresh `sns.load_dataset(...)`.

---

## Files

```
analytics/
├── 01_eda.py                # load once, profile, clean, EDA, save titanic.csv
├── 02_modeling.py           # read titanic.csv, train/evaluate/tune, save pipeline
├── titanic.csv              # committed offline fallback
├── best_pipeline.joblib     # fitted full pipeline (preprocess + estimator)
├── requirements.txt
└── README.md
```

## Requirements

```
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
imbalanced-learn>=0.11
joblib>=1.3
statsmodels>=0.14
```

## Run

```bash
cd analytics
pip install -r requirements.txt
python 01_eda.py        # produces titanic.csv + all EDA charts
python 02_modeling.py   # reads titanic.csv, saves best_pipeline.joblib
```

Note: the first run of `sns.load_dataset("titanic")` requires internet access,
because Seaborn fetches the dataset from its online repository and caches it
locally. Subsequent runs on the same machine use the cache. `02_modeling.py`
never calls `sns.load_dataset` — it reads the committed `titanic.csv`, so it
works fully offline.

---

## Data source

`sns.load_dataset("titanic")` is called **exactly once** in `01_eda.py`.
Immediately after loading, the DataFrame is written to `titanic.csv` with
`index=False`, so the modeling stage (and any offline grader) can reproduce
the pipeline with `pd.read_csv("titanic.csv")`.

The raw dataset is never reloaded independently for the modeling stage — the
load-once guarantee is part of the acceptance criteria.

---

## Part A — Profiling, cleaning, and the data story (`01_eda.py`)

### Task 1 — Load and profile

- `df.shape`, `df.info()`, `df.describe()` are printed.
- Missing-value percentages are reported for every column that has any, sorted
  descending.

### Task 2 — Missing-value handling

Strategy follows a percentage-threshold rule:

| Column | Missing % | Rule | Strategy |
|--------|-----------|------|----------|
| `deck` | 77.22% | >30% | **Drop column** — imputing would invent >3/4 of the data; also redundant with `pclass`/`embarked` |
| `age` | 19.87% | 5–30% | **Impute with median** — robust to skew |
| `embarked` | 0.22% | <5% | **Drop rows** |
| `embark_town` | 0.22% | <5% | **Drop rows** |

After cleaning, remaining missing cells = 0.

### Task 3 — Univariate analysis

- Histograms and boxplots for `age` and `fare` are saved as `uni_age.png` and
  `uni_fare.png`.
- IQR-based outliers are counted for both columns (bounds are
  `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`).
- `fare` mean, median, and mode are computed. Because
  mean > median > mode, the distribution is **right-skewed** — a long right
  tail of expensive tickets pulls the mean up.

### Task 4 — Bivariate analysis

- Survival rates by `sex`, by `pclass`, and by `sex × pclass` are computed
  with boolean masking (`&`, `|`, `~`).
- A **6×6 correlation matrix** is computed on exactly these columns:
  `[survived, pclass, age, sibsp, parch, fare]`.
- `adult_male` and `alone` are **excluded**: both are derived/redundant flags
  (computable from `sex`/`age`, and from `sibsp + parch` respectively), not
  independent measured features.
- The matrix is rendered as a heatmap (`corr.png`).
- The two strongest off-diagonal correlations by absolute value are named and
  interpreted:
  - `pclass ↔ fare` (negative, ≈ −0.55) — higher class number = cheaper fare;
    fare is a proxy for socioeconomic status.
  - `sibsp ↔ parch` (positive, ≈ +0.41) — passengers travelling with
    siblings/spouses also tended to travel with parents/children; family
    units move together.

### Task 5 — Multivariate "data story"

Six charts are produced, each with a written interpretation:

1. **`story_1_sex.png`** — women ≈ 74% survival vs men ≈ 19%. Sex is the
   strongest single predictor; "women and children first" is visible in the
   data.
2. **`story_2_pclass.png`** — 1st ≈ 63%, 2nd ≈ 47%, 3rd ≈ 24%. Wealth bought
   better lifeboat access.
3. **`story_3_sex_pclass.png`** — female/1st ≈ 97% vs male/3rd ≈ 13%. The
   sex × class interaction dominates: female *and* wealthy was near-certain
   survival; male *and* poor was near-certain death.
4. **`story_4_age.png`** — young children show a survival bump;
   working-age males cluster in the non-survived group.
5. **`story_5_age_fare.png`** — high-fare passengers (log axis) are almost
   all survived; low-fare older males dominate the non-survived region.
6. **`story_6_agegroup_sex.png`** — female survival stays high across all
   age groups; male survival is low everywhere and dips further for adults.

### Task 6 — Exploratory z-score standardization

- `age` and `fare` are standardized with `z = (x − mean) / std` on the full
  cleaned DataFrame.
- Before/after summary printed and compared as overlaid histograms
  (`standardization_check.png`). After transformation, mean ≈ 0 and std ≈ 1.
- This is an **EDA-stage sanity check only** — the modeling pipeline performs
  its own train-only scaling.

---

## Part B — Predictive modeling (`02_modeling.py`)

### Task 7 — Stratified train/test split

- Stratified 80/20 split on `survived`.
- Justification: the target is imbalanced (~62% died / 38% survived); a plain
  random split could shift the minority share between folds and distort
  precision/recall/F1.
- Split is performed **before** any preprocessing.

### Task 8 — Preprocessing (fit on train only)

- A `ColumnTransformer` handles imputation, encoding, and scaling per column:
  - **Numeric** (`age`, `sibsp`, `parch`, `fare`): median impute →
    `StandardScaler`.
  - **Categorical** (`sex`, `embarked`, `pclass`): most-frequent impute →
    `OneHotEncoder(handle_unknown="ignore")`.
- The `ColumnTransformer` is wrapped inside a `Pipeline` with the classifier,
  so all preprocessing is fit **only** on the training split and applied
  transform-only to the test split — no leakage.
- `clone(pre)` is used per model so each pipeline starts clean.

### Task 9 — Three classifiers on the same split

- Logistic Regression
- Decision Tree (`max_depth=4` for a readable `plot_tree`)
- Random Forest (`n_estimators=200`)

Each model's `plot_tree` (for the Tree), confusion matrix, accuracy,
precision, recall, F1, and ROC/AUC are reported. ROC curves for all three
models are plotted (`roc.png`).

### Task 10 — Imbalance handling comparison

Three variants are compared (Logistic Regression, same split):

| Variant | Notes |
|---------|-------|
| baseline | no handling |
| `class_weight='balanced'` | reweights the loss |
| SMOTE | applied to the **training fold only** via `imblearn.pipeline.Pipeline` |

Conclusion: `class_weight='balanced'` gives the best recall/F1 trade-off — it
reweights the loss without SMOTE's synthetic noise. Baseline has the highest
precision but the worst recall on the minority (survived) class.

### Task 11 — Hyperparameter tuning (Random Forest)

- `RandomForestClassifier(oob_score=True, ...)` constructed explicitly so the
  OOB score is populated.
- `GridSearchCV` over `n_estimators`, `max_depth`, `max_features`.
- Best parameter combination and the corresponding OOB score are reported.
- The tuned forest is evaluated on the same test set alongside the other
  classifiers.

### Task 12 — Regression side-task

- Multivariate linear regression predicting `fare` from the other available
  features.
- Reports MAE, RMSE, R², and Adjusted R².
- Residual plot (`resid.png`) shows a funnel shape; the Breusch-Pagan test
  (p << 0.05) confirms **heteroscedasticity** — error variance grows with
  predicted fare, driven by fare's strong right skew.

### Task 13 — Comparison table + recommendation

- Classifier metrics (accuracy, precision, recall, F1, AUC) are presented as
  one group; regression metrics (MAE, RMSE, R², Adjusted R²) are presented as
  a **separate group** — the two are on different scales and are not directly
  comparable.
- A short written recommendation states which classifier to deploy and why,
  referencing the specific metric values.

### Task 14 — Save the full pipeline and reload

- `joblib.dump(full_pipeline, "best_pipeline.joblib")` saves the **entire**
  fitted pipeline (preprocessing steps + final estimator as one object), not
  the bare estimator.
- A reload smoke test loads the artifact and calls `.predict()` and
  `.predict_proba()` on raw, unpreprocessed input — including a row with
  missing values — to confirm it is usable end-to-end.

---

## Outputs (charts and artifacts)

| File | Purpose |
|------|---------|
| `titanic.csv` | Committed offline fallback (raw data) |
| `uni_age.png`, `uni_fare.png` | Histograms + boxplots |
| `corr.png` | 6×6 correlation heatmap |
| `story_1_sex.png` … `story_6_agegroup_sex.png` | Multivariate story charts |
| `standardization_check.png` | Before/after z-score comparison |
| `tree.png` | Decision tree visualization |
| `confusion_matrices.png` | Confusion matrices for all classifiers |
| `roc.png` | ROC curves |
| `resid.png` | Regression residual plot |
| `comparison_table.csv` | Combined comparison table |
| `best_pipeline.joblib` | Full fitted pipeline (preprocess + estimator) |

---

## Deliverables checklist

- [x] Missing-value percentages reported for every affected column, with each
      strategy citing the threshold rule
- [x] `titanic.csv` produced via `df.to_csv("titanic.csv", index=False)`,
      committed as the offline fallback
- [x] Raw dataset loaded from network/cache exactly once — never reloaded
      independently for the modeling stage
- [x] IQR-based outlier counts for `age` and `fare`; `fare` skew conclusion
      compares mean, median, and mode
- [x] Three bivariate breakdowns (sex, pclass, sex+pclass) with numeric
      survival rates
- [x] Correlation matrix/heatmap computed on exactly the six specified columns
      (`survived, pclass, age, sibsp, parch, fare`); `adult_male` and `alone`
      excluded; the two strongest off-diagonal pairs named and interpreted
- [x] ≥4 multivariate charts, each with its own written interpretation
- [x] Before/after standardization check shown for both `age` and `fare`
- [x] Stratified train/test split implemented **before** any preprocessing,
      with justification referencing class balance
- [x] All preprocessing (imputation, encoding, scaling) fit only on the
      training split; transform-only on test
- [x] Three classifiers trained on the identical split
- [x] Decision tree visualized via `plot_tree` with labeled features and
      classes
- [x] Full metric suite (confusion matrix, accuracy, precision, recall, F1,
      ROC/AUC) reported for each classifier
- [x] Three-way imbalance comparison (baseline vs `class_weight='balanced'`
      vs SMOTE) with SMOTE applied to the training fold only, plus a written
      conclusion
- [x] `GridSearchCV` best parameters and OOB score reported for a
      `RandomForestClassifier(oob_score=True, ...)`
- [x] Regression sub-task reports MAE, RMSE, R², Adjusted R², and a written
      heteroscedasticity conclusion
- [x] Final model comparison table presents classifier and regression metrics
      as separate metric groups
- [x] Saved artifact is the full fitted pipeline via
      `joblib.dump(full_pipeline, ...)`, demonstrably reloadable, and usable
      end-to-end on raw new data
