# Architecture & Improvement Plan

## Current Architecture (v1)

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE v1 (CURRENT)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RAW DATA → PREPROCESSING → 36 FEATURES → MODEL ZOO → SUBMIT   │
│                                                                  │
│  Components:                                                     │
│  • Preprocessing: Rule-based (no pretrained model)              │
│  • Features: 36 handcrafted (11 transformer groups + LSA)       │
│  • Models: LightGBM, XGBoost, RandomForest, LogisticRegression │
│  • Class Handling: Class weights only (no sampling)             │
│  • Evaluation: 5-fold Stratified CV + calibration               │
│                                                                  │
│  Performance:                                                    │
│  • CV Macro F1: ~0.57                                            │
│  • Kaggle Macro F1: 0.54                                         │
│                                                                  │
│  CONSTRAINT: Pretrained models (BERT, IndoBERT, etc.) TIDAK      │
│  DIBOLEHKAN dalam kompetisi ini.                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Upcoming Architecture (v2) — Target: 0.60+

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE v2 (UPCOMING)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RAW DATA → PREPROCESSING → 45 FEATURES → MODEL ZOO → SUBMIT   │
│                                                                  │
│  Key Changes from v1:                                            │
│  • +8 new features (Hybrid TF-IDF + N-gram Overlap)            │
│  • +SMOTE oversampling (remove noise detection)                 │
│  • +Bayesian hyperparameter tuning (Optuna)                     │
│  • +Feature matrix EDA                                           │
│  • +Stacking ensemble option                                     │
│                                                                  │
│  Expected Performance:                                           │
│  • CV Macro F1: >0.60                                            │
│  • Kaggle Macro F1: >0.60                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### New Feature Groups (v2)

| # | Group | Features | Count | Description |
|---|-------|----------|-------|-------------|
| 13 | **Hybrid TF-IDF** | char_tfidf_cosine, word_tfidf_cosine, hybrid_mean, hybrid_diff | 4 | char_wb (3-5gram) + word (1-2gram) |
| 14 | **N-gram Overlap** | unigram_overlap, bigram_overlap, trigram_overlap, weighted_overlap | 4 | Multi-level n-gram overlap |
| | **TOTAL** | | **45** | (+8 from v1) |

### Current Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING FLOW (v1)                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Feature Extraction (full training set)                  │
│     └─ 36 features via FeatureUnion pipeline                │
│                                                              │
│  2. Noise Detection (Isolation Forest + Confident Learning) │
│     └─ Downweight 30% samples to 0.3                       │
│                                                              │
│  3. Nested Cross-Validation                                 │
│     ├─ Outer: 5-fold Stratified (for OOF evaluation)        │
│     └─ Inner: 85/15 split (for early stopping)              │
│                                                              │
│  4. Model Zoo Training                                      │
│     ├─ LightGBM (is_unbalance=True)                         │
│     ├─ XGBoost (scale_pos_weight=9.0)                       │
│     ├─ RandomForest (class_weight='balanced_subsample')     │
│     └─ LogisticRegression (class_weight='balanced')         │
│                                                              │
│  5. Probability Calibration                                 │
│     └─ Platt scaling on calibration set (15%)               │
│                                                              │
│  6. Threshold Tuning                                        │
│     └─ Sweep 0.01-0.99 to maximize Macro F1                │
│                                                              │
│  7. Final Model                                             │
│     └─ Refit best model on full training data               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Improvement Plan (v2) — Target: 0.60+

### Priority 1: FEATURE ENGINEERING (v2)

#### 1.1 Hybrid TF-IDF Features ⚠️ HIGH IMPACT

**Problem:** Current TF-IDF only uses word-level 1-2gram. Missing character-level patterns.

**Fix:** Add `HybridTfIdfFeatures` class with:
- char_wb TF-IDF (3-5gram) — captures subword patterns, typos
- word-level TF-IDF (1-2gram) — captures semantic meaning
- Cosine similarity for both
- Mean and difference features

**Implementation:**
```python
class HybridTfIdfFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Fit char_wb vectorizer (3-5gram)
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), max_features=5000
        )
        # Fit word vectorizer (1-2gram)
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), max_features=5000
        )
        return self
    
    def transform(self, X):
        # Compute cosine similarity for both
        # Return 4 features: char_cosine, word_cosine, mean, diff
```

**Features Added:** 4 (char_tfidf_cosine, word_tfidf_cosine, hybrid_mean, hybrid_diff)

**Expected Impact:** +0.02-0.03 Macro F1

**Status:** [ ] TODO

---

#### 1.2 N-gram Overlap Features ⚠️ HIGH IMPACT

**Problem:** Current overlap features only use unigram. Missing bigram/trigram patterns.

**Fix:** Add `NgramOverlapFeatures` class with:
- Unigram overlap (Jaccard)
- Bigram overlap
- Trigram overlap
- Weighted overlap (0.2*uni + 0.3*bi + 0.5*tri)

**Implementation:**
```python
class NgramOverlapFeatures(BaseEstimator, TransformerMixin):
    def _compute(self, row):
        h_tokens = row['headline_tokens'].split()
        a_tokens = row['article_tokens'].split()
        
        # Unigram overlap
        unigram_overlap = len(set(h_tokens) & set(a_tokens)) / len(set(h_tokens) | set(a_tokens))
        
        # Bigram overlap
        h_bigrams = set(zip(h_tokens[:-1], h_tokens[1:]))
        a_bigrams = set(zip(a_tokens[:-1], a_tokens[1:]))
        bigram_overlap = len(h_bigrams & a_bigrams) / len(h_bigrams | a_bigrams)
        
        # Trigram overlap (similar)
        # Weighted overlap
```

**Features Added:** 4 (unigram_overlap, bigram_overlap, trigram_overlap, weighted_overlap)

**Expected Impact:** +0.01-0.02 Macro F1

**Status:** [ ] TODO

---

#### 1.3 Feature Matrix EDA ⚠️ MEDIUM IMPACT

**Problem:** No analysis of feature quality, correlations, or importance.

**Fix:** Add EDA cell after feature extraction:

**1.3.1 Feature Distributions**
- Histogram per fitur (cek skewness, outliers)
- Boxplot per class (cek separability)

**1.3.2 Correlation Matrix** ⚠️ IMPORTANT
- Heatmap korelasi antar semua 45 fitur
- Identifikasi fitur redundant (korelasi > 0.8)
-Drop atau combine fitur yang highly correlated
- saves correlation matrix sebagai gambar

```python
# Correlation Matrix
corr = X_df.corr()

# Plot
plt.figure(figsize=(20, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
            center=0, fmt='.2f', square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('../reports/figures/correlation_matrix.png', dpi=150)
plt.show()

# Identify highly correlated pairs
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.8:
            high_corr.append({
                'feature_1': corr.columns[i],
                'feature_2': corr.columns[j],
                'correlation': corr.iloc[i, j]
            })

print("Highly correlated features (|r| > 0.8):")
print(pd.DataFrame(high_corr).to_string(index=False))
```

**1.3.3 Feature Importance (Mutual Information)**
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_df, y, random_state=SEED)
mi_df = pd.DataFrame({
    'feature': feature_cols, 
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("Top 20 features by Mutual Information:")
print(mi_df.head(20).to_string(index=False))

# Plot
plt.figure(figsize=(10, 8))
top_n = min(25, len(mi_df))
plt.barh(range(top_n), mi_df['mi_score'].head(top_n).values)
plt.yticks(range(top_n), mi_df['feature'].head(top_n).values)
plt.xlabel('Mutual Information Score')
plt.title('Top Features by Mutual Information')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../reports/figures/feature_importance_mi.png', dpi=150)
plt.show()
```

**1.3.4 Class Separation Analysis**
```python
# Per-feature boxplot by class
fig, axes = plt.subplots(9, 5, figsize=(25, 40))
for i, col in enumerate(feature_cols):
    ax = axes[i//5, i%5]
    for label in [0, 1]:
        ax.hist(X_df.loc[y == label, col], bins=30, alpha=0.5, 
                label=f'Class {label}', density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('../reports/figures/feature_distributions_by_class.png', dpi=150)
plt.show()
```

**Expected Impact:** +0.005 (indirect — informs feature selection)

**Status:** [ ] TODO

---

### Priority 2: CLASS IMBALANCE (v2)

#### 2.1 Remove Noise Detection ⚠️ HIGH IMPACT

**Problem:** IsolationForest flags minority class as outliers → downweight to 0.3.

**Fix:** Skip Cell 22 entirely. Set all sample weights to 1.0.

**Expected Impact:** +0.02-0.03 Macro F1

**Status:** [ ] TODO

---

#### 2.2 Add SMOTE ⚠️ HIGH IMPACT

**Problem:** No sampling strategy for 9:1 imbalance.

**Fix:** Add SMOTE inside CV folds:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=SEED)
X_resampled, y_resampled = smote.fit_resample(X_train_fold, y_train_fold)
model.fit(X_resampled, y_resampled)
```

**Install:** `pip install imblearn`

**Expected Impact:** +0.02-0.03 Macro F1

**Status:** [ ] TODO

---

### Priority 3: HYPERPARAMETER TUNING (v2)

#### 3.1 Bayesian Optimization ⚠️ HIGH IMPACT

**Problem:** No systematic hyperparameter tuning. Current params are manual.

**Fix:** Use Optuna for Bayesian optimization:
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    # Train with CV and return macro F1
    model = lgb.LGBMClassifier(**params, is_unbalance=True)
    # ... cross-validation ...
    return mean_macro_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**Install:** `pip install optuna`

**Expected Impact:** +0.02-0.03 Macro F1

**Status:** [ ] TODO

---

#### 3.2 Threshold Tuning on Calibration Set ⚠️ MEDIUM IMPACT

**Problem:** Threshold tuned on full OOF data (data leakage).

**Fix:** Tune threshold on calibration set only:
```python
cal_proba = oof_proba[cal_indices]  # Only 15% held-out
# Sweep threshold on cal_proba
# Use best threshold for final predictions
```

**Expected Impact:** +0.01-0.02 Macro F1

**Status:** [ ] TODO

---

### Priority 4: MODEL ENSEMBLE (v2)

#### 4.1 Stacking Ensemble ⚠️ MEDIUM IMPACT

**Problem:** Single model selection may not be optimal.

**Fix:** Add stacking ensemble:
```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ('lgb', lgb.LGBMClassifier(**best_params_lgb)),
    ('xgb', xgb.XGBClassifier(**best_params_xgb)),
    ('lr', LogisticRegression(class_weight='balanced'))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)
```

**Expected Impact:** +0.01-0.02 Macro F1

**Status:** [ ] TODO

---

#### 1.3 Fix Threshold Tuning ⚠️ HIGH IMPACT

**Problem:** Threshold tuned on full OOF data, which is influenced by model training. Should use calibration set.

**Current Code (Cell 30):**
```python
# Uses full oof_proba_calibrated
for thresh in thresholds:
    y_pred_thresh = (oof_proba_calibrated >= thresh).astype(int)
    # ↑ This uses ALL OOF predictions
```

**Fix:** Tune threshold on calibration set only:
```python
cal_proba = oof_proba[cal_indices]  # Only 15% held-out data
# Tune threshold on cal_proba, not oof_proba
```

**Expected Impact:** +0.01-0.02 Macro F1

**Status:** [ ] TODO

---

### Priority 2: FEATURE ENGINEERING

#### 2.1 Fix Proxy NER ⚠️ MEDIUM IMPACT

**Problem:** Current regex `(?<!\.\s+)[A-Z][a-z]+` is too crude. Misses:
- Lowercase proper nouns (common in Indonesian)
- All-caps abbreviations (KPK, DPR, TNI)
- Words at sentence start (excluded by negative lookbehind)

**Current Code (Cell 15):**
```python
h_caps = set(re.findall(r'(?<!\.\s+)[A-Z][a-z]+', h_raw))
```

**Fix:** Simplify regex to match all capitalized words:
```python
h_caps = set(re.findall(r'\b[A-Z][a-zA-Z]+\b', h_raw))
```

**Expected Impact:** +0.005-0.01 Macro F1

**Status:** [ ] TODO

---

#### 2.2 Add Feature Selection ⚠️ MEDIUM IMPACT

**Problem:** 36 features, many may be noisy or correlated. No feature selection step.

**Fix:** Add `SelectKBest` or RFE after feature extraction:
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=25)  # Select top 25 from 36
X_selected = selector.fit_transform(X_model, y_model)
```

**Alternative:** Use feature importance from LightGBM to prune low-importance features.

**Expected Impact:** +0.005-0.01 Macro F1

**Status:** [ ] TODO

---

### Priority 3: MODEL TUNING

#### 3.1 Increase Inner Split Ratio ⚠️ LOW IMPACT

**Problem:** 85/15 split inside 5-fold CV means model trains on only ~68% of data per fold. May cause underfitting.

**Current Code (Cell 26):**
```python
X_inner_train, X_inner_val, ... = train_test_split(
    ..., test_size=0.15, ...  # ← Only 15% for early stopping
)
```

**Fix:** Increase to 20%:
```python
test_size=0.2  # From 0.15
```

**Expected Impact:** +0.005 Macro F1

**Status:** [ ] TODO

---

#### 3.2 Use CV-Based Calibration ⚠️ LOW IMPACT

**Problem:** Calibration set only 15% of data. May be too small for reliable Platt scaling.

**Current Code (Cell 28):**
```python
final_calibrator = LogisticRegression(random_state=SEED)
final_calibrator.fit(oof_proba[cal_indices].reshape(-1, 1), y_model[cal_indices])
```

**Fix:** Use `CalibratedClassifierCV` with cross-validation:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    cv=5,
    method='isotonic'  # or 'sigmoid'
)
calibrated_model.fit(X_model, y_model)
```

**Expected Impact:** +0.005 Macro F1

**Status:** [ ] TODO

---

### Priority 4: OPTIONAL IMPROVEMENTS

#### 4.1 Fix spaCy Tokenizer Performance ⚠️ LOW IMPACT

**Problem:** Creates new `spacy.blank("id")` model for every text sample. Extremely slow.

**Current Code (Cell 7):**
```python
@staticmethod
def _tokenize(text):
    nlp = spacy.blank("id")  # ← New model per call!
    doc = nlp(str(text))
    return [token.text for token in doc]
```

**Fix:** Use `str.split()` instead (identical results for whitespace tokenization):
```python
@staticmethod
def _tokenize(text):
    return str(text).split()
```

**Expected Impact:** 10-100x faster preprocessing, no F1 impact

**Status:** [ ] TODO

---

#### 4.2 Remove Redundant EDA Computation ⚠️ LOW IMPACT

**Problem:** Lexical overlap features computed twice (EDA in Cell 10, Features in Cell 14). EDA versions never used.

**Fix:** Remove Cell 10 overlap computation or reuse its results.

**Expected Impact:** Faster execution, no F1 impact

**Status:** [ ] TODO

---

## Implementation Checklist

### Phase 1: Feature Engineering (v2)
- [ ] Add `HybridTfIdfFeatures` class (char_wb + word-level TF-IDF)
- [ ] Add `NgramOverlapFeatures` class (uni/bi/trigram overlap)
- [ ] Update FeatureUnion with 2 new transformers
- [ ] Feature Matrix EDA:
  - [ ] Correlation matrix + identify highly correlated pairs
  - [ ] Mutual information feature importance
  - [ ] Class separation analysis (boxplot per feature)
  - [ ] Save figures to `reports/figures/`

### Phase 2: Class Imbalance
- [ ] Remove noise detection (Cell 22)
- [ ] Install imblearn: `pip install imblearn`
- [ ] Add SMOTE to model zoo training (Cell 26)

### Phase 3: Hyperparameter Tuning
- [ ] Install Optuna: `pip install optuna`
- [ ] Add Bayesian optimization cell (after model zoo)
- [ ] Fix threshold tuning to use calibration set (Cell 30)

### Phase 4: Model Ensemble
- [ ] Add stacking ensemble option
- [ ] Compare single model vs stacking performance

### Phase 5: Optional Optimizations
- [ ] Fix proxy NER regex (Cell 15)
- [ ] Replace spaCy tokenizer with str.split() (Cell 7)
- [ ] Feature selection based on MI scores

---

## Expected Outcome

| Fix | Macro F1 Delta | Cumulative | Status |
|-----|----------------|------------|--------|
| Hybrid TF-IDF (char_wb + word) | +0.025 | 0.565 | [ ] TODO |
| N-gram overlap features | +0.015 | 0.580 | [ ] TODO |
| Feature matrix EDA | +0.005 | 0.585 | [ ] TODO |
| Remove noise detection | +0.025 | 0.610 | [ ] TODO |
| Add SMOTE | +0.025 | 0.635 | [ ] TODO |
| Bayesian hyperparameter tuning | +0.025 | 0.660 | [ ] TODO |
| Threshold on cal set | +0.015 | 0.675 | [ ] TODO |
| Stacking ensemble | +0.015 | 0.690 | [ ] TODO |

**Target: 0.65-0.70 Macro F1**

---

## Testing Strategy

After implementing fixes, run in this order:

1. **Feature Engineering Test:**
   - Verify new features added to feature matrix (45 features total)
   - Check correlation matrix for highly correlated pairs
   - Review MI scores for feature importance

2. **Class Imbalance Test:**
   - Remove noise detection → verify all weights = 1.0
   - Add SMOTE → verify minority class oversampled

3. **Hyperparameter Tuning Test:**
   - Run Optuna study (100 trials)
   - Compare best params vs default params
   - Verify threshold uses calibration set

4. **Full Pipeline Run:**
   ```bash
   python 01_full_code.py
   ```

5. **Compare Results:**
   - CV Macro F1 should increase from ~0.57 to >0.65
   - Kaggle submission should improve from 0.54

6. **Analyze Confusion Matrix:**
   - Minority recall should increase from 24.5% to >50%
   - Major F1 may decrease slightly (trade-off)

---

## Notes

- **No BM25:** Excluded due to laptop constraints (rank_bm25 package can be memory-intensive)
- **No pretrained models:** Competition rules prohibit pretrained embeddings
- **spaCy blank("id"):** Only used for tokenization, no POS/dependency parsing
- **All features are numeric:** Enables correlation matrix analysis

---

## References

- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Optuna: Hyperparameter Optimization Framework](https://optuna.org/)
- [Scikit-learn Calibration](https://scikit-learn.org/stable/modules/calibration.html)
