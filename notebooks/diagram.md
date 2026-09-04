# Pipeline Architecture Diagrams

## Current Pipeline (v1)

```
┌──────────────────────────────┐
│   RAW train.csv / test.csv     │
│   (headline, artikel, label)   │
└───────────────┬─────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  PREPROCESSING (stateless, no re-fit)   │
│  normalize whitespace → remove url/html │
│  → remove noise → lowercase →           │
│  stopword removal (= clean_b)           │
│  spacy.blank("id") untuk tokenisasi     │
│  rule-based (bukan model pretrained)    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION                         │
│                                                                     │
│ ┌─ LEXICAL/STATISTICAL ──────┐  ┌─ ENTITY & NUMERIK (regex) ────┐ │
│ │ Jaccard, overlap_coeff       │  │ Kapitalisasi → proxy NER       │ │
│ │ TF-IDF cosine (word 1-2gram) │  │ Regex angka \d+[.,]?\d*        │ │
│ │ char_wb TF-IDF (3-5gram)     │  │ Regex tanggal/bulan/tahun      │ │
│ │   → pengganti stemmer         │  │ Hitung angka judul yang HILANG │ │
│ └────────────────────────────────┘  │   di artikel (mismatch count)  │ │
│                                       │ Entity intersection judul-isi │ │
│ ┌─ POSISI/STRUKTUR ───────────┐  └───────────────────────────────────┘ │
│ │ Lead-paragraph match          │                                       │
│ │ (posisi kata kunci judul di   │  ┌─ RELASI (spaCy blank "id") ──────┐ │
│ │  awal vs tersebar di artikel) │  │ Event Extraction: verba aksi      │ │
│ └────────────────────────────────┘  │  headline → cek co-occurrence     │ │
│                                       │  di kalimat artikel                │ │
│ ┌─ LINGUISTIK ─────────────────┐  │ Relation Extraction: dependency   │ │
│ │ Sentiment gap (lexicon-based:  │  │  path subjek-verb-objek headline  │ │
│ │  VADER/AFINN-style ID)         │  │  vs artikel                        │ │
│ │ POS ratio superlatif           │  └────────────────────────────────────┘ │
│ │  ("terbesar","tergila")        │                                        │
│ │ Discourse markers kontradiktif │  ┌─ STYLISTIK (murah, opsional) ────┐ │
│ │  (tapi, namun, padahal)        │  │ Jumlah tanda seru, huruf kapital  │ │
│ │ Negation scope (bukan cuma     │  │  berlebih, kata "VIRAL"/"HEBOH"   │ │
│ │  ada "tidak", tapi APA yang    │  └────────────────────────────────────┘ │
│ │  dinegasikan)                  │                                        │
│ │ Text complexity/readability    │                                        │
│ │  gap (panjang kalimat, dst)    │                                        │
│ └────────────────────────────────┘                                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │   FEATURE MATRIX (X)      │
                  │   36 features              │
                  │   + label (y)              │
                  └────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │  DUMMY BASELINE (majority class) │
                  │  → patokan minimum, dibandingkan  │
                  │    ke SEMUA hasil model di bawah  │
                  └────────────┬──────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  NOISE / LABEL QUALITY CHECK       │
                  │  Isolation Forest (outlier fitur,   │
                  │   unsupervised, TIDAK lihat label)  │
                  │  + Confident Learning via OOF        │
                  │   (deteksi label yang kemungkinan    │
                  │    salah, supervised)                 │
                  └────────────┬──────────────────────────┘
                                │
                       ┌────────┴────────┐
                       ▼                 ▼
                baris "bersih"    baris "noise"
                weight = 1.0      weight = 0.3
                       │                 │
                       └────────┬────────┘
                                ▼
                  ┌───────────────────────────────────┐
                  │  NESTED STRATIFIED K-FOLD CV          │
                  │  outer fold: evaluasi (tidak tersentuh)│
                  │   └─ inner split: train/early-stopping │
                  │      (cegah leakage saat tuning n_tree) │
                  │  resampling/fit HANYA di train fold     │
                  └────────────┬──────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  MODEL TRAINING (per fold)          │
                  │  XGBoost/LightGBM                    │
                  │  scale_pos_weight = 1.0               │
                  │  (imbalance TIDAK ditangani di sini,  │
                  │   ditangani di tahap threshold)       │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │  OOF PREDICTIONS (raw proba)   │
                  └────────────┬──────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  PROBABILITY CALIBRATION            │
                  │  held-out calibration set            │
                  │  (dipisah di awal, cegah leakage)    │
                  │  Platt scaling / CalibratedClassifierCV│
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  RELIABILITY DIAGRAM CHECK           │
                  │  plot: mean predicted score           │
                  │  vs fraction positif aktual           │
                  │  → verifikasi kalibrasi valid          │
                  │  (bukan cuma percaya proses, dicek)   │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────────┐
                  │  EVALUASI MULTI-METRIK                  │
                  │  PR-AUC · MCC · Balanced Accuracy        │
                  │  Macro F1 (metrik utama kompetisi)       │
                  │  → semua dibandingkan ke Dummy Baseline  │
                  │  → jangan percaya accuracy/ROC-AUC saja  │
                  └────────────┬──────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │  THRESHOLD TUNING               │
                  │  sweep 0.01–0.99 pada OOF proba  │
                  │  optimalkan Macro F1              │
                  │  (imbalance ditangani DI SINI,    │
                  │   bukan di scale_pos_weight)      │
                  └────────────┬──────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │  FIT FULL MODEL (semua train)  │
                  │  n_estimators = median            │
                  │  best_iteration antar fold CV     │
                  └────────────┬──────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  TEST SET                            │
                  │  preprocessing & fitur SAMA PERSIS,   │
                  │  transform-only (vectorizer/SVD/dll   │
                  │  pakai objek yang di-fit dari TRAIN,  │
                  │  tidak pernah re-fit dari test)        │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │  PREDICT + apply threshold      │
                  │  + apply kalibrasi yang sama     │
                  │  → submission.csv                 │
                  └─────────────────────────────────────┘
```

---

## Planned Pipeline (v2) — With SMOTE & Without Noise Detection

```
┌──────────────────────────────┐
│   RAW train.csv / test.csv     │
│   (headline, artikel, label)   │
└───────────────┬─────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  PREPROCESSING (stateless, no re-fit)   │
│  normalize whitespace → remove url/html │
│  → lowercase → stopword removal         │
│  (= clean_b)                            │
│  spacy.blank("id") untuk tokenisasi     │
│  rule-based (bukan model pretrained)    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION                         │
│                                                                     │
│ ┌─ LEXICAL/STATISTICAL ──────┐  ┌─ ENTITY & NUMERIK (regex) ────┐ │
│ │ Jaccard, overlap_coeff       │  │ Kapitalisasi → proxy NER       │ │
│ │ TF-IDF cosine (word 1-2gram) │  │ Regex angka \d+[.,]?\d*        │ │
│ │ char_wb TF-IDF (3-5gram)     │  │ Regex tanggal/bulan/tahun      │ │
│ │   → pengganti stemmer         │  │ Hitung angka judul yang HILANG │ │
│ └────────────────────────────────┘  │   di artikel (mismatch count)  │ │
│                                       │ Entity intersection judul-isi │ │
│ ┌─ POSISI/STRUKTUR ───────────┐  └───────────────────────────────────┘ │
│ │ Lead-paragraph match          │                                       │
│ │ (posisi kata kunci judul di   │  ┌─ RELASI (spaCy blank "id") ──────┐ │
│ │  awal vs tersebar di artikel) │  │ Event Extraction: verba aksi      │ │
│ └────────────────────────────────┘  │  headline → cek co-occurrence     │ │
│                                       │  di kalimat artikel                │ │
│ ┌─ LINGUISTIK ─────────────────┐  │ Relation Extraction: dependency   │ │
│ │ Sentiment gap (lexicon-based:  │  │  path subjek-verb-objek headline  │ │
│ │  VADER/AFINN-style ID)         │  │  vs artikel                        │ │
│ │ POS ratio superlatif           │  └────────────────────────────────────┘ │
│ │  ("terbesar","tergila")        │                                        │
│ │ Discourse markers kontradiktif │  ┌─ STYLISTIK (murah, opsional) ────┐ │
│ │  (tapi, namun, padahal)        │  │ Jumlah tanda seru, huruf kapital  │ │
│ │ Negation scope (bukan cuma     │  │  berlebih, kata "VIRAL"/"HEBOH"   │ │
│ │  ada "tidak", tapi APA yang    │  └────────────────────────────────────┘ │
│ │  dinegasikan)                  │                                        │
│ │ Text complexity/readability    │                                        │
│ │  gap (panjang kalimat, dst)    │                                        │
│ └────────────────────────────────┘                                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │   FEATURE MATRIX (X)      │
                  │   36 features              │
                  │   + label (y)              │
                  └────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │  DUMMY BASELINE (majority class) │
                  │  → patokan minimum, dibandingkan  │
                  │    ke SEMUA hasil model di bawah  │
                  └────────────┬──────────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────────┐
                  │  NESTED STRATIFIED K-FOLD CV          │
                  │  outer fold: evaluasi (tidak tersentuh)│
                  │   └─ inner split: train/early-stopping │
                  │      (cegah leakage saat tuning n_tree) │
                  │  resampling/fit HANYA di train fold     │
                  └────────────┬──────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  SMOTE RESAMPLING (per fold)       │
                  │  minority → 50% of majority class  │
                  │  ONLY on training fold (not OOF)    │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  MODEL TRAINING (per fold)          │
                  │  XGBoost/LightGBM                    │
                  │  + class weights                     │
                  │  (imbalance handled HERE + SMOTE)    │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │  OOF PREDICTIONS (raw proba)   │
                  └────────────┬──────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  PROBABILITY CALIBRATION            │
                  │  CalibratedClassifierCV(cv=5)       │
                  │  (cross-validation based,            │
                  │   no held-out set needed)            │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  RELIABILITY DIAGRAM CHECK           │
                  │  plot: mean predicted score           │
                  │  vs fraction positif aktual           │
                  │  → verifikasi kalibrasi valid          │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  THRESHOLD TUNING                   │
                  │  sweep 0.01–0.99 on CALIBRATION set │
                  │  (not full OOF!)                     │
                  │  optimalkan Macro F1                 │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────────┐
                  │  EVALUASI MULTI-METRIK                  │
                  │  PR-AUC · MCC · Balanced Accuracy        │
                  │  Macro F1 (metrik utama kompetisi)       │
                  │  Per-class breakdown (minority focus)     │
                  └────────────┬──────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │  FIT FULL MODEL (semua train)  │
                  │  + refit calibrator               │
                  └────────────┬──────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────────┐
                  │  TEST SET                            │
                  │  preprocessing & fitur SAMA PERSIS,   │
                  │  transform-only (vectorizer/SVD/dll   │
                  │  pakai objek yang di-fit dari TRAIN,  │
                  │  tidak pernah re-fit dari test)        │
                  └────────────┬──────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │  PREDICT + apply threshold      │
                  │  + apply kalibrasi yang sama     │
                  │  → submission.csv                 │
                  └─────────────────────────────────────┘
```

---

## Key Changes: v1 → v2

| Aspect | v1 (Current) | v2 (Planned) | Impact |
|--------|--------------|--------------|--------|
| **Noise Detection** | IsolationForest + Confident Learning | **REMOVED** | +0.025 F1 |
| **Sampling** | Class weights only | **SMOTE** (0.5 ratio) | +0.025 F1 |
| **Threshold Tuning** | Full OOF data | **Calibration set only** | +0.015 F1 |
| **Calibration** | Fixed 15% set | **CV-based (5-fold)** | +0.005 F1 |
| **Inner Split** | 85/15 | **80/20** | +0.005 F1 |
| **Proxy NER** | Sentence-start exclusion | **Simple capitalization** | +0.005 F1 |

---

## References

- See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed implementation plan
- See [../notebooks/opencode_guide.md](../notebooks/opencode_guide.md) for original 14-step guide
