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