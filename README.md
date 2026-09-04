# Penyisihan DAC IFEST 2026

Repository eksperimen untuk kompetisi [Penyisihan DAC IFEST 2026](https://www.kaggle.com/competitions/penyisihan-dac-ifest-2026/overview).

## Anggota

| Peran | Nama |
| --- | --- |
| Anggota 1 | `<nama-anggota-1>` |
| Anggota 2 | `<nama-anggota-2>` |
| Anggota 3 | `<nama-anggota-3>` |

## Deskripsi Kompetisi

Kompetisi ini berfokus pada klasifikasi biner pasangan judul dan isi berita. Model harus menentukan apakah judul didukung oleh isi berita (`1`, **Sesuai**) atau tidak (`0`, **Tidak Sesuai**). Penilaian tidak cukup dilakukan dengan kemiripan kata karena pasangan dapat memiliki tokoh, lokasi, waktu, angka, atau topik yang sama tetapi merujuk pada peristiwa berbeda.

Metrik evaluasi utama adalah **Macro F1-Score**, sehingga performa pada kedua kelas perlu diperhatikan.

## Deskripsi Data

Dataset berisi pasangan judul dan isi berita dengan kolom berikut:

- `id`: identifier unik pasangan berita.
- `title`: judul berita.
- `content`: isi atau badan berita.
- `label`: target klasifikasi; `1` untuk Sesuai dan `0` untuk Tidak Sesuai. Kolom ini tersedia pada data training.

Data asli disimpan di `data/raw/penyisihan-dac-ifest-2026/`. Data turunan yang belum final disimpan di `data/interim/`, data siap pemodelan di `data/processed/`, dan sumber eksternal di `data/external/`.

Format submission wajib memiliki header `id,label`, dengan satu baris untuk setiap data uji.

## Arsitektur Pipeline

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE OVERVIEW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RAW DATA → PREPROCESSING → 36 FEATURES → MODEL ZOO → SUBMIT   │
│                                                                  │
│  Key Components:                                                 │
│  • Preprocessing: Stateless, rule-based (no pretrained model)   │
│  • Features: 36 handcrafted features (11 transformer groups)    │
│  • Models: LightGBM, XGBoost, RandomForest, LogisticRegression │
│  • Class Handling: SMOTE + class weights + threshold tuning     │
│  • Evaluation: 5-fold Stratified CV + calibration               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Groups (36 Features → 45 Features)

### Current Features (v1)

| # | Group | Features | Count | Description |
|---|-------|----------|-------|-------------|
| 1 | **Lexical** | jaccard_a, overlap_a, jaccard_b, overlap_b, dice_b | 5 | Text overlap metrics |
| 2 | **Entity/Numeric** | proxy_ner, numeric_mismatch, numeric_total, date_mismatch, entity_ratio | 5 | Named entity & number matching |
| 3 | **Position** | lead_paragraph, headline_coverage | 2 | Keyword position analysis |
| 4 | **Linguistic** | sentiment_gap, superlative, discourse, negation, readability | 5 | Language feature analysis |
| 5 | **Relation** | event_extraction, svo_overlap, relation_score | 3 | Subject-Verb-Object extraction |
| 6 | **Stylistic** | exclamation, caps_ratio, buzzword | 3 | Writing style features |
| 7 | **TF-IDF Cosine** | tfidf_cosine | 1 | TF-IDF similarity |
| 8 | **Length Ratio** | word_ratio, char_ratio | 2 | Text length comparison |
| 9 | **Keyword Alignment** | keyword_coverage, keyword_context_match, keyword_absent_count | 3 | TF-IDF keyword matching |
| 10 | **Entity Overlap** | person_overlap, org_overlap, location_overlap, entity_overlap_ratio | 4 | Regex-based entity matching |
| 11 | **Topic Coherence** | topic_cosine, topic_match, topic_kl_div | 3 | NMF topic modeling |
| 12 | **LSA** | lsa_cosine_sim | 1 | Latent Semantic Analysis |
| | **Subtotal** | | **37** | |

### Upcoming Features (v2)

| # | Group | Features | Count | Description |
|---|-------|----------|-------|-------------|
| 13 | **Hybrid TF-IDF** | char_tfidf_cosine, word_tfidf_cosine, hybrid_mean, hybrid_diff | 4 | char_wb (3-5gram) + word (1-2gram) TF-IDF |
| 14 | **N-gram Overlap** | unigram_overlap, bigram_overlap, trigram_overlap, weighted_overlap | 4 | Multi-level n-gram overlap |
| | **TOTAL** | | **45** | |

### Model Zoo

| Model | Class Weight | Hyperparameters |
|-------|--------------|-----------------|
| **LightGBM** | `is_unbalance=True` | n_estimators=1000, max_depth=6, lr=0.05, num_leaves=31 |
| **XGBoost** | `scale_pos_weight=9.0` | n_estimators=500, max_depth=6, lr=0.05, min_child_weight=10 |
| **RandomForest** | `class_weight='balanced_subsample'` | n_estimators=300, max_depth=10, min_samples_leaf=10 |
| **LogisticRegression** | `class_weight='balanced'` | C=0.1, max_iter=1000 |

**Best Model Selection:** By Macro F1 on OOF predictions

### Training Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING FLOW                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Feature Extraction (full training set)                  │
│     └─ 36 features via FeatureUnion pipeline                │
│                                                              │
│  2. Nested Cross-Validation                                 │
│     ├─ Outer: 5-fold Stratified (for OOF evaluation)        │
│     └─ Inner: 80/20 split (for early stopping)              │
│                                                              │
│  3. Class Imbalance Handling                                │
│     ├─ SMOTE (synthetic minority oversampling)              │
│     └─ Class weights in model configuration                 │
│                                                              │
│  4. Probability Calibration                                 │
│     └─ Platt scaling on calibration set (15%)               │
│                                                              │
│  5. Threshold Tuning                                        │
│     └─ Sweep 0.01-0.99 to maximize Macro F1                │
│                                                              │
│  6. Final Model                                             │
│     └─ Refit best model on full training data               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Macro F1** | Primary metric (competition) | >0.60 |
| **PR-AUC** | Precision-Recall AUC | >0.55 |
| **MCC** | Matthews Correlation Coefficient | >0.30 |
| **Balanced Accuracy** | Average of both class recalls | >0.55 |
| **Minority F1** | F1 for "Tidak Sesuai" class | >0.30 |

## Setup Environment

Gunakan Python 3.10 atau lebih baru, lalu install dependency dari `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Jika menggunakan pipeline spaCy yang membutuhkan model bahasa, install modelnya secara terpisah:

```bash
python -m spacy download xx_sent_ud_sm
```

Resource NLTK dapat diunduh dari notebook atau script hanya jika memang dibutuhkan oleh eksperimen tersebut.

## Struktur Repository

```text
data/
├── external/                 # Data dari sumber pihak ketiga.
├── interim/                  # Data hasil transformasi sementara.
├── processed/                # Dataset final untuk pemodelan.
└── raw/penyisihan-dac-ifest-2026/ # Data kompetisi asli dan immutable.
docs/
├── COMPETITION.md            # Catatan kompetisi dan evaluasi.
├── LLM_CODING_GUIDELINES.md # Aturan bantuan coding dan tracking.
└── ARCHITECTURE.md           # Detail arsitektur & improvement plan.
notebooks/                    # Notebook eksperimen.
src/                          # Kode feature engineering dan model.
submissions/                  # Output submission per notebook.
```

## Alur Kerja Branch

1. Buat satu branch kerja untuk setiap anggota, misalnya `member/andi` atau `andi`.
2. Gunakan `experiment_id` untuk membedakan eksperimen di dalam branch anggota tersebut; tidak perlu membuat branch baru untuk setiap eksperimen.
3. Simpan catatan eksperimen dan output sementara di lokasi yang sesuai, lalu simpan notebook di `notebooks/` dan submission final di `submissions/<nama-notebook>/`.
4. Sebelum merge, jalankan notebook dari awal sampai akhir dan perbarui tracking JSON untuk semua output yang dibuat.

Script `src/hash_outputs.py` dapat digunakan untuk menghitung SHA-256 beberapa file output sekaligus.

Detail aturan reproducibility dan tracking ada di [LLM_CODING_GUIDELINES.md](docs/LLM_CODING_GUIDELINES.md).

## File Utama

| File | Deskripsi |
|------|-----------|
| `opencode_output/01_full_code.py` | Pipeline utama (~2000 lines, 34 cells) |
| `opencode_output/01_full_v1.md` | TODO plan & workflow |
| `opencode_output/01_baseline.md` | Baseline experiments |

## Performance History

| Version | Macro F1 (CV) | Macro F1 (Kaggle) | Notes |
|---------|---------------|-------------------|-------|
| v0 (baseline) | ~0.52 | 0.52 | Dummy baseline |
| v1 (current) | ~0.57 | 0.54 | 36 features, 4 models |
| v2 (planned) | >0.60 | >0.60 | +8 features, SMOTE, Bayesian tuning |

## Upcoming Improvements (v2)

### Feature Engineering
- [ ] **Hybrid TF-IDF**: char_wb (3-5gram) + word-level (1-2gram) cosine similarity
- [ ] **N-gram Overlap**: unigram, bigram, trigram overlap metrics
- [ ] **Feature Matrix EDA**: distribusi, korelasi, feature importance analysis

### Sampling & Class Imbalance
- [ ] **SMOTE**: Synthetic Minority Oversampling (ratio 0.5)
- [ ] **Remove noise detection**: IsolationForest harmful untuk minority class

### Hyperparameter Tuning
- [ ] **Bayesian Optimization**: Optuna/Scikit-Optimize (bukan grid search)
- [ ] **Threshold tuning**: On calibration set, bukan full OOF

### Model
- [ ] **Stacking ensemble**: LogReg + XGBoost + LightGBM → meta-learner

## Notes
- **Pretrained models TIDAK DIBOLEHKAN** dalam kompetisi ini
- Semua approach harus traditional ML (features + classifiers)

## License

MIT