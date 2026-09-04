# 01_full_v1 — Full Pipeline: Preprocessing → Feature Engineering → Modelling

> **Format:** Todo list per cell. Deskripsi task, input/output, reuse dari baseline, dan catatan penting.
> **Ikuti:** `diagram.md` flow + `opencode_guide.md` 14 tahap.
> **Reuse:** Kode dari `01_baseline.md` sebisa mungkin, implement dari 0 hanya yang belum ada.

---

# TAHAP 1: PREPROCESSING

> **Goal:** Load data, EDA, buat `TextPreprocessor` transformer (stateless), apply ke train & test, EDA overlap/similarity, simpan processed data + config.
> **Memory target:** Habis tahap ini, yang tersisa di memory: `train` (dengan kolom clean_a/clean_b/tokens), `test` (sama), wordlists dict, pipeline config.

---

## Cell 0: Imports + Load Data + Memory Helpers

**Task:**
- Import semua library yang dibutuhkan (numpy, pandas, matplotlib, seaborn, wordcloud, nltk, sklearn, Sastrawi, spacy, joblib, gc, os, re, json, time, tqdm)
- Download resource NLTK yang diperlukan (stopwords)
- Load `train.csv` dan `test.csv` dari `data/raw/penyisihan-dac-ifest-2026/`
- Definisikan konstanta: `RANDOM_SEED = 42`, set random seed untuk Python, NumPy
- Definisikan memory helper functions:
  - `mem_usage()` → return current memory usage dalam MB
  - `cleanup()` → `gc.collect()` + print memory usage
  - `log_memory(tag)` → print tag + memory usage
- Log: `"Start memory: XX MB"`

**Input:** `data/raw/penyisihan-dac-ifest-2026/train.csv`, `data/raw/penyisihan-dac-ifest-2026/test.csv`
**Output:** DataFrame `train`, `test` di memory

**Reuse dari baseline:** `Cell 0` — tambah spacy, sklearn.pipeline, sklearn.base, gc
**Catatan:**
- Tambahkan `spacy` ke `requirements.txt` jika belum ada
- `spacy.blank("id")` hanya tokenizer, bukan model pretrained — tidak perlu download model besar
- Cek apakah `spacy` sudah terinstall, kalau belum: `pip install spacy`

---

## Cell 1: Dataset Inspection

**Task:**
- Cek shape, columns, dtypes, missing values, duplicate rows
- Auto-discover text columns (`title`, `content`) dan label column (`label`)
- Validasi kolom sesuai yang diharapkan

**Input:** `train`, `test` dari Cell 0
**Output:** Print info dataset, variabel `headline_col`, `article_col`, `label_col`

**Reuse dari baseline:** `Cell 1` — langsung reuse

---

## Cell 2: Label Distribution

**Task:**
- Hitung value_counts label (0: Tidak Sesuai, 1: Sesuai)
- Visualisasi bar chart dengan persentase
- Hitung imbalance ratio
- Catat majority baseline accuracy

**Input:** `train` dari Cell 1
**Output:** Bar chart, print imbalance ratio

**Reuse dari baseline:** `Cell 2` — langsung reuse

---

## Cell 3: Text Length Analysis

**Task:**
- Hitung char length dan word length untuk headline dan artikel
- Statistik deskriptif (mean, median, min, max, std)
- Histogram distribusi panjang teks
- Boxplot per label
- Outlier analysis (IQR method) — analisis SAJA, jangan hapus outlier

**Input:** `train` dari Cell 2
**Output:** Kolom `headline_char_len`, `headline_word_len`, `article_char_len`, `article_word_len` di train. Visualisasi.

**Reuse dari baseline:** `Cell 3` — langsung reuse

---

## Cell 4: Vocabulary Analysis

**Task:**
- Top unigrams & bigrams (global dan per label) menggunakan `CountVectorizer`
- Analisis kata yang paling membedakan antara label 0 dan 1

**Input:** `train` dari Cell 3
**Output:** Print top ngrams per label

**Reuse dari baseline:** `Cell 4` — langsung reuse

---

## Cell 5: WordCloud

**Task:**
- Buat WordCloud untuk: semua headline, semua artikel, per label (Sesuai/Tidak Sesuai) × headline/article
- Total 6 visualisasi

**Input:** `train` dari Cell 4
**Output:** 6 WordCloud plots

**Reuse dari baseline:** `Cell 5` — langsung reuse

---

## Cell 6: Load Wordlists + InSet Lexicon

**Task:**
- Load `data/external/wordlists.json` → dict berisi: action_verbs, superlatives, discourse_markers, negation_words, buzzwords, month_names
- Load InSet lexicon dari `E:\GRINDING\DATA SCIENCE\IFEST_2026\inset_lexicon\InSet\positive.tsv` dan `negative.tsv` (format TSV: `word\tweight`)
- Gabungkan menjadi satu dict `sentiment_lexicon = {"kata": skor}` (positive: +1 s.d. +5, negative: -1 s.d. -5)
- Simpan semua wordlists ke variabel global yang bisa diakses oleh feature transformer classes nanti

**Input:** `data/external/wordlists.json`, `positive.tsv`, `negative.tsv`
**Output:** Dict `WORDLISTS` berisi semua wordlists, dict `SENTIMENT_LEXICON`

**Reuse:** Tidak ada — data sudah disediakan, tinggal load
**Catatan:**
- InSet Lexicon: 3609 positive + 6609 negative words, weight -5 s.d. +5
- Beberapa kata mungkin mengandung spasi (misal "putus tali gantung") — pertahankan apa adanya
- Wordlists JSON sudah lengkap untuk semua grup fitur

---

## Cell 7: `TextPreprocessor` Class

**Task:**
- Buat class `TextPreprocessor(BaseEstimator, TransformerMixin)` yang bisa di-pass ke sklearn Pipeline
- Method `.fit(X, y=None)` → no-op (stateless), return self
- Method `.transform(X)` → terima DataFrame dengan kolom `[headline, artikel]`, kembalikan DataFrame dengan 6 kolom turunan:
  - `headline_clean_a` — light cleaning (whitespace, URL, HTML, noise)
  - `headline_clean_b` — normalized (clean_a + lowercase + stopword removal)
  - `headline_tokens` — list token dari clean_b (untuk fitur relasi)
  - `artikel_clean_a` — light cleaning
  - `artikel_clean_b` — normalized
  - `artikel_tokens` — list token dari clean_b

**Step preprocessing (urutan WAJIB):**
1. `normalize_whitespace(text)` — collapse multiple space/newline jadi single space
2. `remove_url(text)` — regex `https?://\S+|www\.\S+`
3. `remove_html(text)` — regex `<[^>]+>`
4. `remove_noise(text)` — remove email, karakter non-alfanumerik berlebih (TAPI pertahankan tanda baca: `.,;:!?%$()-/+` untuk fitur stylistik di Tahap 2)
5. → **clean_a DONE** (untuk fitur stylistik)
6. `lowercase(text)`
7. `stopword_removal(text)` — pakai Sastrawi StopWordRemoverFactory
8. Tokenisasi pakai `spacy.blank("id")` → hasilkan list token (text per token, bukan objek Doc)
9. → **clean_b + tokens DONE** (untuk fitur lexical/semantic)

**Input:** DataFrame dengan kolom `[title, content]` (atau `headline, artikel`)
**Output:** DataFrame dengan 6 kolom turunan

**Reuse dari baseline:** Fungsi `normalize_whitespace`, `remove_url`, `remove_html`, `remove_noise`, `lowercase` dari `Cell 6` baseline — bungkus ulang jadi method di class
**Catatan:**
- SpaCy blank "id" TIDAK punya POS tagger atau dependency parser native untuk Bahasa Indonesia
- Tokenisasi pakai `nlp = spacy.blank("id")` lalu `doc = nlp(text)` → `[token.text for token in doc]`
- Fitur POS-ratio dan dependency path di Tahap 2 pakai **heuristik regex + wordlist**, BUKAN true parsing
- Tulis komentar jujur di kode soal keterbatasan ini

---

## Cell 8: Apply TextPreprocessor ke Train + Test

**Task:**
- Inisialisasi `TextPreprocessor`
- `fit_transform` di train → dapat 6 kolom baru
- `transform` di test → 6 kolom baru (TRANSFORM ONLY, tidak fit ulang)
- Log: jumlah baris, contoh output, memory usage
- **cleanup()** setelah selesai

**Input:** `train`, `test` dari Cell 1-5
**Output:** `train` dan `test` dengan 6 kolom turunan tambahan

**Catatan:**
- Pastikan test menggunakan object yang SUDAH fit dari train
- Jangan pernah fit ulang apapun di test

---

## Cell 9: Save Processed Data

**Task:**
- Simpan hasil preprocessing ke `data/interim/`:
  - `train_clean_a.csv` — headline_clean_a, artikel_clean_a
  - `train_clean_b.csv` — headline_clean_b, artikel_clean_b
  - `train_tokens.csv` — headline_tokens, artikel_tokens
  - `processing_metadata.json` — info proses (tanggal, jumlah baris, steps, dst)
- Simpan juga raw sebagai referensi

**Input:** `train` dari Cell 8
**Output:** File CSV + JSON di `data/interim/`

**Reuse dari baseline:** `Cell 6b` — adaptasi ke struktur kolom baru (6 kolom turunan)

---

## Cell 10: Lexical Overlap EDA

**Task:**
- Hitung Jaccard, overlap_coefficient, dice untuk clean_a dan clean_b
- Perbandingan distribusi per label (histogram)
- Separability score per variant × metric
- Rekomendasi variant terbaik berdasarkan separation gap

**Input:** `train` dari Cell 8 (kolom clean_a, clean_b)
**Output:** Metrics columns (jaccard_overlap_a/b, overlap_coeff_a/b, dice_a/b), visualisasi, separability table

**Reuse dari baseline:** `Cell 7` — adaptasi (hanya 2 variant: clean_a + clean_b, drop raw)

---

## Cell 11: LSA Similarity EDA

**Task:**
- Hitung LSA cosine similarity dari clean_b (TF-IDF + SVD)
- Simpan fitted objects: `tfidf_lsa_vectorizer.joblib`, `svd_lsa_model.joblib`
- Visualisasi distribusi per label + separability score
- **Hapus** `tfidf_matrix`, `lsa_matrix` setelah cosine sim tersimpan → cleanup()

**Input:** `train` dari Cell 8 (kolom clean_b)
**Output:** `feat_lsa_cosine_sim` column, saved joblib files, visualisasi

**Reuse dari baseline:** `Cell 8` — adaptasi ke clean_b pipeline baru
**Memory:** `tfidf_matrix` (~sparse) + `lsa_matrix` (~dense) bisa besar → hapus setelah simpan cosine sim

---

## Cell 12: Comparative Summary

**Task:**
- Gabungkan semua separation gap dari Cell 10-11
- Sort by gap descending
- Top 5 most discriminative features
- Best variant per source
- Rekomendasi untuk feature selection

**Input:** Separability tables dari Cell 10-11
**Output:** Combined summary table, print rekomendasi

**Reuse dari baseline:** `Cell 9` — langsung reuse

---

## Cell 13: Save Pipeline Config JSON

**Task:**
- Simpan pipeline config v1 ke `data/processing/pipeline_config_v1.json`
- Berisi: dataset info, preprocessing steps, variants, eda_comparison_summary, libraries_used, limitations (spacy blank)

**Input:** Semua info dari Cell 0-12
**Output:** `data/processing/pipeline_config_v1.json`

**Reuse dari baseline:** `Cell 10` — adaptasi

---

### CHECKPOINT TAHAP 1

**Memory check:** Log memory usage. Yang tersisa di memory:
- `train` DataFrame (dengan kolom clean_a/clean_b/tokens + text length columns)
- `test` DataFrame (sama)
- `WORDLISTS` dict (~200KB)
- `SENTIMENT_LEXICON` dict (~500KB)
- Pipeline config dict
- Variabel EDA (feature_cols, separability tables)

**Bersihkan:** Hapus variabel EDA yang tidak dipakai lagi (ngrams analysis, wordcloud objects, dll)

---

# TAHAP 2: FEATURE ENGINEERING (6 Grup + Noise Detection + Dummy Baseline)

> **Goal:** Implementasi 6 grup fitur dari opencode_guide sebagai custom Transformer classes, assembly via FeatureUnion, noise detection, dummy baseline, dan final feature matrix.
> **Memory target:** Habis tahap ini, yang tersisa: `feature_matrix` (hanya id + label + feature_cols), `feature_names`, `sample_weight`, wordlists dict.

---

## Cell 14: A. LexicalFeatures Class

**Task:**
- Buat class `LexicalFeatures(BaseEstimator, TransformerMixin)`
- `.fit(X, y=None)` → no-op (stateless)
- `.transform(X)` → terima DataFrame dengan kolom `[headline_clean_b, artikel_clean_b]`, kembalikan DataFrame dengan 3 kolom:
  - `feat_jaccard` — Jaccard similarity antara headline_clean_b tokens dan artikel_clean_b tokens
  - `feat_overlap_coeff` — overlap coefficient
  - `feat_dice` — Dice coefficient (2 * |intersection| / (|H| + |A|))

**Implementasi:**
- Reuse fungsi `compute_lexical_metrics` dari `Cell 11a` baseline
- Tokenisasi: split clean_b text by whitespace (sudah di-lowercase dan stopword removed oleh TextPreprocessor)
- Handle edge case: empty text → 0

**Input:** DataFrame dengan kolom `[headline_clean_b, artikel_clean_b]`
**Output:** DataFrame 3 kolom numerik

**Reuse dari baseline:** `Cell 11a` — ambil fungsi `compute_lexical_metrics`, bungkus jadi TransformerMixin

---

## Cell 15: B. EntityNumericFeatures Class

**Task:**
- Buat class `EntityNumericFeatures(BaseEstimator, TransformerMixin)`
- `.transform(X)` → DataFrame dengan 5 kolom:
  - `feat_proxy_ner_intersection` — rasio kata berkapital (bukan di awal kalimat) yang muncul di headline JUGA di artikel, dibanding total kata kapital di headline
  - `feat_numeric_mismatch_count` — jumlah angka di headline yang TIDAK ADA (atau tidak cocok dalam toleransi) di artikel
  - `feat_numeric_total` — total angka di headline + artikel (untuk normalisasi)
  - `feat_date_mismatch` — 1 jika ada tanggal/bulan yang berbeda antara headline dan artikel, 0 jika sama
  - `feat_entity_ratio` — same as proxy_ner_intersection (digunakan untuk konsistensi)

**Implementasi:**
- Proxy NER: regex `[A-Z][a-z]+` pada headline_clean_a (raw-ish, pertahankan kapitalisasi), bukan di clean_b (sudah lowercase). Cek intersection dengan artikel_clean_a
- Regex angka: `\d+[.,]?\d*` — ekstrak semua angka, hitung mismatch (angka di headline yang tidak ada di artikel, dengan toleransi fuzzy matching untuk angka desimal)
- Regex tanggal/bulan: pakai `WORDLISTS['month_names']` + pattern DD/MM/YYYY, DD-MM-YYYY, dst
- Hitung failure rate (berapa % baris yang gagal ekstraksi per fitur) → simpan untuk logging

**Input:** DataFrame dengan kolom `[headline_clean_a, artikel_clean_a, headline_clean_b, artikel_clean_b]` + wordlists
**Output:** DataFrame 5 kolom numerik

**Reuse dari baseline:** Tidak ada — implement dari 0
**Catatan:**
- Proxy NER bukan true NLP NER, hanya heuristik kapitalisasi
- Regex angka harus handle format ID: `1.000` (seribu), `1,5` (satu koma lima), `Rp 50.000`

---

## Cell 16: C. PositionFeatures Class

**Task:**
- Buat class `PositionFeatures(BaseEstimator, TransformerMixin)`
- `.transform(X)` → DataFrame dengan 2 kolom:
  - `feat_lead_paragraph_match` — rasio kemunculan kata kunci headline (non-stopword) di 3 kalimat pertama artikel dibanding kemunculan di keseluruhan artikel. Score tinggi = headline banyak disebut di awal artikel (indikasi sesuai)
  - `feat_headline_coverage_ratio` — jumlah unique headline tokens yang muncul di artikel / total headline tokens

**Implementasi:**
- Split artikel_clean_b jadi kalimat (split by `.`, `!`, `?`, atau pattern kalimat sederhana)
- Ambil 3 kalimat pertama
- Hitung kata kunci headline (non-stopword) yang muncul di awal vs keseluruhan
- Rasio = early_count / total_count (handle division by zero → 0)

**Input:** DataFrame dengan kolom `[headline_clean_b, artikel_clean_b]` + headline_tokens, artikel_tokens
**Output:** DataFrame 2 kolom numerik

**Reuse dari baseline:** Tidak ada — implement dari 0
**Catatan:**
- Sentence splitting pakai regex sederhana, BUKAN spacy sentence splitter (karena spacy.blank tidak punya sentence boundary detection)

---

## Cell 17: D. LinguisticFeatures Class

**Task:**
- Buat class `LinguisticFeatures(BaseEstimator, TransformerMixin)`
- `.transform(X)` → DataFrame dengan 5 kolom:
  - `feat_sentiment_gap` — `|sentiment(headline) - sentiment(artikel)|` pakai InSet lexicon. Skor headline = rata-rata weight semua kata yang ada di lexicon. Skor artikel = sama. Gap = absolute difference.
  - `feat_superlative_ratio` — jumlah kata superlatif di headline / total kata headline (pakai `WORDLISTS['superlatives']`)
  - `feat_discourse_marker_count` — jumlah kemunculan discourse markers di artikel yang berdekatan (window ±1 kalimat) dengan kalimat yang overlap dengan headline
  - `feat_negation_contradiction` — 1 jika ada kata negasi di artikel yang dinegasikan objeknya (token setelah negasi dalam window 3 token) dan objek tersebut juga muncul di headline TANPA negasi, 0 jika tidak
  - `feat_readability_gap` — `|avg_sentence_length(headline) - avg_sentence_length(artikel)|` + rasio panjang kata rata-rata

**Implementasi:**
- Sentiment: load `SENTIMENT_LEXICON` dari Cell 6, split text jadi tokens, cari di lexicon, hitung rata-rata weight
- Superlative: check setiap n-gram headline terhadap `WORDLISTS['superlatives']` (beberapa superlative ada spasi, misal "paling besar")
- Discourse markers: split artikel jadi kalimat, cari overlap dengan headline, cek apakah discourse marker muncul di kalimat sekitar overlap
- Negation: scan artikel tokens untuk negation words (`WORDLISTS['negation_words']`), ambil objek (3 token setelah negasi), cek apakah objek muncul di headline tanpa negasi
- Readability: split jadi kalimat, hitung rata-rata panjang kalimat dan rata-rata panjang kata

**Input:** DataFrame dengan kolom `[headline_clean_a, artikel_clean_a, headline_clean_b, artikel_clean_b, headline_tokens, artikel_tokens]` + wordlists + sentiment_lexicon
**Output:** DataFrame 5 kolom numerik

**Reuse dari baseline:** Tidak ada — implement dari 0
**Catatan:**
- Sentiment lexicon InSet pakai word-level matching, jadi perlu tokenisasi dulu
- Beberapa kata di InSet ada spasi (misal "putus tali gantung") — pertahankan, bisa match multi-token
- Failure rate per fitur harus dicatat (berapa % baris yang gagal)

---

## Cell 18: E. RelationFeatures Class

**Task:**
- Buat class `RelationFeatures(BaseEstimator, TransformerMixin)`
- `.transform(X)` → DataFrame dengan 3 kolom:
  - `feat_event_extraction` — rasio verba aksi headline yang co-occur di artikel (pakai `WORDLISTS['action_verbs']`). Hitung verba aksi di headline, cek apakah verba yang sama (atau sinonim sederhana) muncul di artikel
  - `feat_svo_overlap` — heuristic SVO (Subjek-Verb-Objek) overlap antara headline dan artikel. Karena spacy.blank tidak punya dependency parser, pakai urutan token + wordlist: subjek = kata sebelum verba aksi, objek = kata sesudah verba aksi. Bandingkan SVO headline vs artikel.
  - `feat_relation_score` — gabungan dari event_extraction dan svo_overlap (rata-rata atau weighted)

**Implementasi:**
- Event extraction: scan headline_tokens untuk action_verbs, untuk setiap verba yang ditemukan, cek apakah verba yang sama muncul di artikel_tokens (atau 1-2 token sebelum/sesudah verba di artikel — context window)
- Heuristic SVO: untuk setiap action_verb di headline, ambil 2 token sebelum (subjek) dan 2 token sesudah (objek). Lakukan hal yang sama di artikel. Hitung overlap
- **CATAT DI KODE:** `heuristic_svo_extraction()` BUKAN `dependency_parse()` — ini heuristik, bukan true NLP parsing. spacy.blank("id") tidak punya dependency parser native untuk Bahasa Indonesia.

**Input:** DataFrame dengan kolom `[headline_tokens, artikel_tokens]` + `WORDLISTS['action_verbs']`
**Output:** DataFrame 3 kolom numerik

**Reuse dari baseline:** Tidak ada — implement dari 0
**Catatan:**
- `WORDLISTS['action_verbs']` sudah ada 242 kata — cukup komprehensif untuk berita Indonesia
- Failure rate untuk event_extraction dan svo_overlap harus dicatat terpisah

---

## Cell 19: F. StylisticFeatures Class

**Task:**
- Buat class `StylisticFeatures(BaseEstimator, TransformerMixin)`
- `.transform(X)` → DataFrame dengan 3 kolom:
  - `feat_exclamation_count` — jumlah tanda seru `!` di headline_clean_a (versi light, pertahankan tanda baca)
  - `feat_caps_ratio` — rasio huruf kapital berlebihan (ALL CAPS words / total words) di headline_clean_a
  - `feat_buzzword_count` — jumlah kemunculan buzzwords (`WORDLISTS['buzzwords']`) di headline_clean_a (case-insensitive)

**Implementasi:**
- Exclamation: `headline_clean_a.count('!')`
- Caps ratio: split headline_clean_a jadi words, hitung yang ALL CAPS (len >= 3), bagi total words
- Buzzwords: scan headline_clean_a (lowercase) untuk setiap buzzword (beberapa buzzword ada spasi, misal "bikin geger")
- Pakai `clean_a` (bukan clean_b) karena butuh tanda baca dan kapitalisasi asli

**Input:** DataFrame dengan kolom `[headline_clean_a]` + `WORDLISTS['buzzwords']`
**Output:** DataFrame 3 kolom numerik

**Reuse dari baseline:** Tidak ada — implement dari 0

---

## Cell 20: FeatureUnion Assembly → `feature_pipeline`

**Task:**
- Gabungkan semua 6 grup transformer (A-F) via `FeatureUnion`:
  ```python
  feature_pipeline = FeatureUnion([
      ('lexical', LexicalFeatures()),
      ('entity_numeric', EntityNumericFeatures()),
      ('position', PositionFeatures()),
      ('linguistic', LinguisticFeatures()),
      ('relation', RelationFeatures()),
      ('stylistic', StylisticFeatures()),
  ])
  ```
- `.fit_transform(train)` → feature matrix X
- Simpan `feature_names` (list nama kolom fitur final, urut) ke JSON
- Log: jumlah fitur total, vocabulary size TF-IDF, extraction failure rates

**Input:** `train` DataFrame dengan semua kolom yang diperlukan + wordlists
**Output:** `X` (numpy array atau DataFrame), `feature_names` list

**Reuse dari baseline:** Tidak ada — ini struktur baru sesuai opencode_guide
**Catatan:**
- FeatureUnion otomatis handle concatenation
- Setiap transformer harus return DataFrame (bukan numpy array) supaya feature names bisa di-track
- Alternatif: bisa pakai `ColumnTransformer` kalau beberapa fitur butuh pipeline preprocessing dulu, tapi karena TextPreprocessor sudah di-cell sebelumnya, FeatureUnion sudah cukup

---

## Cell 21: Dummy Baseline

**Task:**
- `DummyClassifier(strategy='most_frequent')`, fit di train
- Evaluasi dengan metrik yang SAMA seperti Tahap 3 nanti: PR-AUC, MCC, Balanced Accuracy, Macro F1
- Simpan angka ini sebagai patokan minimum di JSON
- SEMUA model wajib dibandingkan ke sini

**Input:** X (feature matrix), y (label)
**Output:** Dummy metrics dict, disimpan ke JSON

**Reuse dari baseline:** `Cell 13` baseline (majority baseline) — tapi tambah PR-AUC dan MCC

---

## Cell 22: Noise Detection

**Task:**
- **Isolation Forest** (unsupervised): fit di X (feature matrix), tanpa lihat label. Deteksi baris outlier fitur. Simpan `outlier_score` per baris. `contamination=0.1` (default, bisa tune)
- **Confident Learning** via Out-of-Fold predictions:
  - Latih model dasar (misal RandomForest sederhana) dengan 3-fold CV
  - Kumpulkan OOF predicted probability
  - Tandai baris dengan predicted label confidence tinggi (>0.8) TAPI beda dari label asli → kandidat mislabel
- **Assign `sample_weight`:**
  - Baris "bersih" = 1.0
  - Baris "noise" (Isolation Forest outlier DAN confident-learning flag) = 0.3
  - Threshold "noise": kedua metode harus flag (AND logic, bukan OR) → lebih konservatif
- Simpan ke JSON: jumlah baris noise, threshold, contoh index baris noise
- **cleanup()** setelah selesai — hapus Isolation Forest model, OOF predictions

**Input:** X (feature matrix), y (label)
**Output:** `sample_weight` array (sama panjang dengan train), noise_detection info dict

**Reuse dari baseline:** Tidak ada — implement dari 0
**Catatan:**
- Isolation Forest bisa pakai `sklearn.ensemble.IsolationForest`
- Confident Learning: kalau `cleanlab` tersedia, pakai. Kalau tidak, implement manual (sudah dijelaskan di opencode_guide)
- Catatan di kode: kriteria pasti "noise" = Isolation Forest outlier DAN confident learning flag

---

## Cell 23: Feature Matrix Assembly + Separability

**Task:**
- Gabungkan fitur dari 6 grup (Cell 20) dengan fitur baseline yang mungkin masih relevan (LSA cosine sim dari Cell 11, TF-IDF cosine dari baseline jika dihitung ulang)
- Final feature columns = gabungan fitur grup A-F + fitur baseline yang lolos seleksi
- Separability analysis: hitung gap per fitur, sort by gap
- **HAPUS** semua kolom clean_a/clean_b/tokens dari DataFrame → simpan hanya `id`, `label`, `feature_cols`
- Log final feature count

**Input:** Feature matrix dari Cell 20, fitur baseline dari Cell 11
**Output:** Final `feature_matrix` DataFrame (compact), `feature_cols` list

**Reuse dari baseline:** `Cell 12` — adaptasi
**Memory:** Penting! Hapus kolom teks yang besar. DataFrame harusnya jauh lebih kecil setelah ini.

---

## Cell 24: Save Feature Matrix + Config v3

**Task:**
- Simpan `feature_matrix` ke `data/interim/feature_matrix.csv`
- Simpan `data/processing/pipeline_config_v3.json` berisi:
  - Semua `feature_names`
  - `n_features`
  - `extraction_failure_rate` per grup fitur
  - `limitations` (spacy blank, heuristik SVO, dll)
  - `noise_detection` info
  - `sample_weight` distribution

**Input:** Feature matrix, wordlists info, noise_detection info
**Output:** `data/interim/feature_matrix.csv`, `data/processing/pipeline_config_v3.json`

---

### CHECKPOINT TAHAP 2

**Memory check:** Log memory usage. Yang tersisa di memory:
- `feature_matrix` DataFrame (compact: id + label + feature_cols) — HARUS lebih kecil dari sebelumnya
- `feature_names` list
- `sample_weight` array
- `WORDLISTS` dict (~200KB)
- `SENTIMENT_LEXICON` dict (~500KB)

**Bersihkan:** Hapus `train` DataFrame yang besar (yang punya kolom clean_a/clean_b/tokens). Yang diperlukan untuk Tahap 3 hanya feature_matrix + label + sample_weight.

---

# TAHAP 3: MODELLING

> **Goal:** Nested CV, model training, OOF predictions, calibration, evaluation, threshold tuning, fit final model, predict test, submission.
> **Memory target:** Habis tahap ini, yang tersisa: final model, calibrator, threshold, submission.csv, evaluation results.

---

## Cell 25: Nested Stratified K-Fold CV Setup

**Task:**
- Definisikan outer CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — untuk EVALUASI FINAL
- Definisikan inner split: 85/15 `train_test_split` (stratified) di dalam tiap outer train fold — untuk early stopping n_estimators
- Split calibration set di AWAL: 15% dari train dengan stratified split, simpan index-nya. Calibration set ini TIDAK boleh overlap dengan CV manapun.
- Log: jumlah fold, ukuran calibration set

**Input:** feature_matrix (X, y), sample_weight
**Output:** cv_outer, calibration_index

**Catatan:**
- Calibration set dipisah SEBELUM CV dimulai (opsi dari opencode_guide Tahap 8)
- X dan y diambil dari feature_matrix

---

## Cell 26: Model Training per Fold

**Task:**
- Untuk tiap outer fold:
  1. Split train/test sesuai outer fold
  2. **Fit feature_pipeline ulang** di outer train fold (JANGAN pakai fit dari fold lain — ini leakage)
  3. Split lagi inner (85/15) untuk early stopping
  4. Train XGBoost atau LightGBM dengan early_stopping pakai inner split
  5. Simpan: model object, `best_iteration`, OOF raw probability untuk baris di outer test fold
  6. `scale_pos_weight = 1.0` (imbalance TIDAK ditangani di sini)
- Model: XGBoost (`XGBClassifier`) sebagai primary. Buat kode modular supaya gampang switch ke LightGBM.
- Simpan model per fold (untuk audit)

**Input:** feature_matrix, sample_weight, cv_outer, calibration_index
**Output:** List model per fold, best_iteration per fold, OOF probability array

**Reuse dari baseline:** `Cell 14` — adaptasi: ganti train/test split jadi nested CV, tambah early stopping, hapus class_weight/scale_pos_weight
**Catatan:**
- Feature pipeline di-fit ULANG di setiap outer train fold → ini mencegah leakage
- Kalau pakai XGBoost: `early_stopping_rounds=50`, `eval_metric='logloss'`

---

## Cell 27: OOF Predictions Collection

**Task:**
- Kumpulkan raw probability (belum dikalibrasi, belum threshold) dari tiap outer fold
- Hasil: satu array OOF proba untuk SELURUH train set (tiap baris diprediksi oleh model yang TIDAK dilatih dengan baris itu)
- Hitung `best_iteration` median dari semua fold (untuk model final)

**Input:** OOF predictions dari Cell 26
**Output:** `oof_proba` array, `median_best_iteration`

---

## Cell 28: Probability Calibration

**Task:**
- Pakai calibration set yang sudah dipisah di Cell 25
- Platt scaling: `sklearn.calibration.CalibratedClassifierCV(method='sigmoid')`
- Fit di calibration set, transform proba dari OOF predictions (untuk baris di luar calibration set)
- Bisa juga isotonic: `method='isotonic'` — bandingkan mana yang lebih baik

**Input:** OOF proba, calibration_index
**Output:** Calibrated OOF proba, calibrator object

**Catatan:**
- Brier score SEBELUM dan SESUDAH kalibrasi → simpan ke JSON
- Kalibrasi harus di-verify dengan reliability diagram (Cell 29)

---

## Cell 29: Reliability Diagram + Brier Score

**Task:**
- Plot mean predicted probability (dibagi 10 bins) vs fraction of positives aktual di tiap bin (`sklearn.calibration.calibration_curve`)
- Hitung Brier score sebelum dan sesudah kalibrasi
- Simpan plot sebagai gambar/artifact
- Simpan angka Brier score ke JSON

**Input:** OOF proba (before & after calibration), y labels
**Output:** Reliability diagram plot, Brier scores

---

## Cell 30: Evaluation Multi-Metrik

**Task:**
- Hitung SEMUA metrik pada OOF (setelah kalibrasi), bandingkan ke Dummy Baseline (Cell 21):
  - **PR-AUC** (`average_precision_score`) — metrik utama untuk imbalance
  - **MCC** (`matthews_corrcoef`) — paling robust untuk imbalance
  - **Balanced Accuracy** — rata-rata recall per kelas
  - **Macro F1** (`f1_score(average='macro')`) — **METRIK UTAMA kompetisi**
- JANGAN laporkan accuracy atau ROC-AUC sebagai metrik utama (imbalance membuat keduanya menyesatkan)
- Tabel perbandingan: Dummy Baseline vs Model, per metrik

**Input:** Calibrated OOF proba, y labels, dummy metrics
**Output:** Evaluation table, print perbandingan

---

## Cell 31: Threshold Tuning

**Task:**
- Sweep threshold 0.01 s.d. 0.99 (step 0.01) pada calibrated OOF proba
- Optimalkan **Macro F1** (bukan accuracy)
- Simpan kurva threshold vs Macro F1 (plot)
- `best_threshold` = threshold dengan Macro F1 tertinggi
- Threshold ini yang menangani class imbalance (BUKAN scale_pos_weight di training)

**Input:** Calibrated OOF proba, y labels
**Output:** `best_threshold`, threshold vs Macro F1 curve plot, threshold info dict

---

## Cell 32: Fit Full Model

**Task:**
- Fit ulang `feature_pipeline` di SELURUH train (bukan per fold lagi)
- Fit model final dengan `n_estimators = median(best_iteration)` dari Cell 27
  - **Kenapa median, bukan mean?** Lebih robust ke outlier fold. Kalau satu fold punya best_iteration=1000 sementara lainnya ~200, mean akan terpengaruh, median tidak.
- Fit ulang calibrator juga di seluruh train (atau tetap pakai calibration set yang sama → pilih yang lebih stabil)
- Simpan: final model, calibrator, best_threshold, feature_pipeline

**Input:** feature_matrix, sample_weight, median_best_iteration, calibration set
**Output:** Saved final objects (model, calibrator, feature_pipeline, threshold)

---

## Cell 33: Test Set Predict → Submission

**Task:**
- Load test.csv
- Preprocessing: `feature_pipeline.transform(test_df)` — **TRANSFORM ONLY**, objek sudah di-fit dari Cell 32
- Predict proba → apply calibrator → apply `best_threshold`
- **JANGAN fit ulang apapun di sini** — ini poin paling kritis untuk mencegah leakage
- Simpan `submission.csv` (format: `id,label`)
- Log: predicted label distribution, waktu eksekusi

**Input:** `test.csv`, saved objects dari Cell 32
**Output:** `submissions/01_full_v1/submission.csv`

**Reuse dari baseline:** `Cell 17` — adaptasi: pakai pipeline object yang sudah fit, bukan re-fit

---

## Cell 34: Final Report + Diagrams

**Task:**
- **Reliability diagram** (dari Cell 29, finalize)
- **Threshold vs Macro F1 curve** (dari Cell 31, finalize)
- **Confusion matrix** final (OOF, pakai best_threshold)
- **Feature importance** (top-20) dari model final
- **Tabel perbandingan:** skor lama (0.52) vs skor baru per metrik (PR-AUC, MCC, BalAcc, MacroF1)
- **run_log.json** final — semua processing info sesuai skema opencode_guide

**Input:** Semua results dari Cell 25-33
**Output:** Diagram plots, final comparison table, `run_log.json`

---

### CHECKPOINT TAHAP 3 (FINAL)

**Memory check:** Final cleanup. Hapus semua variabel intermediate.
**Yang tersisa:**
- `submissions/01_full_v1/submission.csv`
- `data/processing/run_log.json`
- Diagram plots (disimpan sebagai gambar)
- Final model objects (joblib)

---

# MEMORY MANAGEMENT SUMMARY

| Checkpoint | Action | Est. Memory Saved |
|------------|--------|-------------------|
| Setelah Cell 8 (preprocessing selesai) | `cleanup()`, hapus raw text variables | ~50-100MB |
| Setelah Cell 11 (LSA selesai) | Hapus `tfidf_matrix`, `lsa_matrix` | ~100-200MB |
| Setelah Cell 22 (noise detection selesai) | Hapus Isolation Forest model, OOF predictions | ~50MB |
| Setelah Cell 23 (feature matrix final) | HAPUS semua kolom clean_a/clean_b/tokens dari DataFrame | **~200-500MB** (paling besar) |
| Setelah Cell 26 (training selesai) | Hapus per-fold model objects | ~100MB |
| Setelah Cell 33 (submission tersimpan) | Hapus test DataFrame | ~50MB |
| Cell 34 (final cleanup) | Hapus semua yang tidak perlu | Sisa minimum |

**Helper function pattern:**
```python
def cleanup():
    import gc
    gc.collect()
    # print memory usage

def log_memory(tag):
    import psutil
    process = psutil.Process()
    print(f"[{tag}] Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

---

# DEPENDENCY TAMBAHAN

Tambahkan ke `requirements.txt`:
```
spacy
psutil  # untuk memory monitoring (opsional)
```

Install:
```bash
pip install spacy psutil
python -m spacy download xx_sent_ud_sm
```

---

# INPUT DATA SUMMARY

| File | Path | Format | Catatan |
|------|------|--------|---------|
| Train | `data/raw/penyisihan-dac-ifest-2026/train.csv` | CSV (id, title, content, label) | Label: 0=Tidak Sesuai, 1=Sesuai |
| Test | `data/raw/penyisihan-dac-ifest-2026/test.csv` | CSV (id, title, content) | Tanpa label |
| Wordlists | `data/external/wordlists.json` | JSON | action_verbs(242), superlatives(55), discourse_markers(21), negation_words(12), buzzwords(33), month_names(12) |
| Sentiment (+) | `E:\GRINDING\DATA SCIENCE\IFEST_2026\inset_lexicon\InSet\positive.tsv` | TSV (word, weight) | 3609 kata, weight +1 s.d. +5 |
| Sentiment (-) | `E:\GRINDING\DATA SCIENCE\IFEST_2026\inset_lexicon\InSet\negative.tsv` | TSV (word, weight) | 6609 kata, weight -1 s.d. -5 |
