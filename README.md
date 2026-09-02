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
└── LLM_CODING_GUIDELINES.md # Aturan bantuan coding dan tracking.
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