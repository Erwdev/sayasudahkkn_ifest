# LLM Coding Guidelines

Panduan ini digunakan saat meminta bantuan LLM untuk membuat atau mengubah kode, notebook, data turunan, model, dan file submission pada repository ini.

## Reproducibility Notebook

- Pada awal notebook, definisikan konstanta reproducibility seperti `RANDOM_SEED = 42`.
- Gunakan seed tersebut pada library yang relevan, termasuk Python, NumPy, dan model machine learning.
- Catat versi Python, library penting, dataset input, dan parameter eksperimen jika tersedia.
- Jalankan notebook dari awal sampai akhir sebelum hasilnya digunakan atau dibagikan.
- Hindari mengandalkan state, file, atau variabel dari eksekusi notebook sebelumnya.

Contoh awal notebook:

```python
RANDOM_SEED = 42

import random
import numpy as np

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

## Identitas Output

Setiap file output yang dibuat atau diubah harus memiliki identitas yang dapat ditelusuri. Aturan ini berlaku untuk:

- model dan checkpoint;
- dataset interim atau processed;
- hasil feature engineering;
- file submission;
- evaluasi, prediksi, dan artefak eksperimen lainnya.

Untuk setiap output, sertakan atau catat:

1. `experiment_id` atau nama eksperimen;
2. `source_code` berupa nama notebook/script dan commit atau branch jika tersedia;
3. `input_files` beserta hash file jika memungkinkan;
4. `output_file` dan hash output setelah selesai dibuat;
5. `random_seed`;
6. parameter penting dan metrik evaluasi;
7. waktu pembuatan dalam UTC.

Gunakan nama file yang memuat identitas eksperimen jika sesuai, misalnya:

```text
submissions/01_baseline/01_baseline_seed42.csv
```

Jangan menimpa output eksperimen sebelumnya tanpa alasan yang jelas. Buat file baru atau gunakan identifier versi baru.

## Tracking JSON

Repository ini menggunakan tracking JSON ringan dan tidak menggunakan DVC. Simpan manifest eksperimen di lokasi yang dekat dengan output, misalnya:

```text
submissions/01_baseline/metadata.json
data/interim/metadata.json
```

Contoh minimal:

```json
{
  "experiment_id": "01_baseline_seed42",
  "source_code": "notebooks/01_baseline.ipynb",
  "branch": "member/nama-anggota",
  "random_seed": 42,
  "input_files": [
    {
      "path": "data/raw/penyisihan-dac-ifest-2026/train.csv",
      "sha256": "<sha256-input>"
    }
  ],
  "outputs": [
    {
      "path": "submissions/01_baseline/01_baseline_seed42.csv",
      "sha256": "<sha256-output>"
    }
  ],
  "metrics": {
    "macro_f1": null
  },
  "created_at_utc": "YYYY-MM-DDTHH:MM:SSZ"
}
```

Hash dapat dibuat dengan `sha256sum path/to/file` atau script repository:

```bash
python src/hash_outputs.py submissions/01_baseline/01_baseline_seed42.csv
```

Untuk beberapa output, berikan semua path dalam satu perintah. Script mencetak JSON yang dapat disalin ke manifest:

```bash
python src/hash_outputs.py \
  data/interim/features_seed42.csv \
  submissions/01_baseline/01_baseline_seed42.csv
```

Jika hash belum dapat dihitung, gunakan placeholder yang jelas dan lengkapi sebelum output dipakai sebagai hasil final.

## Branch Dan Perubahan

- Gunakan satu branch kerja per individu, misalnya `member/nama-anggota`.
- Bedakan eksperimen dalam branch tersebut dengan `experiment_id`, seed, parameter, dan nama output.
- LLM harus mempertahankan perubahan pengguna dan tidak menghapus output atau data yang tidak terkait.
- Perubahan kode harus sekecil mungkin dan tidak mengubah struktur data secara diam-diam.
- Sebelum merge, periksa status Git, jalankan notebook dari awal, dan perbarui tracking JSON.
- Jangan menambahkan DVC atau sistem versioning data lain tanpa keputusan tim.

## Checklist Sebelum Membagikan Output

- [ ] Notebook memiliki konstanta random seed di bagian awal.
- [ ] Input dan output memiliki path serta identitas eksperimen.
- [ ] Hash atau code/commit ID telah dicatat.
- [ ] Tracking JSON telah diperbarui.
- [ ] Output lama tidak tertimpa tanpa alasan.
- [ ] Notebook berhasil dijalankan dari awal sampai akhir.
- [ ] Submission memiliki kolom `id,label` dan label hanya `0` atau `1`.
