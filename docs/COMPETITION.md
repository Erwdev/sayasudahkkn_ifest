# Penyisihan DAC IFEST 2026

Sumber: https://www.kaggle.com/competitions/penyisihan-dac-ifest-2026/overview

## Ringkasan

Peserta membangun model klasifikasi biner untuk menilai kesesuaian judul berita dengan isi berita. Label `1` berarti informasi utama pada judul didukung dan dibahas dalam isi (**Sesuai**), sedangkan label `0` berarti informasi tidak didukung, membahas peristiwa berbeda, atau memiliki konteks yang mengubah makna (**Tidak Sesuai**).

Model diharapkan memahami hubungan kontekstual, bukan hanya menghitung kata yang sama. Perbedaan tokoh, lokasi, waktu, angka, tindakan, dan konteks peristiwa dapat membuat pasangan tidak sesuai walaupun kata kuncinya mirip.

## Data

Training set berisi pasangan `title` dan `content` dengan `label`. Test set berisi pasangan yang sama tanpa label dan identitas `id` yang harus dipertahankan untuk submission. Dataset Kaggle menyediakan `train.csv`, `test.csv`, dan `sample_submission.csv`.

## Evaluasi Dan Submission

Metrik yang digunakan adalah **Macro F1-Score**, yaitu rata-rata F1-Score untuk kelas 0 dan 1. File submission harus berupa CSV dengan kolom `id,label`; label hanya boleh bernilai `0` atau `1`.

Detail resmi dapat berubah, sehingga halaman Kaggle di atas tetap menjadi sumber kebenaran.
