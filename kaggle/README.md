# Kaggle Competition Template

This folder is a reusable starter template for Kaggle competitions.

## Structure

```text
kaggle
├── input
│   └── {competition-slug}
│       ├── test.csv
│       └── train.csv
├── src
├── working
│   ├── notebook.ipynb
│   └── submission.csv
├── Original_README.md
└── README.md
```

## How to use

1. Copy this `kaggle/` folder into your project.
2. Rename `input/{competition-slug}` to your real competition slug.
3. Put downloaded Kaggle dataset files in that folder.
4. Add your scripts in `src/`.
5. Work in `working/notebook.ipynb` and export predictions to `working/submission.csv`.
6. Keep the original competition description in `Original_README.md`.

## Share with friends

- Send this folder as-is.
- Your friend only needs to replace `{competition-slug}` and dataset files to start.
