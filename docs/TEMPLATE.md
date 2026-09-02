# Repository Template

This repository is a reusable starter template for Kaggle competitions. The project is intentionally kept simple and does not use DVC.

## Structure

See the root `README.md` for the current structure and workflow.

## How to use

1. Rename `data/raw/{competition-slug}` to the real competition slug, such as `data/raw/penyisihan-dac-ifest-2026`.
2. Put downloaded, immutable Kaggle files in that raw-data folder.
3. Add reusable scripts in `src/` and experiments in `notebooks/`.
4. Use one Git branch per individual, such as `member/your-name`, and distinguish experiments with an `experiment_id`; no additional working folder is required.
5. Write each notebook's predictions under `submissions/<notebook-name>/`.

## Share with friends

- Share the repository as-is.
- Teammates only need to replace the competition slug/data and fill in the member names.
