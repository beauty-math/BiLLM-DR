# Data Directory

This directory is intentionally empty in the GitHub release.

Place Cdataset/Fdataset-style files as follows:

```text
data/Cdataset/Cdataset.mat
data/Cdataset/drug_desc.csv
data/Cdataset/disease_desc.csv
data/Fdataset/Fdataset.mat
data/Fdataset/drug_desc.csv
data/Fdataset/disease_desc.csv
```

LLM checkpoints are not included. Set `MODEL_PATH` to your local pretrained model directory, or place the model under `data/LLM/` if you prefer the default layout.
