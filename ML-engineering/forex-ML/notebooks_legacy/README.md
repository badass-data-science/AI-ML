# Legacy notebooks (deprecated)

These three notebooks are the original, pre-restructure pipeline: pull data from
InfluxDB, engineer features, window into arrays, and split into train/val/test. They
are kept here **for ad-hoc exploration only** — none of them are part of the actual
pipeline anymore.

The real pipeline is the `forex_ml` package at the repo root, orchestrated by the
Prefect flows in `forex_ml/flows/` and versioned end-to-end with DVC (`dvc.yaml`). See
the top-level `README.md` for how to run it.

| Legacy notebook | Superseded by |
|---|---|
| `prepare-training-and-inference-data.ipynb` | `forex_ml/data/features.py` + `forex_ml/flows/prepare_data_flow.py` |
| `make-all-training-data.ipynb` | `forex_ml/flows/prepare_all_flow.py` |
| `prepare-ml-ts-data.ipynb` | `forex_ml/data/splitting.py` + `forex_ml/flows/split_flow.py` |

`lstm.py` (the training script) and `granularity_to_seconds.py` were removed outright
rather than parked here — they're fully superseded by `forex_ml/training/` and
`forex.eda.eda_config.granularity_to_seconds_map` respectively, with no exploratory
value left in the originals.
