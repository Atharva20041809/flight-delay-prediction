# Flight Delay Prediction (Machine Learning + Deep Learning)

Predict whether a flight will **arrive 15 or more minutes late**, using only information that is known **2 hours before scheduled departure**.

- **Phase 1 (done):** schedule-only features, Logistic Regression and XGBoost.
- **Phase 2 (done):** leakage-tested history features, a PyTorch MLP with learned embeddings, ablation studies, probability calibration, and an offline rescheduling what-if analysis.

Dataset: 2013 NYC flights (EWR, JFK, LGA), 336,776 rows, 327,346 with a known arrival delay (cancelled flights removed). Overall delay rate: 24.5%.

---

## Main findings (Phase 2)

| Model | Features | PR AUC | ROC AUC | Brier |
|---|---|---|---|---|
| Logistic Regression | Phase 1 (16) | 0.308 | 0.568 | 0.209 |
| Logistic Regression | Phase 2 (23) | 0.470 | 0.668 | 0.179 |
| Embedding MLP (PyTorch) | Phase 2 (23) | 0.472 | 0.669 | 0.179 |

All numbers are on the **test set** (Nov 6 to Dec 31, 2013). Five MLP seeds give PR AUC 0.4734 ± 0.0054.

In plain words:

1. **History features are what matter.** Adding seven past-only features (recent delays at the origin airport, congestion, carrier/route habits) raises PR AUC from 0.308 to about 0.47.
2. **The MLP only matches Logistic Regression.** The 0.003 gap is smaller than the seed noise (0.005), so we do not claim the neural network is better.
3. **Embeddings did not beat one-hot encoding** (0.4725 vs 0.4803). Dropout, weight decay and embedding size changed the score by less than two seed standard deviations.
4. **Calibration got worse after fitting on validation.** Raw MLP ECE is 0.041; after temperature scaling it is 0.072, after Platt scaling 0.117. Reason: the delay rate is 25.9% in train, 15.4% in validation and 26.7% in test (dataset shift).
5. **Rescheduling gives a small gain under strong assumptions.** About 100 to 242 expected delays avoided, which is 1.0% to 2.4% of all expected delays in the test set.

---

## Problem setup

| Item | Choice |
|---|---|
| Target | `delayed = 1` if `arr_delay >= 15` minutes |
| Prediction time | Scheduled departure minus 2 hours |
| Split | Chronological 70 / 15 / 15, no shuffling |
| Train | 229,142 flights (Jan 1 to Sep 13), delay rate 25.94% |
| Validation | 49,102 flights (Sep 13 to Nov 6), delay rate 15.39% |
| Test | 49,102 flights (Nov 6 to Dec 31), delay rate 26.69% |

**Leakage rules.** The flight's own `dep_time`, `dep_delay`, `arr_time`, `arr_delay` and `air_time` are never used as inputs. Other flights' outcomes are used only if they were already known at the prediction time. Scalers, encoders, imputation values, thresholds, temperatures and early stopping use training or validation data only. The test set is used only for reporting.

---

## Phase 2 in detail

### 1. History features (`src/phase2_history_features.py`)

| Feature | Meaning |
|---|---|
| `origin_delay_rate_last_3h` | Share of flights from the same origin that had already departed (within 3 h before the prediction time) with 15+ min departure delay |
| `origin_avg_dep_delay_last_3h` | Mean departure delay of those flights |
| `origin_delay_rate_last_24h` | Same, over the last 24 hours |
| `origin_flights_scheduled_same_hour` | Scheduled departures from the origin in that hour (congestion proxy) |
| `carrier_hist_delay_rate` | Carrier's delay rate on earlier days only |
| `route_hist_delay_rate` | Origin-destination delay rate on earlier days only |
| `carrier_origin_hour_delay_rate` | Carrier, origin and hour delay rate on earlier days only |

Phase 2 uses 20 numeric and 3 categorical inputs (23 features). The leakage test is in `tests/test_phase2_no_leakage.py` and checks that changing future or same-day outcomes does not change the features of earlier flights.

### 2. MLP with embeddings (`src/dl_model.py`)

- Embeddings: carrier 4, origin 3, destination 8 (plus one out-of-vocabulary id each).
- Input: 3 embeddings + 20 standardised numeric features = 35.
- Layers: `Linear(35,128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128,64) → ReLU → Dropout(0.3) → Linear(64,1)`.
- Parameters: 14,097 (912 in embeddings).
- Loss: class-weighted binary cross-entropy (`pos_weight` about 2.85).
- Optimiser: AdamW, learning rate 1e-3, weight decay 1e-4, batch size 1024.
- Early stopping on validation PR AUC (patience 5, at most 40 epochs, best weights restored).
- Trains in about 12 seconds on a CPU.

### 3. Ablations (`src/phase2_ablations.py`, notebook 03)

Embeddings vs one-hot, with vs without history features, dropout (0, 0.2, 0.3, 0.5), weight decay (0, 1e-4, 1e-3), embedding size (small, default, large), with vs without class weight, and Logistic Regression vs MLP. A 5-seed run measures random noise, so only differences larger than the noise are claimed.

### 4. Calibration (`src/phase2_calibration.py`, notebook 04)

Temperature scaling (Guo et al., 2017) for the MLP and Platt scaling for both models, fitted on validation only. Metrics: ECE (10 bins), Brier score, PR AUC, ROC AUC, and recall at fixed precision. The decision threshold is chosen on validation (precision about 0.40) and applied unchanged to test.

### 5. Rescheduling what-if (`src/phase2_rescheduling.py`, notebook 05)

A flight with calibrated risk at or above a threshold is moved to the best hour within ±2 hours (same day, 06:00 to 22:00) if the risk falls by more than a penalty; at most 5 moved flights are added per airport-hour. **Assumptions:** calibrated probability is treated as true probability; only the hour features change; the model's hour effect is treated as causal (the weakest assumption); crews, gates and aircraft rotations are not modelled. This is an offline simulation, not an operational recommendation.

---

## Limitations

- One year of data from three airports; the validation period is not representative (15.4% delays).
- No weather, aircraft-rotation or network features.
- Seed noise (0.005) is larger than most ablation effects; only five seeds were used.
- No hybrid or temporal model was built.
- An early internal XGBoost run on the same history features reached PR AUC 0.407 (below both models here). It was removed from the final Phase 2 pipeline, so no claim about gradient boosting with these features is made. Phase 1 XGBoost results are unchanged.
- In the rescheduling study about 80% of moves are to *later* hours; we did not investigate why.

---

## Project structure

```
flight-delay-prediction/
├── data/
│   ├── raw/                       # flights.csv (download separately, see below)
│   └── processed/                 # processed data (generated)
├── notebooks/
│   ├── 01 ... 05                  # Phase 1 notebooks (EDA, features, baselines, XGBoost, evaluation)
│   ├── phase2_01_features_baselines.ipynb
│   ├── phase2_02_dl_mlp_embeddings.ipynb
│   ├── phase2_03_ablations.ipynb
│   ├── phase2_04_calibration_validation.ipynb
│   └── phase2_05_rescheduling_whatif.ipynb
├── src/
│   ├── data_loader.py, preprocessing.py, feature_engineering.py    # shared
│   ├── train.py, evaluate.py, calibration.py                       # Phase 1
│   ├── phase2_history_features.py
│   ├── phase2_train.py
│   ├── dl_model.py
│   ├── phase2_ablations.py
│   ├── phase2_calibration.py
│   └── phase2_rescheduling.py
├── tests/
│   └── test_phase2_no_leakage.py
├── models/                        # saved models (generated)
├── reports/
│   ├── figures/                   # plots (phase2_* and dl_* are Phase 2)
│   └── metrics/                   # result tables (csv / md)
├── docs/                          # Phase 1 and Phase 2 notes and reports
├── run_phase2_all.py              # runs the full Phase 2 pipeline
└── requirements.txt
```

---

## How to run

### 1. Set up

```bash
git clone https://github.com/Atharva20041809/flight-delay-prediction.git
cd flight-delay-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Download `flights.csv` (2013 NYC flights) from Kaggle and place it at `data/raw/flights.csv`. It is not stored in the repository because of its size.

### 3. Run Phase 2

Option A, the whole pipeline in one command (this also creates the saved models in `models/`):

```bash
python3 run_phase2_all.py
```

Option B, the notebooks, in this order:

1. `phase2_01_features_baselines.ipynb` (features, leakage tests, Logistic Regression)
2. `phase2_02_dl_mlp_embeddings.ipynb` (the MLP)
3. `phase2_03_ablations.ipynb` (ablations and seed noise)
4. `phase2_04_calibration_validation.ipynb` (needs the saved models in `models/`)
5. `phase2_05_rescheduling_whatif.ipynb` (needs the saved models in `models/`)

If `models/` is empty, run `python3 run_phase2_all.py` once first.

### 4. Run the leakage test

```bash
python3 -m tests.test_phase2_no_leakage
```

---

## Where the results are saved

| File (in `reports/metrics/`) | Content |
|---|---|
| `phase2_baseline_comparison.csv` | Logistic Regression, Phase 1 vs Phase 2 features |
| `phase2_mlp_vs_baselines.csv` | MLP next to the baselines |
| `phase2_ablation_results.csv` / `.md` | All ablation groups |
| `phase2_mlp_seed_variation.csv` | 5-seed noise estimate |
| `phase2_calibration_results.csv` / `.md` | ECE, Brier, PR AUC, ROC AUC, operating points |
| `phase2_recall_at_precision.csv` | Maximum recall at minimum precision 0.3, 0.4, 0.5 |
| `phase2_rescheduling_main_summary.csv`, `phase2_rescheduling_sensitivity.csv` / `.md` | Rescheduling results |

Figures are in `reports/figures/`. The design decisions and assumptions are written in `docs/phase2_notes.md`.

---

## Notes for contributors

- The existing `.gitignore` ignores `models/*`, `reports/figures/*` and `reports/metrics/*`. Change it if you want the Phase 2 models, plots and tables to be pushed.
- `data/raw/*` and `data/processed/*` should stay ignored (large files).
- Phase 1 files must not be modified.
- The same split, seed (42) and prediction time are used everywhere, so results are comparable.

---

## Team

| Member | Phase 2 part |
|---|---|
| [Member 1] | Data, leakage and history features |
| [Member 2] | Deep learning model and regularisation |
| [Member 3] | Results and ablations |
| [Member 4] | Calibration and validation |
| [Member 5] | Rescheduling what-if and conclusions |

Team ID: [T0XX]

---

## Key references

- Rebollo and Balakrishnan (2014), *Characterization and prediction of air traffic delays*, Transp. Res. C.
- Guo and Berkhahn (2016), *Entity embeddings of categorical variables*.
- Guo, Pleiss, Sun and Weinberger (2017), *On calibration of modern neural networks*, ICML.
- Grinsztajn, Oyallon and Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS.
- Ovadia et al. (2019), *Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift*, NeurIPS.

The full reference list is in the Phase 2 report.
