# Phase 2 Design Decisions and Implementation Notes

This document records all methodological choices, operational assumptions, and architectural configurations implemented for Phase 2 of the Flight Delay Prediction project.

---

## 1. Leakage Prevention and Historical Features
- **Prediction Cutoff Time $T$**: Defined strictly as $\text{scheduled\_datetime} - 2\text{ hours}$.
- **Forbidden Variables**: The flight's own `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, and `air_time` are explicitly excluded from model inputs.
- **Rolling Origin Features**:
  - `origin_delay_rate_last_3h`, `origin_avg_dep_delay_last_3h`, and `origin_delay_rate_last_24h` consider only flights departing from the same origin airport that had **actually departed** (i.e. $\text{scheduled\_datetime} + \text{dep\_delay} < T$).
  - A departed flight is counted as departure-delayed if $\text{dep\_delay} \ge 15$ minutes (standard FAA threshold).
  - Computed using binary search (`np.searchsorted`) and prefix sums, guaranteeing $O(N \log N)$ complexity and strict mathematical exclusion of any departures occurring at or after $T$.
- **Congestion Proxy**:
  - `origin_flights_scheduled_same_hour`: Counts total scheduled departures from the same origin airport during the scheduled departure hour. Since timetables are fixed months ahead of time, this contains zero outcome leakage.
- **Daily Historical Delay Rates**:
  - `carrier_hist_delay_rate`, `route_hist_delay_rate`, and `carrier_origin_hour_delay_rate` aggregate delay rates strictly over calendar dates prior to flight date $D$ (shifted by at least 1 full day).
  - Missing values (e.g., first day of the year or unobserved route combinations) are imputed using **training set statistics only** (global training delay rate of ~25.94%), preventing information flow from validation or test periods.
- **Verification**: A formal test suite in `tests/test_no_leakage.py` empirically verifies feature invariance when same-day or future outcomes are perturbed.

---

## 2. Chronological Data Split and Evaluation Protocol
- **Split Proportions**: 70% Training (229,142 flights, Jan 1 to Sep 13), 15% Validation (49,102 flights, Sep 13 to Nov 6), and 15% Test (49,102 flights, Nov 6 to Dec 31).
- **Temporal Distribution Shift (Known Limitation)**:
  - Training delay rate: 25.94%
  - Validation delay rate: 15.39% (autumn period with milder weather and reduced congestion)
  - Test delay rate: 26.69% (winter holiday peak and convective weather)
  - Rather than hiding this shift, all threshold selections are performed strictly on the validation set to mirror real-world prospective deployment.
- **Validation Threshold Selection**:
  - Decision thresholds are selected on the validation split targeting a precision of $\approx 0.40$ (configurable), and subsequently applied unmodified to test data. Test data is never used for hyperparameter tuning or threshold picking.

---

## 3. Deep Learning Model Architecture
- **Categorical Entity Embeddings**:
  - `carrier`: Embedding dimension = 4
  - `origin`: Embedding dimension = 3
  - `dest`: Embedding dimension = 8
  - Includes an explicit Out-Of-Vocabulary (OOV) index for unseen categories.
- **Numeric Features**:
  - 20 standardized features (13 baseline scheduled/cyclical features + 7 historical congestion features).
  - Scaled using `StandardScaler` fitted strictly on the training set.
- **Feedforward Trunk**:
  - Input: Concat(embeddings, numeric) $\to$ `Linear(35, 128)` $\to$ `BatchNorm1d(128)` $\to$ `ReLU` $\to$ `Dropout(0.3)` $\to$ `Linear(128, 64)` $\to$ `ReLU` $\to$ `Dropout(0.3)` $\to$ `Linear(64, 1)`.
- **Optimization**:
  - Optimizer: `AdamW` (learning rate $10^{-3}$, weight decay $10^{-4}$).
  - Mini-batch size: 1024.
  - Loss Function: `BCEWithLogitsLoss` with positive class weight $w \approx 2.85$ (ratio of negatives to positives in train set).
  - Early stopping: Monitored on validation PR AUC with patience of 5 epochs (maximum 40 epochs).

---

## 4. Probability Calibration
- **Temperature Scaling (Guo et al. 2017)**:
  - Single positive scalar parameter $T$ optimized on validation logits by minimizing negative log-likelihood (NLL).
  - Applied to scale test set logits: $\hat{p} = \sigma(z_{\text{test}} / T)$.
- **Platt Scaling**:
  - Univariate logistic regression fitted on validation log-odds for the Logistic Regression baseline.
- **Metric Tracking**:
  - Brier score and Expected Calibration Error (ECE, 10 uniform bins) evaluated before and after calibration.

---

## 5. Offline Rescheduling What-If Simulation
- **Operational Assumptions**:
  1. *Calibrated Probability Representation*: Calibrated model predictions are treated as the true probability of delay.
  2. *Alternative Slot Search Space*: Candidate departure times are restricted to $\pm 2$ hours of original scheduled departure on the same day, within standard operational airport hours (06:00 to 22:00).
  3. *Switching Penalty*: Rescheduling incurs operational friction (gate reassignments, passenger coordination), represented as an required minimum risk reduction penalty $\delta \in \{0.03, 0.08\}$.
  4. *Airport-Hour Capacity Constraints*: A maximum of $C = 5$ additional rescheduled flights are permitted per airport-hour slot to prevent secondary congestion bottlenecks.
- **Academic Scope Disclaimer**: This is an offline simulation designed to evaluate how neural delay risk scores could assist scheduling decisions under explicit constraints; it does not constitute an operational aviation dispatch recommendation.

