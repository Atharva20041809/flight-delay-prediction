# Flight Delay Prediction: Viva Preparation Notes

This document contains 12 anticipated viva questions covering evaluation, methodology, and results, designed with clear, 2–3 line answers grounded strictly in the project's empirical results.

---

### Q1: What is the main objective of this project, and how is a flight delay defined?
**Answer:** The objective is to predict whether a scheduled commercial flight will experience a substantial arrival delay before it departs. Following standard FAA guidelines, a flight is labeled delayed (`delayed = 1`) if its arrival delay is 15 minutes or greater (`arr_delay >= 15`); otherwise, it is labeled on-time (`delayed = 0`).

### Q2: Why was Logistic Regression selected as the baseline model?
**Answer:** Logistic Regression [Cox, 1958] is an industry-standard benchmark for binary classification because it is computationally fast, highly interpretable, and provides calibrated probabilities via a linear combination of features. It serves as a reference point to prove whether complex tree-based ensembles like XGBoost justify their additional complexity.

### Q3: Why was XGBoost chosen as the primary non-linear model?
**Answer:** XGBoost [Chen & Guestrin, 2016] is a state-of-the-art gradient boosted decision tree algorithm capable of capturing complex non-linear relationships, threshold cutoffs, and feature interactions (such as late-afternoon departures interacting with congested airports) without needing manual feature cross-products.

### Q4: Why is overall Accuracy a misleading metric for this project?
**Answer:** Because the dataset is imbalanced: only 24.47% of flights in the dataset (and 26.69% in the test set) are delayed. A naive dummy model predicting "no delay" for every single flight achieves ~73.3% accuracy while failing to detect a single delayed flight, proving that accuracy alone cannot evaluate delay detection quality.

### Q5: Why do we evaluate ROC-AUC and PR-AUC instead of only threshold-dependent metrics?
**Answer:** ROC-AUC and PR-AUC evaluate the model's fundamental ranking ability across *all* possible decision thresholds rather than a single arbitrary cutoff. PR-AUC is especially vital here because it focuses directly on the minority positive class (delayed flights) without being artificially inflated by the high volume of true negative (on-time) flights.

### Q6: What does the Brier Score measure, and what did our results show?
**Answer:** The Brier Score [Brier, 1950] measures probability calibration and sharpness as the mean squared error between predicted probabilities and actual 0/1 outcomes (lower is better). Both models achieved virtually identical Brier scores (~0.2091 for XGBoost vs. ~0.2092 for Logistic Regression), indicating comparable overall calibration despite XGBoost's better ranking ability.

### Q7: How was class imbalance handled during model training and evaluation?
**Answer:** In Logistic Regression, we configured `class_weight="balanced"` to penalize false negatives inversely proportional to class frequencies. For XGBoost, we tuned the decision threshold on validation data down to `0.30` (instead of the default `0.50`), allowing the model to capture over 50.5% of true delays.

### Q8: Why did we use a chronological train/val/test split instead of random k-fold cross-validation?
**Answer:** Flight records are time-series events that exhibit strong seasonal trends and serial delay propagation. A random shuffle split would leak future information (such as system-wide delays on a specific day) into past training samples, artificially inflating test metrics. A chronological split (70% train, 15% validation, 15% test) accurately simulates real-world deployment on future unseen flights.

### Q9: How was data leakage prevented in feature engineering?
**Answer:** We strictly excluded all post-departure and operational outcome variables—including actual departure time (`dep_time`), departure delay (`dep_delay`), arrival time (`arr_time`), arrival delay (`arr_delay`), and flight air time (`air_time`). Only features strictly known from the flight timetable prior to departure (scheduled time, date, route, distance, and carrier) were retained.

### Q10: Why did Logistic Regression report 0.0000 Precision and Recall in the evaluation table?
**Answer:** In the saved baseline bundle, the decision threshold was set at `0.7223` (derived from a high-precision validation target). At this high threshold, the model's predicted probabilities on the test set never exceeded 0.7223, resulting in zero positive predictions ($TP = 0, FP = 0$), which trivially yielded 0.0000 Precision, 0.0000 Recall, and a deceptive 73.31% Accuracy.

### Q11: What were the most influential features driving delays according to XGBoost?
**Answer:** Feature importance analysis in `notebooks/04_xgboost.ipynb` showed that regional carriers (`carrier_EV` importance: 0.0598, `carrier_DL`: 0.0232), scheduled hour of departure (`hour`: 0.0452, `hour_sin`: 0.0253), seasonal cosine transforms (`day_of_year_cos`: 0.0309), and high-traffic destination airports (`dest_BOS`: 0.0219) had the strongest impact on delay predictions.

### Q12: What are the primary limitations of the current pipeline, and what would we do next?
**Answer:** The primary limitation is that models rely solely on static schedule data without real-time airport weather (METAR) or preceding aircraft tail-turnaround delays (`tailnum` propagation). Next steps would involve Phase 2 temporal deep learning (e.g., LSTMs or airport entity embeddings) and joining live weather streams to boost predictive power beyond static schedules.
