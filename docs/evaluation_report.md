# Model Evaluation Report: Flight Delay Prediction

## 1. Evaluation Methodology and Metric Definitions

Evaluating binary flight delay prediction under class imbalance requires multiple complementary metrics. Relying solely on overall accuracy can be severely misleading when the majority of flights arrive on time.

The metrics evaluated in this project (as computed in `notebooks/05_model_evaluation.ipynb` and `src/evaluate.py`) are defined as follows:

- **Accuracy**: The fraction of total predictions that the model got right across both delayed and on-time flights ($\frac{TP + TN}{TP + TN + FP + FN}$).
- **Precision**: The fraction of flights predicted as delayed that were truly delayed ($\frac{TP}{TP + FP}$), measuring the trustworthiness of positive delay alerts.
- **Recall (Sensitivity)**: The fraction of all actual delayed flights that the model successfully detected ($\frac{TP}{TP + FN}$), measuring coverage of delay events.
- **F1 Score**: The harmonic mean of precision and recall ($\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$), balancing false alarms against missed delays in a single number.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: The probability that the model ranks a randomly chosen delayed flight higher than a randomly chosen on-time flight across all possible classification thresholds.
- **PR-AUC (Precision-Recall - Area Under Curve / Average Precision)**: The area under the precision-recall curve across all operating thresholds, evaluating delay detection quality specifically on the minority positive class without being inflated by true negatives.
- **Brier Score**: The mean squared difference between predicted probabilities and actual binary outcomes ($\frac{1}{N} \sum (p_i - y_i)^2$), measuring probability calibration and sharpness (lower is better, where 0 indicates perfect probability assignment) [Brier, 1950].

---

## 2. Evaluation Results

Both models were trained on the chronological training set (229,142 flights from January to mid-September 2013) and evaluated on the final 15% unseen test set (49,102 flights from November 6 to December 31, 2013, with an empirical delay rate of 26.69%).

The exact metrics computed from the saved model bundles in `notebooks/05_model_evaluation.ipynb` and recorded in `reports/metrics/model_comparison.csv` are presented below:

| Metric | Logistic Regression Baseline | XGBoost Model |
| :--- | :---: | :---: |
| **Selected Decision Threshold** | `0.7223` | `0.3000` |
| **PR-AUC (Average Precision)** | **0.3084** | **0.3338** |
| **ROC-AUC** | **0.5682** | **0.5985** |
| **Precision** | **0.0000** | **0.3364** |
| **Recall** | **0.0000** | **0.5054** |
| **F1 Score** | **0.0000** | **0.4040** |
| **Accuracy** | **0.7331** | **0.6019** |
| **Brier Score (lower is better)** | **0.2092** | **0.2091** |

---

## 3. Model Comparison and Performance Analysis

Based on the empirical evaluation results, **XGBoost performs significantly better than the Logistic Regression baseline** for practical flight delay forecasting. This conclusion is supported by four key observations:

1. **Superior Threshold-Independent Discrimination**:
   Across both threshold-free ranking metrics, XGBoost achieves higher performance:
   - **PR-AUC**: XGBoost reaches **0.3338** compared to **0.3084** for Logistic Regression (+2.54 percentage points).
   - **ROC-AUC**: XGBoost reaches **0.5985** compared to **0.5682** for Logistic Regression (+3.03 percentage points).
   This confirms that XGBoost assigns higher relative risk scores to actual delayed flights than the baseline does, regardless of what decision threshold is chosen.

2. **Actionable Delay Detection in Practice**:
   At its validation-tuned threshold of `0.3000`, XGBoost captures **50.54% of all delayed flights** (Recall = 0.5054), identifying 6,624 true delays out of 13,106 actual delays in the test period, with a precision of **33.64%** and an F1 score of **0.4040**. In contrast, the Logistic Regression baseline at its saved threshold (`0.7223`) predicted zero positive flights, resulting in a Recall of **0.0000** and Precision of **0.0000**.

3. **Ability to Model Non-Linear and Interaction Effects**:
   Flight delays depend on complex interactions between time of day, seasonal factors, carrier delays, and airport congestion. In `notebooks/04_xgboost.ipynb`, feature importance analysis demonstrates that XGBoost captures non-linear splits on regional carrier effects (`carrier_EV` importance: 0.0598, `carrier_DL`: 0.0232), departure time (`hour`: 0.0452, `hour_sin`: 0.0253), seasonal patterns (`day_of_year_cos`: 0.0309), and high-density destinations (`dest_BOS`: 0.0219). Logistic regression [Cox, 1958] assumes a linear decision boundary on the log-odds scale and cannot model these combinatorial relationships without manual cross-product terms.

4. **The Accuracy Paradox Under Class Imbalance**:
   Logistic Regression reports a higher overall accuracy (**73.31%** vs. XGBoost's **60.19%**). However, this accuracy is completely trivial: in the test set, 73.31% of flights (35,996 out of 49,102) were on-time. By predicting that *no flight would ever be delayed*, Logistic Regression trivially achieves 73.31% accuracy while failing to detect a single delayed flight. XGBoost's 60.19% accuracy reflects an active classifier that accepts some false alarms in exchange for catching more than half of the true delays.

5. **Probability Calibration**:
   Both models achieve nearly identical Brier scores (**0.2091** for XGBoost vs. **0.2092** for Logistic Regression) [Brier, 1950]. However, calibration curve inspections in `notebooks/05_model_evaluation.ipynb` reveal that XGBoost probabilities cover a broader, more informative dynamic range, whereas Logistic Regression probabilities cluster tightly around the mean delay rate.

---

## 4. Project Limitations

1. **Class Imbalance**: Delays of $\ge$15 minutes represent only ~24.5% of flights in the dataset (~26.7% in the test set). Extreme class imbalance makes achieving simultaneous high recall (>70%) and high precision (>60%) difficult without additional signal.
2. **Missing Dynamic Operational Features**: The current models use only static timetable information (scheduled date, time, route, and carrier). Critical real-world determinants of flight delays are absent:
   - **Inbound Aircraft Status (Tail Propagation)**: Arrival delay of the preceding flight operated by the same aircraft (`tailnum`).
   - **Real-Time Weather Data**: Visibility, precipitation, wind shear, and convective storms at origin and destination airports.
   - **Airport Runway & Airspace Congestion**: Queue lengths, Ground Delay Programs (GDP), and gate holds.
3. **Absence of Temporal Deep Learning**: Although the project title and initial proposal cite *"Temporal Deep Learning"*, Phase 1 of this repository implements only tabular classification using Logistic Regression and XGBoost. No recurrent neural networks (LSTM/GRU) or entity embedding networks are currently implemented.
4. **Temporal Generalizability**: The dataset covers only 2013 New York area departures. Shifts in airline scheduling, airport infrastructure, air traffic control policies, and post-2013 operational patterns may degrade performance if applied to current flight operations without retraining.

---

## 5. Future Work

- **Phase 2 Deep Learning Extension**: Implement an entity embedding neural network or an LSTM model to capture sequential route histories and airport congestion states.
- **Weather Feature Integration**: Join hourly METAR surface weather records from NOAA with flight schedules via `time_hour` and `origin`.
- **Inbound Tail Delay Tracking**: Construct rolling lag features representing the departure or arrival delay of the same physical aircraft on its immediately preceding leg.
- **Cost-Sensitive Threshold Tuning**: Align the probability threshold with airline or passenger utility functions (e.g., weighing the cost of a missed passenger connection against the cost of an unnecessary delay alert).

---

## 6. References

1. **Brier, G. W. (1950)**. Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
2. **Chen, T., & Guestrin, C. (2016)**. XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.
3. **Cox, D. R. (1958)**. The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B (Methodological)*, 20(2), 215–242.
