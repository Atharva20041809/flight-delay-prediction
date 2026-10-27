# Literature Review: Flight Delay Prediction

## 1. Introduction
Flight delays affect passengers, airlines, airport operations, and connecting flights. Machine learning can use historical flight information and scheduled departure details to estimate the probability of a delay before departure.

## 2. Flight Delay Prediction as Classification
This project treats delay prediction as a binary classification problem. A flight is labeled delayed when its arrival delay is at least 15 minutes. Flights with smaller arrival delays are labeled not delayed. Rows without an arrival-delay label are excluded.

## 3. Logistic Regression
Logistic Regression is used as a baseline model. It estimates the probability of a binary outcome and provides a simple reference for evaluating more complex models.

## 4. Gradient Boosting with XGBoost
XGBoost is a gradient-boosted decision-tree method. It builds trees sequentially, with later trees improving predictions made by earlier trees. It can model nonlinear relationships and interactions between flight features.

## 5. Feature Engineering
The project uses scheduled time, calendar information, carrier, origin, destination, and flight distance. Cyclical encoding represents recurring time variables such as hour and day of week using sine and cosine transformations, preserving their cyclic relationships.

## 6. Evaluation
Models are assessed using Precision, Recall, F1-score, PR AUC, ROC AUC, accuracy, and Brier score. Precision and Recall are important because missing a delayed flight and incorrectly flagging an on-time flight have different consequences. The classification threshold affects this trade-off.

## 7. Data Leakage and Temporal Evaluation
Only information available from the schedule and flight plan should be used as model input. Actual arrival delay and other post-flight outcomes must not be used as predictive features. A chronological split provides a more realistic test of predictions on later flights than a randomly shuffled split.

## 8. Scope and Limitations
Phase 1 compares Logistic Regression and XGBoost using the available 2013 New York City flight dataset. The results may not generalize to other years, airports, or changing airline operations. The dataset is historical, and the models do not establish causal reasons for delays.

## References
See [references.bib](references.bib) for the main methodological references.
