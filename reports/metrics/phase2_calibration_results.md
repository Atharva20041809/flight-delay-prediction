| Model | Calibration | ECE | Brier | PR AUC | ROC AUC | Val threshold | Test precision | Test recall | Max recall @ prec>=0.40 |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | None (raw) | 0.0314 | 0.1785 | 0.4695 | 0.6676 | 0.4842 | 0.6453 | 0.1825 | 0.5438 |
| Logistic Regression | Platt scaling | 0.1289 | 0.1964 | 0.4695 | 0.6676 | 0.2716 | 0.6465 | 0.1822 | 0.5438 |
| PyTorch MLP | None (raw) | 0.0407 | 0.1788 | 0.4723 | 0.6688 | 0.4706 | 0.6041 | 0.2296 | 0.5327 |
| PyTorch MLP | Temperature scaling | 0.0722 | 0.1837 | 0.4723 | 0.6688 | 0.4480 | 0.6039 | 0.2302 | 0.5327 |
| PyTorch MLP | Platt scaling | 0.1175 | 0.1925 | 0.4723 | 0.6688 | 0.2535 | 0.6027 | 0.2320 | 0.5327 |
