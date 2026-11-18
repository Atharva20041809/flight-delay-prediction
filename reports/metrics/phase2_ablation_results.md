| Group | Variant | Threshold | PR AUC | ROC AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|---|
| A: Architecture | MLP with Embeddings (Default) | 0.4706 | 0.4725 | 0.6690 | 0.6002 | 0.2347 | 0.3374 | 0.1789 |
| A: Architecture | MLP with One-Hot Encoding | 0.4570 | 0.4803 | 0.6771 | 0.6182 | 0.2284 | 0.3336 | 0.1766 |
| B: Features | With History Features (Default) | 0.4706 | 0.4725 | 0.6690 | 0.6002 | 0.2347 | 0.3374 | 0.1789 |
| B: Features | Without History Features (Phase 1 Features) | 0.7148 | 0.3239 | 0.5889 | 0.0000 | 0.0000 | 0.0000 | 0.1948 |
| C: Dropout | Dropout 0.0 | 0.4661 | 0.4794 | 0.6768 | 0.5927 | 0.2495 | 0.3512 | 0.1765 |
| C: Dropout | Dropout 0.2 | 0.4751 | 0.4795 | 0.6770 | 0.5632 | 0.2983 | 0.3900 | 0.1796 |
| C: Dropout | Dropout 0.3 (Default) | 0.4706 | 0.4725 | 0.6690 | 0.6002 | 0.2347 | 0.3374 | 0.1789 |
| C: Dropout | Dropout 0.5 | 0.4751 | 0.4783 | 0.6744 | 0.5755 | 0.2780 | 0.3749 | 0.1816 |
| D: Weight Decay | No Weight Decay (0.0) | 0.4706 | 0.4721 | 0.6685 | 0.6007 | 0.2348 | 0.3376 | 0.1789 |
| D: Weight Decay | Weight Decay 1e-4 (Default) | 0.4706 | 0.4725 | 0.6690 | 0.6002 | 0.2347 | 0.3374 | 0.1789 |
| D: Weight Decay | Weight Decay 1e-3 | 0.4661 | 0.4718 | 0.6682 | 0.6020 | 0.2341 | 0.3371 | 0.1787 |
| E: Embedding Size | Default (4, 3, 8) | 0.4706 | 0.4725 | 0.6690 | 0.6002 | 0.2347 | 0.3374 | 0.1789 |
| E: Embedding Size | Small (2, 2, 4) | 0.4661 | 0.4699 | 0.6697 | 0.5807 | 0.2544 | 0.3538 | 0.1791 |
| E: Embedding Size | Large (8, 6, 16) | 0.4661 | 0.4803 | 0.6803 | 0.5686 | 0.2800 | 0.3753 | 0.1780 |
| F: Class Weighting | With pos_weight (Default) | 0.4706 | 0.4725 | 0.6690 | 0.6002 | 0.2347 | 0.3374 | 0.1789 |
| F: Class Weighting | Without pos_weight (Unweighted) | 0.2399 | 0.4741 | 0.6701 | 0.6313 | 0.2029 | 0.3071 | 0.2012 |
| G: Model Comparison | Logistic Regression (Phase 2) | 0.4842 | 0.4695 | 0.6676 | 0.6453 | 0.1825 | 0.2845 | 0.1785 |
| G: Model Comparison | PyTorch MLP (Phase 2) | 0.4706 | 0.4725 | 0.6690 | 0.6002 | 0.2347 | 0.3374 | 0.1789 |
