# Cybersecurity Threat Detection with Deep Learning

Sequential neural network achieving **93.6% test accuracy** (95.6% train, 96.0% val) on BETH dataset (763K+ system logs) for binary classification of malicious events.

## Methodology
- **Dataset**: BETH cybersecurity logs (`sus_label`: 0=benign, 1=malicious) with 99.8% class imbalance
- **Architecture**: PyTorch MLP (input→128→64→1, sigmoid) + StandardScaler normalization
- **EDA**: `userId` exhibits strongest correlation with target (ρ=0.86); multimodal `returnValue`/`argsNum` distributions
- **Results**: 95.6% train | 96.0% val | **93.6% test accuracy**

## Key Metrics

| Split      | Accuracy |
|------------|----------|
| Training   | 95.6%    |
| Validation | 96.0%    |
| Testing    | 93.6%    |

## Tech Stack

* Python (PyTorch, Scikit-learn, Pandas, Matplotlib, Seaborn)
* Jupyter Notebook

## Research Impact

Demonstrates deep learning efficacy for anomaly detection in severely imbalanced cybersecurity logs, surpassing traditional rule-based systems.

## Next Steps

- Class weighting/SMOTE for imbalance
- Dropout + early stopping
- Confusion matrix + ROC-AUC
- Real-time deployment
