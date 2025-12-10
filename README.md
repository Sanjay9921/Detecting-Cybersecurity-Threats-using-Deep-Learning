# Cybersecurity Threat Detection with Deep Learning

This project implements a deep learning pipeline for detecting malicious events in large-scale system logs using the BETH cybersecurity dataset. The final PyTorch model achieves 95.6% training accuracy, 96.0% validation accuracy, and **93.6% test accuracy** on over 760k log events.

## Overview

* The goal is to classify system log events as benign or malicious based on process and user metadata from the BETH dataset.
* The workflow covers data loading, preprocessing, exploratory data analysis, model training, and evaluation in a single, reproducible Jupyter notebook.

## Data and Problem Setting

* The project uses the BETH “Real Cybersecurity Data for Anomaly Detection Research” dataset, consisting of labeled Linux system call logs with a binary sus_label target (0 benign, 1 suspicious).
* The training set is **highly imbalanced** (about 99.8 percent benign), while test and validation splits exhibit different benign malicious ratios, reflecting realistic deployment conditions.

### Feature Schema

* Core numerical features per event include: ``processId``, ``threadId``, ``parentProcessId``, ``userId``, ``mountNamespace``, ``argsNum``, ``returnValue``, and ``sus_label`` as the binary target.
* The notebook confirms seven feature columns and one label across train, validation, and test splits, each with roughly 189k test and validation events and 763k training events.

## Exploratory Data Analysis

The EDA module inspects class imbalance, feature distributions, and correlations with the target. ``userId`` shows the strongest positive correlation with ``sus_label`` (approximately 0.86), while other features have weak linear relationships, motivating non-linear modeling.​

### Key findings

* **Training split:** about 99.8 percent benign versus 0.2 percent malicious; validation split is similarly skewed, whereas the test split is dominated by malicious events. 
* ``processId``, ``threadId``, and ``parentProcessId`` exhibit heavy-tailed distributions with large numbers of statistical outliers, typical for process logs.
* ``argsNum`` and ``returnValue`` show clearly different multimodal patterns between benign and malicious events.

## Preprocessing Pipeline

The notebook loads preprocessed CSV splits for train, validation, and test sets, separates features and labels, and applies standard scaling. ``StandardScaler`` is fit on the training features and applied to validation and test features to normalize process-level identifiers and stabilize neural network optimization.

## Model Architecture

The threat detector is a fully connected multilayer perceptron implemented in PyTorch.

### Architecture

* Input layer with 7 standardized numerical features.
* Hidden layer 1: 128 units with ReLU activation.
* Hidden layer 2: 64 units with ReLU activation.
* Output layer: 1 neuron with sigmoid activation for binary classification.

The model is trained with stochastic gradient descent, L2 weight decay, and a binary loss applied to sigmoid outputs. Predictions are thresholded and evaluated on train, validation, and test splits within the notebook.

## Results

The final trained model achieves:

| Split	| Accuracy |
|-------|----------|
| Training | 95.6% |
| Validation | 96.0% |
| Testing | 93.6% |

Despite extreme training imbalance, the network learns a robust decision boundary that generalizes across splits with different benign malicious mixtures. A PyTorch ``Accuracy`` metric is used to validate correctness of the manual accuracy computation across all splits.

## Technical Stack

* Python with Jupyter Notebook for experimentation and reporting.
* PyTorch for model definition, training loops, and tensor operations.
* Scikit-learn for scaling, accuracy verification, and classical utilities.
* Pandas, NumPy, Matplotlib, and Seaborn for data handling and visualization.

## Research Impact

The project demonstrates that a relatively compact MLP can achieve high accuracy on severely imbalanced, high-volume system log data without extensive feature engineering. By learning from raw process and user metadata, the model can flag rare but high-impact malicious events and has clear potential as a component in modern anomaly detection pipelines. 

## Next Steps

Planned extensions outlined in the notebook include:​

* Integrating class weighting or synthetic oversampling (for example SMOTE) to address imbalance more explicitly.
* Adding regularization such as dropout and early stopping to further harden the model against overfitting.
* Computing confusion matrices, precision recall, and ROC AUC to characterize performance under different thresholds and class priors.
* Exploring real-time or near real-time deployment scenarios on streaming log data.