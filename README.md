# TutorGPT Dacon Pipeline

Welcome to the **TutorGPT Dacon Pipeline**. This project is dedicated to the "Care" philosophy—understanding that every data point represents a student's journey. Our goal is to predict learner completion accurately so that interventions can be provided to those who need them most.

## 🏆 Competition Overview
**Goal**: Predict whether a learner will complete a course (1) or not (0).
**Metric**: F1 Score (Macro/Binary depending on specific regulation, currently optimizing for robustness).

## 🏗 Project Structure

This project follows a service-oriented architecture adapted for Python Data Science:

```
TutorGPT_Dacon/
├── data/
│   └── generate_mock.py    # Generates realistic mock data for testing the pipeline
├── src/
│   ├── preprocessing.py    # PreprocessingService: Handles missing values & encoding
│   ├── model.py            # EnsembleModel: XGBoost + LightGBM + CatBoost
│   ├── train.py            # TrainingService: Stratified K-Fold CV
│   └── inference.py        # InferenceService: Averaged predictions from all folds
├── main.py                 # CLI Entry point
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
To generate data, train models, and create a submission in one go:

```bash
python main.py --mode all
```

### 3. Individual Steps

**Generate Data Only:**
```bash
python main.py --mode data
```

**Train Only:**
```bash
python main.py --mode train
```

**Inference Only:**
```bash
python main.py --mode infer
```

## 🧠 Key Technical Features

*   **Robust Preprocessing**: We use a stateful `PreprocessingService` that fits on training data and transforms test data, ensuring no data leakage. It handles categorical encoding and missing value imputation intelligently.
*   **Ensemble Modeling**: A weighted soft-voting ensemble of **XGBoost**, **LightGBM**, and **CatBoost** leverages the unique strengths of each gradient boosting framework.
*   **Stratified Cross-Validation**: We train on 5 folds to ensure our F1 score estimates are reliable and not just a result of a lucky random split.
*   **Fold-Averaged Inference**: The final prediction is the average probability of all 5 fold models, significantly reducing variance and overfitting risks.

## 🤝 The "Care" Philosophy
In code, "Care" means:
1.  **Readability**: Clear variable names and type hints.
2.  **Modularity**: Separation of concerns (Data vs Model vs Training).
3.  **Safety**: Logging, error handling, and avoiding silent failures.

---
*Built by the TutorGPT Team*
