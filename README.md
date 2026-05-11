# ⚖️ Linear Models, Sparsity & Interpretability

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-Gradient%20Descent-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![PyGAM](https://img.shields.io/badge/PyGAM-Interpretability-teal)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> A from-scratch JAX implementation of Logistic Regression with L1/L2 penalties and interaction terms — applied to criminal recidivism prediction (COMPAS) and spacecraft telemetry (UCI Shuttle) with full model interpretability analysis.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Part 1: Build the Model](#-part-1-build-the-model)
- [Part 2: Analysis & Interpretability](#-part-2-analysis--interpretability)
- [Key Findings](#-key-findings)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [References](#-references)

---

## 🔍 Overview

This project has two components:

1. **Build** — Implement logistic regression from scratch using JAX with gradient descent, optional L1/L2 sparsity penalties, pairwise interaction terms, and a separate penalty for interaction terms only.

2. **Analyze** — Apply those models alongside Generalized Additive Models (GAMs) to two real-world datasets, comparing predictive performance, sparsity, and interpretability.

---

## 📂 Repository Structure

```
.
├── logistic_regression.py                      # Custom JAX logistic regression implementation
├── util.py                                     # Shared evaluation & visualization utilities
├── Assignment_2_Linear_Models-Validate_Model.ipynb   # Step-by-step model build & validation
├── Assignment_2_Linear_Models_Analysis.ipynb         # COMPAS + Shuttle analysis
├── shuttle.csv                                 # UCI Shuttle dataset
└── README.md
```

---

## 📊 Datasets

### COMPAS Recidivism Dataset
Loaded directly from the [ProPublica GitHub repository](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv).

| Detail | Value |
|---|---|
| Task | Binary classification — predict 2-year recidivism |
| Labels | `two_year_recid` (true label), `score_text` (COMPAS prediction) |
| Features | Age, sex, race, juvenile/prior offense counts, charge degree |
| Preprocessing | Row filtering, one-hot encoding, standard scaling |

### UCI Shuttle Dataset (`shuttle.csv`)
Spacecraft sensor telemetry from NASA.

| Detail | Value |
|---|---|
| Rows | ~5,000 |
| Features | 7 continuous sensor readings (`Rad Flow`, `Fpv Close`, `Fpv Open`, etc.) |
| Task | Binary classification on the `label` column |

---

## 🔨 Part 1: Build the Model

**Notebook:** `Assignment_2_Linear_Models-Validate_Model.ipynb`

The custom `LogisticRegression` class in `logistic_regression.py` is built in 4 incremental steps:

### Step 1️⃣ — Core Logistic Regression
- Forward pass via sigmoid activation: `σ(Xw + b)`
- Binary cross-entropy loss
- Gradient descent weight updates using JAX's `grad`
- Validated against scikit-learn's implementation on a toy Gaussian dataset

### Step 2️⃣ — Sparsity Penalty
- Adds optional **L1** or **L2** regularization on feature weights
- Controlled by `penalty` (`'l1'`, `'l2'`, or `None`) and `alpha`
- Validated by showing increasing sparsity as `alpha` grows

### Step 3️⃣ — Interaction Terms
- Generates all pairwise feature products `xᵢ * xⱼ` via `itertools.combinations`
- Features scaled post-interaction with a fitted `StandardScaler`
- Validated by checking interaction weights appear in the weight plot

### Step 4️⃣ — Separate Interaction Penalty
- `interaction_alpha` applies a separate L1/L2 penalty to interaction weights only
- Validated by zeroing out interaction terms with a high `interaction_alpha` while keeping base feature weights intact

---

## 📈 Part 2: Analysis & Interpretability

**Notebook:** `Assignment_2_Linear_Models_Analysis.ipynb`

### Section 1 — COMPAS Labels (predicting COMPAS's own scores)
| Model | Notes |
|---|---|
| Logistic Regression (default) | Baseline; weights reveal `priors_count` and `age` as top drivers |
| GAM (`LogisticGAM`) | Spline terms for continuous features, factor terms for binary; partial dependence plots per feature |

Includes a replication of the **ProPublica risk decile histograms** comparing Black and White defendants.

### Section 2 — True Recidivism Labels
| Model | Notes |
|---|---|
| Logistic Regression (default) | Baseline on `two_year_recid` |
| Sparsest L1 model | Fewest non-zero coefficients within 0.01 AUC of baseline; alpha explored on log-10 scale |
| Logistic Regression + interactions | Pairwise interaction terms, no penalty |
| Logistic Regression + interaction L1 | `interaction_alpha=0.01` penalizes only interaction terms |
| GAM | Partial dependence plots for full interpretability |

### Section 3 — UCI Shuttle Dataset
All five models from Section 2 are re-run on the Shuttle dataset to compare how model behavior changes with a non-social, sensor-based dataset.

---

## 💡 Key Findings

- **COMPAS bias** — Risk decile distributions visually confirm the ProPublica finding: Black defendants receive systematically higher risk scores than White defendants.
- **Feature importance** — Across models, `priors_count` and `age` are the strongest predictors of recidivism; race features have smaller but non-zero effects in COMPAS's own scoring.
- **Sparsity** — L1 regularization effectively zeros out low-signal features; useful for producing parsimonious, auditable models.
- **Interpretability** — GAMs outperform logistic regression with interaction terms for interpretability, as partial dependence plots make per-feature relationships immediately visible.
- **Shuttle dataset** — Models behave differently on sensor data vs. recidivism data, illustrating how dataset structure affects regularization and interaction term usefulness.

---

## ⚙️ Requirements

```bash
pip install jax jaxlib numpy pandas scikit-learn matplotlib pygam
```

| Package | Purpose |
|---|---|
| `jax` / `jaxlib` | Automatic differentiation for custom gradient descent |
| `numpy` / `pandas` | Data manipulation |
| `scikit-learn` | Preprocessing, train/test split, baseline models |
| `pygam` | Generalized Additive Models with spline and factor terms |
| `matplotlib` | Weight plots and partial dependence visualizations |

---

## 🚀 Usage

1. Clone the repo and place `shuttle.csv` in the root directory
2. Install dependencies
3. Run the notebooks in order — validate first, then analyze

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt

# Step 1: Build & validate the model
jupyter notebook "Assignment_2_Linear_Models-Validate_Model.ipynb"

# Step 2: Run the full analysis
jupyter notebook "Assignment_2_Linear_Models_Analysis.ipynb"
```

> **Note:** `logistic_regression.py` and `util.py` must be in the same directory as the notebooks.

---

## 📚 References

- [ProPublica COMPAS Analysis](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- [COMPAS Dataset (GitHub)](https://github.com/propublica/compas-analysis)
- [PyGAM Documentation](https://pygam.readthedocs.io/en/latest/)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/)
- [UCI Shuttle Dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle))
- Angwin et al. (2016). *Machine Bias.* ProPublica.

---

<p align="center">Made with ⚖️ and JAX</p>
