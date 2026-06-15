# 🎯 Career Success Prediction — Leakage-Free ML & NLP Pipeline

This project is a **full-stack machine learning pipeline** developed for the *Datathon 2026 (Kaggle-style competition)*.  
It predicts `career_success_score` (0–100) using multimodal student data including tabular features, structured categorical data, and natural language mentor feedback.

The system is designed not just for prediction, but for **robust generalization under covariate shift, leakage-free training, and interpretable feature engineering**.

---

# 🏆 Competition Context

This project was developed for **Datathon 2026**, a data science competition organized by BTK Akademi in collaboration with Google and Girvak.

The dataset includes:
- Technical skill assessments
- Academic and project performance
- Internship and experience data
- Social/soft skills
- Mentor feedback (natural language)

Evaluation metric: **Mean Squared Error (MSE)**

---

# 🧠 Architecture Overview

```

```
                      ┌──────────────────────┐
                      │  Raw Tabular Data    │
                      └─────────┬────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
```

Tabular Features           NLP Features                Embeddings
(skills, stats,            (TF-IDF + SVD,             (BERT + PCA)
interactions,              keyword features)           + clustering)
engineered metrics)
│                           │                           │
└──────────────┬────────────┴────────────┬────────────┘
▼                         ▼
Leakage-Free K-Fold Pipeline (5-Fold CV)
│
┌────────────┴────────────┐
▼                         ▼
CatBoost Regressor      Weighted CatBoost (shift-aware)
│
└────────────┬────────────┘
▼
LightGBM Classifier (edge-case detection)
▼
Final Ensemble Output

```
---
# ⚙️ Key Innovations

## 🔒 1. Leakage-Free Full Pipeline
- TF-IDF, SVD, scaling, clustering **all fitted inside each fold**
- No information leakage between train/test or folds
- Realistic evaluation using strict K-Fold CV

---

## 📉 2. Covariate Shift Handling
- Strong distribution shift between train and test years (2019–2026)
- Sample weighting based on **test/train year ratio**
- Clipped importance weights to prevent dominance

---

## 🧠 3. Multimodal Feature Fusion
- Tabular engineered features (174+ features)
- NLP signals:
  - TF-IDF + SVD (semantic compression)
  - Keyword sentiment features
- Dense embeddings:
  - Turkish BERT embeddings
  - PCA compression (768 → 50)
  - KMeans clustering + target encoding

---

## ⚙️ 4. Advanced Feature Engineering
- Feature interactions (multiplicative & polynomial)
- Experience scoring system
- Visibility score (GitHub + LinkedIn + portfolio)
- Temporal interaction features (year-aware learning)
- Readability score for Turkish text

---

## 📊 5. Robust Evaluation Strategy
- OOF RMSE (overall performance)
- Segment-based evaluation:
  - 2019–2024 vs 2025–2026
- Expected test MSE estimation via distribution weighting
- Threshold tuning for classifier integration

---

# 📈 Results Summary

- Stable RMSE across folds (~8.7–9.0 range)
- Strong generalization gap analysis across time
- Weighted vs unweighted comparison for shift robustness
- Final submission selected via two-stage blending

---

# 🚀 What I Would Improve Next

Even though the pipeline is strong, future improvements could include:

## 1. Transformer-based tabular models
- FT-Transformer or TabTransformer instead of manual feature engineering

## 2. Better temporal modeling
- Explicit time-series formulation for application_year drift

## 3. Stacking / Meta-learner
- Replace simple averaging with:
  - Ridge stacking
  - Neural meta-model

## 4. Better NLP integration
- Fine-tuned Turkish BERT instead of frozen embeddings
- Sentence-level sentiment modeling

## 5. Automated feature selection
- SHAP-based pruning
- Boruta / permutation importance filtering

---

# 📌 Key Takeaway

This project is not just a Kaggle solution.

It is a **leakage-free, shift-aware, multimodal machine learning system** designed to model complex human career outcomes from heterogeneous data sources.

---

# 🏁 Tech Stack

- Python (Pandas, NumPy, Scikit-learn)
- CatBoost / LightGBM
- SentenceTransformers (BERT)
- TF-IDF + SVD
- KMeans clustering
- PySR (Symbolic Regression)
```

---
