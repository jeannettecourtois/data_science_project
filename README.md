# Prostate Cancer Data Analysis  
## PCA and Linear Regression using Object-Oriented Python

## 📌 Project Overview

This project is part of the **ISEP Data Science course (Academic Year 2025–2026)**.  
The goal is to analyze a real medical dataset related to **prostate cancer** using:

- **Descriptive statistics**
- **Principal Component Analysis (PCA)**
- **Simple and Multiple Linear Regression**
- **Feature selection using Best Subset Selection**

The entire project is implemented in **Python**, following **Object-Oriented Programming (OOP)** principles to ensure modularity, clarity, and reusability :contentReference[oaicite:0]{index=0}.

---

## 📂 Dataset Description

The dataset `prostate.txt` contains medical information about patients who underwent **radical prostatectomy**.

### Variables

| Variable | Description |
|--------|------------|
| `vol` | Volume of cancer |
| `wht` | Prostate weight |
| `age` | Patient age |
| `bh` | Quantity of benign hyperplasia |
| `pc` | Capsular penetration |
| `psa` | Prostate-specific antigen level |

⚠️ A **logarithmic transformation** is applied to all variables except `age`.  
Transformed variables are prefixed with `l` (e.g. `lvol`, `lpsa`).  
All analyses are performed on the transformed data.

---

## 🧠 Objectives

1. Perform **exploratory data analysis** and descriptive statistics  
2. Analyze correlations and visualize relationships  
3. Apply **PCA** and interpret:
   - Variance explained
   - Loadings
   - Scores and correlation circle
4. Build **simple linear regression** models
5. Perform **multiple linear regression** with feature selection
6. Evaluate models using:
   - \( R^2 \)
   - Adjusted \( \bar{R}^2 \)
   - Hypothesis testing
7. Make predictions for new patient profiles

---

## 🏗️ Project Structure (OOP Design)

```text
├── data/
│   └── prostate.txt
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── descriptive_analysis.py
│   ├── pca_analysis.py
│   ├── regression.py
│   └── model_selection.py
├── main.py
├── requirements.txt
└── README.md

