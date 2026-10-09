# Crop Yield Prediction for AgriBoost India

Predictive regression model built to forecast agricultural yield outcomes (`quintal_per_hectare`) based on environmental factors, soil properties, and farming inputs.

---

## Tech Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline in Python to estimate crop yield:
1. **Data Preprocessing & Encoding:** Encodes categorical features (soil type, crop type, irrigation) using One-Hot Encoding (`pd.get_dummies`).
2. **Model Training:** Applies a **Linear Regression** model to train on historical agricultural records.
3. **Validation & Evaluation:** Assesses prediction accuracy using Mean Squared Error (MSE) and $R^2$ Score.
4. **Custom Prediction Pipeline:** Automatically aligns and predicts expected yield for new, sample farming inputs.

---
