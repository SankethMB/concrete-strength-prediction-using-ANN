# 🧱 Prediction of Concrete Compressive Strength using ANN

This project uses an **Artificial Neural Network (ANN)** to predict the **compressive strength of concrete** based on its mix composition. It uses **Yeh’s UCI dataset**, applies preprocessing, trains an ANN model, and includes **SHAP explainability**.

## 📊 Dataset

* Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
* Contains 1030 instances of concrete mixes.
* Features include:

  * Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer,
  * Coarse Aggregate, Fine Aggregate, Age
  * Target: Compressive strength (MPa)

## ⚙️ Technologies Used

* Python 🐍
* TensorFlow / Keras
* Scikit-learn
* SHAP (for model explainability)
* Joblib (to save model artifacts)

## 🧠 Model Overview

* ANN architecture:

  * Dense layers: \[256, 128, 64]
  * Activation: ReLU
  * Dropout regularization
 * Early stopping to prevent overfitting
 * Evaluation metrics:

  * R² Score
  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)


## 📊 Explainability with SHAP

This project uses **SHAP (SHapley Additive exPlanations)** to interpret the ANN model and identify the most influential features affecting concrete strength prediction.


## 💡 Credits

* Dataset by: I-Cheng Yeh (1998)
* Tools: TensorFlow, SHAP, Scikit-learn


