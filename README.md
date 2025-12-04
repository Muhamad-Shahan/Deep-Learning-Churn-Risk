# 🛡️ Deep Learning Churn Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deep-learning-churn-risk-gmybgricffrketl39uy5pv.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

## 📄 Abstract
Customer attrition (churn) is a critical metric for banking institutions. Traditional statistical models often struggle to capture complex, non-linear patterns in customer behavior. 

This project implements a **Deep Learning** solution using an **Artificial Neural Network (ANN)** to predict customer churn probability with high precision. The model was trained on 10,000 banking records and deployed as a real-time interactive dashboard.

## 🚀 Live Demo
> **[Click here to launch the ChurnGuard AI Dashboard](https://deep-learning-churn-risk-gmybgricffrketl39uy5pv.streamlit.app/)**

## 🧠 Model Architecture
We utilized a Multi-Layer Perceptron (MLP) built with **TensorFlow/Keras**:
* **Input Layer:** 128 Neurons (ReLU activation) + Batch Normalization
* **Hidden Layers:** Dropout layers (0.3) were added to prevent overfitting.
* **Output Layer:** 1 Neuron (Sigmoid activation) for probability output (0-1).
* **Optimizer:** Adam (Learning Rate = 0.01).

## 📊 Performance
The model was optimized using **Keras Tuner** to find the ideal hyperparameters.
* **Accuracy:** ~93.5%

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Deployment:** Streamlit Cloud
* **Tuning:** Keras Tuner

## 💻 Installation
To run this locally, you need TensorFlow installed.

```bash
git clone [https://github.com/Muhammad-Shahan/Deep-Learning-Churn-Risk.git](https://github.com/Muhammad-Shahan/Deep-Learning-Churn-Risk.git)
pip install -r requirements.txt
streamlit run app.py
