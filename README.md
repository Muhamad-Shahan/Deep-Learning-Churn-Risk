# 🛡️ Deep Learning Churn Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deep-learning-churn-risk-gmybgricffrketl39uy5pv.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Tuner-red.svg)](https://keras.io/keras_tuner/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

## 🚀 Project Overview
This is an end-to-end **Deep Learning** solution designed to predict bank customer churn. 

Instead of relying on standard logistic regression or tree-based models, I implemented a **Deep Neural Network (ANN) using TensorFlow & Keras** to capture complex, non-linear patterns in customer behavior. The system is deployed as a live web application, allowing stakeholders to assess risk in real-time.

> **[🔴 Click here to view the Live Dashboard](https://deep-learning-churn-risk-gmybgricffrketl39uy5pv.streamlit.app/)**

## 💡 Key Technical Achievements
This project goes beyond basic model training by addressing real-world data challenges:

* **🧠 Deep Learning Architecture:** Built a Multi-Layer Perceptron (MLP) with **TensorFlow** to outperform baseline statistical methods.
* **⚖️ Handling Class Imbalance:** Applied **Computed Class Weights** during training. This forced the model to pay 4x more attention to the minority class (Churners), preventing the common "accuracy paradox" where models just guess "Stay" for everyone.
* **🎛️ Automated Hyperparameter Tuning:** Utilized **Keras Tuner** (Hyperband algorithm) to mathematically determine the optimal number of neurons (128), learning rate (0.01), and dropout rates.
* **📉 High-Recall Optimization:** Focused on maximizing **Recall (0.75)** for the churn class, ensuring that 75% of at-risk customers are successfully identified.

## 📊 Model Performance
After extensive tuning and evaluation on 10,000 banking records:

| Metric | Score | Significance |
|--------|-------|--------------|
| **Accuracy** | **89.5%** | High overall correctness on unseen data. |

## 🛠️ Tech Stack & Workflow
1.  **Data Processing:** Pandas & NumPy for cleaning; Scikit-Learn for One-Hot Encoding and Standard Scaling.
2.  **Modeling:** TensorFlow/Keras (Sequential API).
    * *Input Layer:* 128 Neurons (ReLU) + Batch Normalization.
    * *Regularization:* Dropout layers (0.3) to prevent overfitting.
    * *Output:* Sigmoid activation for probability scoring (0-1).
3.  **Deployment:** Streamlit Cloud for the frontend interface.

## 💻 Installation (Run Locally)
To run this Deep Learning app on your own machine, you need TensorFlow installed.

```bash
# 1. Clone the repo
git clone [https://github.com/Muhammad-Shahan/Deep-Learning-Churn-Risk.git](https://github.com/Muhammad-Shahan/Deep-Learning-Churn-Risk.git)

# 2. Install dependencies (includes TensorFlow)
pip install -r requirements.txt

# 3. Launch the App
streamlit run app.py
