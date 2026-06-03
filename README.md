# 🚀 Custom Linear Regression Engine

An interactive Machine Learning web application powered by a custom Gradient Descent algorithm, built entirely from scratch using only NumPy and Streamlit.

## 🧠 The Philosophy
Most machine learning tutorials start by importing `sklearn.linear_model.LinearRegression`. While efficient, this acts as a "black box" that hides the fundamental mathematics of how artificial intelligence actually learns. 

This project bypasses Scikit-learn entirely. The backend engine was built from the ground up to demonstrate a deep understanding of core Python logic, calculus (partial derivatives), and the Gradient Descent optimization loop that powers modern neural networks without relying on Object-Oriented shortcuts.

---

## 📐 The Mathematics (Under the Hood)

The engine (`engine.py`) learns iteratively by optimizing the slope ($m$) and y-intercept ($b$) of a line through the following mathematical steps:

### 1. The Prediction Function
Generates a prediction ($\hat{y}$) for a given input ($x$) based on current weights.

$$\hat{y} = mx + b$$

### 2. The Cost Function (Mean Squared Error)
Calculates the average squared difference between the actual data ($y$) and the model's predictions ($\hat{y}$). The engine's goal is to minimize this number.

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### 3. Gradient Descent (Partial Derivatives)
Calculates the gradient (slope) of the Cost Function with respect to both $m$ and $b$ to determine the direction of the steepest descent.

$$D_m = \frac{-2}{N} \sum_{i=1}^{N} x_i(y_i - \hat{y}_i)$$

$$D_b = \frac{-2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)$$

### 4. The Weight Update Rule
Adjusts the weights by taking a small step (controlled by the Learning Rate, $\alpha$) in the opposite direction of the gradient.

$$m_{new} = m - (\alpha \cdot D_m)$$

$$b_{new} = b - (\alpha \cdot D_b)$$

---

## 🏗️ Project Architecture

* **`engine.py`**: The pure NumPy mathematical backend. Contains the custom `train()`, `update_weights()`, and `calculate_cost()` loop, as well as an $R^2$ accuracy grader.
* **`app.py`**: The interactive Streamlit front-end. Handles CSV file uploading, data cleaning (handling delimiters and removing `NaN` values), hyperparameter tuning via UI sliders, and Matplotlib data visualization.

---

## 💻 How to Run Locally

### Prerequisites
Make sure you have Python installed on your Windows machine, then open your terminal and install the required dependencies:

```bash
pip install numpy pandas matplotlib streamlit