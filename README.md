# 🚀 Custom Linear Regression Engine

An interactive Machine Learning web application powered by a custom Gradient Descent algorithm, built entirely from scratch using NumPy, Pandas, and Streamlit.

## 🧠 The Philosophy
Most machine learning tutorials start by importing `sklearn.linear_model.LinearRegression`. While efficient, this acts as a "black box" that hides the fundamental mathematics of how artificial intelligence actually learns.

This project bypasses Scikit-learn entirely. The backend engine is built from the ground up to demonstrate a deep understanding of core Python logic, calculus (partial derivatives), and the Gradient Descent optimization loop that powers modern neural networks — without relying on Object-Oriented shortcuts.

---

## 📐 The Mathematics (Under the Hood)

The engine (`engine.py`) learns iteratively by optimizing the slope ($m$) and y-intercept ($b$) of a line through the following mathematical steps.

### 1. Feature Standardization
Before training starts, the input feature is rescaled to have a mean of 0 and a standard deviation of 1. Gradient descent is extremely sensitive to the scale of its input — without this step, the *same* learning rate that converges nicely on one dataset can diverge into `NaN` on another, purely because the numbers happen to be larger.

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation of $x$.

### 2. The Prediction Function
Generates a prediction ($\hat{y}$) for a given input ($x$) based on the current weights.

$$\hat{y} = mx + b$$

### 3. The Cost Function (Mean Squared Error)
Calculates the average squared difference between the actual data ($y$) and the model's predictions ($\hat{y}$). The engine's goal is to minimize this number.

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### 4. Gradient Descent (Partial Derivatives)
Calculates the gradient of the Cost Function with respect to both $m$ and $b$ — computed as vectorized NumPy array operations rather than a row-by-row Python loop, for both speed and clarity — to determine the direction of steepest descent.

$$D_m = \frac{-2}{N} \sum_{i=1}^{N} x_i(y_i - \hat{y}_i)$$

$$D_b = \frac{-2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)$$

### 5. The Weight Update Rule
Adjusts the weights by taking a small step (controlled by the Learning Rate, $\alpha$) in the opposite direction of the gradient.

$$m_{new} = m - (\alpha \cdot D_m)$$

$$b_{new} = b - (\alpha \cdot D_b)$$

Steps 2–5 repeat for the configured number of iterations, entirely in standardized space. If an update ever produces a non-finite cost (the learning rate was too high), training stops early and keeps the last stable weights instead of returning `NaN`.

### 6. Rescaling Back to Original Units
Once training finishes, the learned weights are converted from standardized space back into the original units of the data, so predictions and the plotted fit line make sense against the raw, uploaded values:

$$m_{original} = \frac{m}{\sigma}, \qquad b_{original} = b - \frac{m \cdot \mu}{\sigma}$$

### 7. Model Evaluation ($R^2$)
The model is graded with the $R^2$ (coefficient of determination) score. Whenever the dataset is large enough, this is computed on a held-out test split rather than the training data itself, so the reported accuracy reflects how well the model generalizes — not just how well it memorized what it already saw.

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 🏗️ Project Architecture

* **`engine.py`** — the pure NumPy mathematical backend. Contains:
  * `standardize()` — rescales the input feature for stable training
  * `predict()`, `calculate_cost()`, `update_weights()` — the core prediction, cost, and gradient functions, fully vectorized
  * `train()` — runs the gradient descent loop, standardizing internally and rescaling the weights back before returning them
  * `train_test_split()` — a hand-written splitter for honest, held-out evaluation
  * `calculate_r2()` — the $R^2$ accuracy grader, guarded against divide-by-zero on a constant target
* **`app.py`** — the interactive Streamlit front-end. Handles CSV upload and delimiter selection, cleaning of missing values in the selected columns, numeric-column validation, hyperparameter tuning via sidebar sliders, an optional train/test split toggle, Matplotlib visualizations of cost and fit, and a CSV export of the model's predictions.

---

## ✨ Features

* Gradient descent implemented from scratch — no `sklearn`, `torch`, or `statsmodels`
* Automatic feature standardization, so training is stable regardless of your data's original scale
* Optional train/test split for an honest, held-out accuracy score
* Vectorized NumPy training loop for fast performance, even on larger datasets
* Friendly error handling for bad separators, missing numeric columns, or empty files
* Interactive sidebar for uploading data, choosing columns, and tuning hyperparameters
* Cost-history and model-fit plots, plus a downloadable CSV of predictions

---

## 💻 How to Run Locally

### Prerequisites
Make sure you have Python installed, then open your terminal and install the required dependencies:

```bash
pip install numpy pandas matplotlib streamlit
```

### Run the app
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`) and upload a CSV to get started.