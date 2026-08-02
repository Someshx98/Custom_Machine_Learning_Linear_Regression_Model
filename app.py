import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import engine as en

st.set_page_config(page_title="Custom ML Engine", page_icon="📈", layout="wide")

PRIMARY = "#4C72B0"
ACCENT = "#DD8452"
MUTED = "#9AA0A6"
TEST_COLOR = "#55A868"

SEP_OPTIONS = {"Comma ( , )": ",", "Semicolon ( ; )": ";", "Tab": "\t"}


@st.cache_data
def load_data(uploaded_file, sep):
    return pd.read_csv(uploaded_file, sep=sep)


st.title("📈 Custom Machine Learning Engine")
st.caption("A linear regression model trained with hand-written gradient descent — no scikit-learn involved.")

with st.sidebar:
    st.header("1. Upload Data")
    sep_label = st.selectbox("CSV Separator", list(SEP_OPTIONS.keys()))
    separator = SEP_OPTIONS[sep_label]
    file = st.file_uploader("Upload Your Dataset", type=["csv"])

if file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.stop()

try:
    df = load_data(file, separator)
except Exception as e:
    st.error(f"Couldn't read that file with the selected separator. Details: {e}")
    st.stop()

if df.empty:
    st.error("That file doesn't contain any rows.")
    st.stop()

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) < 2:
    found = ", ".join(numeric_columns) if numeric_columns else "none"
    st.error(f"Need at least two numeric columns to run a regression (found: {found}). Double-check your separator is correct.")
    st.stop()

st.subheader("Data Preview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Numeric Columns", len(numeric_columns))
st.dataframe(df.head(10), use_container_width=True)

with st.sidebar:
    st.header("2. Choose Columns")
    X = st.selectbox("Feature (X)", numeric_columns, index=0)
    Y = st.selectbox("Target (Y)", numeric_columns, index=1)

    st.header("3. Hyperparameters")
    iterations = st.slider("Training Iterations", min_value=100, max_value=5000, value=1000, step=100)
    learning_rate = st.slider(
        "Learning Rate (α)", min_value=0.001, max_value=1.0, value=0.100, step=0.001,
        format="%.3f",
        help="Data is standardized internally, so this range works well regardless of your data's original scale.",
    )

    n_rows = len(df)
    can_split = n_rows >= 10
    use_split = st.checkbox(
        "Hold out a test set (recommended)", value=can_split, disabled=not can_split,
        help="Evaluate the model on rows it never trained on, for an honest accuracy score.",
    )
    if not can_split:
        st.caption("Dataset is small — training and evaluating on all rows.")
    test_size = 0.2
    if use_split and can_split:
        test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100

    train_clicked = st.button("🚀 Train Model", use_container_width=True, type="primary")

if X == Y:
    st.warning("X and Y are the same column, so the model will trivially learn y = x.")

if train_clicked:
    clean = df[[X, Y]].dropna()
    if clean.empty:
        st.error(f"'{X}' and '{Y}' have no overlapping non-missing values.")
        st.stop()

    x_data = clean[X].astype(float).values
    y_data = clean[Y].astype(float).values

    with st.spinner("Training your model..."):
        if use_split and can_split and len(x_data) >= 10:
            x_train, x_test, y_train, y_test = en.train_test_split(x_data, y_data, test_size=test_size)
        else:
            x_train, y_train = x_data, y_data
            x_test, y_test = x_data, y_data

        m, b, cost_history = en.train(x_train, y_train, 0.0, 0.0, learning_rate, iterations)

    if len(cost_history) < iterations:
        st.warning(f"Training stopped early at iteration {len(cost_history)} because it became unstable — try a lower learning rate for a full run.")

    train_r2 = en.calculate_r2(y_train, en.predict(x_train, m, b))
    test_r2 = en.calculate_r2(y_test, en.predict(x_test, m, b))

    st.session_state["results"] = {
        "m": m, "b": b, "cost_history": cost_history,
        "train_r2": train_r2, "test_r2": test_r2,
        "x_data": x_data, "y_data": y_data,
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test,
        "X": X, "Y": Y,
        "used_split": bool(use_split and can_split and len(x_data) >= 10),
    }

results = st.session_state.get("results")

if results and results["X"] == X and results["Y"] == Y:
    m, b = results["m"], results["b"]

    st.subheader("Results")
    col_m, col_b, col_train_r2, col_test_r2 = st.columns(4)
    col_m.metric("Slope (m)", f"{m:.4f}")
    col_b.metric("Intercept (b)", f"{b:.4f}")
    col_train_r2.metric("Train R²", f"{results['train_r2']:.4f}")
    if results["used_split"]:
        col_test_r2.metric("Test R²", f"{results['test_r2']:.4f}")
    else:
        col_test_r2.metric("Test R²", "—", help="No held-out test set was used for this run.")

    st.caption(f"Learned equation: **y = {m:.4f} · x + {b:.4f}**")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Cost History")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        ax1.plot(results["cost_history"], color=PRIMARY, linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost (MSE)")
        ax1.set_title("Error Dropping Over Time")
        ax1.grid(alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_right:
        st.markdown("#### Model Fit")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        if results["used_split"]:
            ax2.scatter(results["x_train"], results["y_train"], color=MUTED, alpha=0.6, s=25, label="Train Data")
            ax2.scatter(results["x_test"], results["y_test"], color=TEST_COLOR, alpha=0.85, s=25, label="Test Data")
        else:
            ax2.scatter(results["x_data"], results["y_data"], color=MUTED, alpha=0.6, s=25, label="Data")

        # Use an evenly-spaced line (not the raw, possibly-unsorted x_data)
        # so the fit line is always a clean straight line, never a jagged
        # zig-zag from connecting points out of order.
        x_line = np.linspace(results["x_data"].min(), results["x_data"].max(), 100)
        y_line = en.predict(x_line, m, b)
        ax2.plot(x_line, y_line, color=ACCENT, linewidth=2.5, label="Best Fit")

        ax2.set_xlabel(results["X"])
        ax2.set_ylabel(results["Y"])
        ax2.set_title("Actual vs. Predicted")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)
        plt.close(fig2)

    with st.expander("📥 Download Results"):
        out_df = pd.DataFrame({
            results["X"]: results["x_data"],
            results["Y"]: results["y_data"],
            "Predicted": en.predict(results["x_data"], m, b),
        })
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    st.success("Model trained successfully!")
elif results:
    st.info("Column selection changed since the last training run — click **Train Model** to retrain.")