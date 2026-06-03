import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import engine as en

st.title("Custom Machine Learning Engine")

delm = st.selectbox("Your CSV Separator: ", [",", ";", "\\t"])

file = st.file_uploader("Upload Your Dataset.", type = ["csv"])

if file is not None:
    f = pd.read_csv(file, sep = delm)
    f = f.dropna()
    st.write("### The First 10 roes of your Dataset")
    st.dataframe(f.head(10))
    
    numeric_columns = f.select_dtypes(include = ['float64', 'int64']).columns
    
    X = st.selectbox("Select the X-Set", f.columns)
    Y = st.selectbox("Select the Y-Set", f.columns)
    
    iterations = st.slider("No of Iterations: ", min_value = 100, max_value = 5000, value = 1000)
    learning_rate = st.slider("Select Your learning rate: ", min_value = 0.0001, max_value = 0.1, value = 0.01, step = 0.0001)
    
    x_data = f[X].astype(float).values
    y_data = f[Y].astype(float).values
    
    if st.button("Train Model"):
        
        # Engine Strating with the starting points 0.0 and 0.0
        m, b, cost_history = en.train(x_data, y_data, 0.0, 0.0, learning_rate, iterations)
        
        # Grading the Model
        predictions = en.predict(x_data, m, b)
        score = en.calculate_r2(y_data, predictions)
        st.success(f"Model Accuracy (R²): {score}")
        
        # Plotting the results
        st.write("### Cost History (Error Dropping over time)")
        fig1, ax1 = plt.subplots()
        ax1.plot(cost_history, color="blue")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Cost (MSE)")
        st.pyplot(fig1)
        
        st.write("### Final Model Fit")
        fig2, ax2 = plt.subplots()
        ax2.scatter(x_data, y_data, color="gray", label="Real Data") 
        ax2.plot(x_data, predictions, color="red", linewidth=2, label="Engine's Best Fit") 
        ax2.legend()
        st.pyplot(fig2)