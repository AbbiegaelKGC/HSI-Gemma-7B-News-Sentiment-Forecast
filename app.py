import streamlit as st
import pandas as pd
import joblib
import datetime as dt
import xgboost as xgb
import matplotlib.pyplot as plt

def load_data(file_path):
    # Load CSV data into a pandas DataFrame
    return pd.read_csv(file_path)

def main():
    st.title("HK Index Price Prediction")

    # File paths
    test_file_path = "test_data.csv"
    train_file_path = "train_data.csv"
    hsi_file_path = "hsi.csv"
    model_file_path = "hsi_xgboost_model1.bin"

    # Load the model
    model = joblib.load(model_file_path)

    # Load the data
    test_data = load_data(test_file_path)
    train_data = load_data(train_file_path)
    hsi_data = load_data(hsi_file_path)

    # Convert 'Date' to datetime format for filtering
    test_data["Date"] = pd.to_datetime(test_data["Date"])

    # Define date limits
    min_date = test_data["Date"].min()
    max_date = pd.Timestamp("2024-04-26")

    # Main page - date selection
    selected_date = st.date_input("Select a Date", value=min_date, min_value=min_date, max_value=max_date)

    # Filter test data to dates up to and including selected date
    selected_data = test_data[test_data["Date"] <= pd.Timestamp(selected_date)]

    if selected_data.empty:
        st.write("No data available up to the selected date.")
        return

    # Prepare data for prediction
    X_test = selected_data.drop(columns=["Date", "Close"])

    # Convert to DMatrix
    dmatrix_test = xgb.DMatrix(X_test)

    # Make predictions
    y_pred = model.predict(dmatrix_test)

    # Convert predictions into a DataFrame
    predictions_df = pd.DataFrame({
        "Date": selected_data["Date"],
        "Predicted Close": y_pred
    })

    # Merge with actual prices for comparison
    results = pd.merge(predictions_df, selected_data[['Date', 'Close']], on='Date')

    # Display results in a table
    st.write("Predicted vs. Actual Closing Prices:")
    st.dataframe(results)

    # Plot the comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(results["Date"], results["Close"], label="Actual Closing Price", color="green")
    ax.plot(results["Date"], results["Predicted Close"], label="Predicted Closing Price", color="red")

    # Formatting
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)

if __name__ == "__main__":
    main()
