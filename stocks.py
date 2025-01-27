import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Streamlit UI
st.title("Stock Price Prediction System")
st.write("Predict closing stock prices using machine learning models.")

# Upload FAANG dataset
uploaded_file = st.file_uploader("Upload FAANG Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Check the columns
    required_columns = {'date', 'company', 'close'}
    if not required_columns.issubset(data.columns.str.lower()):
        st.error(f"Dataset must include the following columns: {required_columns}")
    else:
        data.columns = data.columns.str.lower()  # Make columns lowercase for consistency
        
        # Show first few rows of the data
        st.write("Preview of the Dataset:", data.head())

        # Select company from dropdown
        selected_company = st.selectbox("Select Company", data['company'].unique())

        # Filter data for selected company
        company_data = data[data['company'] == selected_company]

        # Check if there are enough rows for training
        if company_data.shape[0] < 10:
            st.warning(f"Not enough data for {selected_company} to train the model after preprocessing. "
                       "Please select a different company or upload more data.")
        else:
            # Preprocess data: Convert 'date' column to datetime and sort
            company_data['date'] = pd.to_datetime(company_data['date'])
            company_data = company_data.sort_values('date')

            # Fill missing values in 'close' with the previous available value (if any)
            company_data['close'] = company_data['close'].fillna(method='ffill')

            # Check if data is still available after filling
            if company_data['close'].isna().sum() > 0:
                st.warning("There are still missing values after preprocessing, please upload more complete data.")
            else:
                # Prepare features and target
                company_data['Prev_Close'] = company_data['close'].shift(1)
                company_data = company_data.dropna(subset=['Prev_Close'])

                # Features (X) and target (y)
                X = company_data[['Prev_Close']]
                y = company_data['close']

                # Ensure at least 20 samples exist for splitting into train/test
                if len(X) < 20:
                    st.warning(f"The dataset for {selected_company} has fewer than 20 rows, which may lead to poor performance. "
                               "Consider uploading more data.")
                else:
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                    # Scale features
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train a Linear Regression model
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test_scaled)

                    # Calculate R² score
                    r2 = r2_score(y_test, y_pred)

                    # Display R² score and predictions
                    st.write(f"### Model Performance for {selected_company}")
                    st.write(f"**R² Score**: {r2:.4f}")

                    # Display predictions
                    predictions = pd.DataFrame({
                        'Date': company_data['date'].iloc[-len(y_test):].values,
                        'Actual Close': y_test.values,
                        'Predicted Close': y_pred
                    })
                    st.write("### Predictions:")
                    st.write(predictions)

                    # Plot actual vs predicted values
                    st.line_chart(predictions.set_index('Date')[['Actual Close', 'Predicted Close']])
