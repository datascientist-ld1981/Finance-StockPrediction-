Stock Price Prediction System
This project is a Streamlit-based web application designed to predict closing stock prices of FAANG companies (Facebook, Amazon, Apple, Netflix, Google) using machine learning models. It allows users to upload a dataset, preprocess it, and train a linear regression model for stock price predictions.

Features
Upload Dataset: Upload a CSV file containing stock data for FAANG companies.
Company Selection: Select a specific company from the dataset for prediction.
Data Preprocessing: Automatically handles missing values, sorts data by date, and generates lag features for training.
Model Training: Trains a Linear Regression model on the historical closing prices.
Model Evaluation: Displays the R² score and compares actual vs predicted prices.
Visualization: Interactive line chart to compare actual and predicted stock prices.
How to Use
Clone the repository:

bash
Copy
Edit
git clone https://github.com/<your-username>/stock-price-prediction.git
cd stock-price-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
streamlit run app.py
Upload a CSV dataset with the following required columns:

date: Date of the stock price (YYYY-MM-DD).
company: Company name (e.g., Facebook, Amazon).
close: Closing stock price.
Select a company, train the model, and view the predictions along with the R² score.

Dataset Requirements
The CSV file should have the following columns:

date: The date in YYYY-MM-DD format.
company: Name of the company (e.g., Facebook, Apple, etc.).
close: The closing stock price for the corresponding date and company.
An example dataset:

date	company	close
2025-01-01	Facebook	270.50
2025-01-02	Facebook	272.30
2025-01-01	Amazon	3400.00
2025-01-02	Amazon	3450.50
Key Features of the Code
Preprocessing:

Converts the date column to datetime format.
Handles missing values using forward fill (ffill).
Sorts data by date for consistent time series analysis.
Feature Engineering:

Generates a new feature Prev_Close as a lagged value of the closing price.
Model:

Uses Linear Regression to predict the next day's closing price based on the previous day's closing price.
Evaluation:

The R² score is calculated to assess the model's accuracy.
Predictions are compared with actual values and visualized in a line chart.
Example Output
Predictions Table
Date	Actual Close	Predicted Close
2025-01-15	275.80	274.90
2025-01-16	277.50	276.40
Visualization
The app provides an interactive line chart comparing actual and predicted closing prices for the selected company.

Tools & Libraries Used
Streamlit: For building the interactive web application.
Pandas: For data manipulation and preprocessing.
NumPy: For numerical computations.
scikit-learn: For machine learning tasks, including feature scaling, linear regression, and evaluation.
Future Enhancements
Add support for more advanced machine learning models (e.g., Random Forest, XGBoost, LSTM).
Enable predictions for multiple companies simultaneously.
Incorporate additional features such as trading volume, open price, and high/low prices for improved accuracy.
Add a feature to save trained models for later use.
Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

Author
Developed by Lakshmi Devi
GitHub: datascientist-ld1981

License
This project is licensed under the MIT License.

