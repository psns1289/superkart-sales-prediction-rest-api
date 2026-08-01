
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained sales prediction model
model = joblib.load("superkart_sales_model_v1_0.joblib")

# List of perishable product types (used for feature engineering in batch prediction)
perishables = [
    "Dairy",
    "Meat",
    "Fruits and Vegetables",
    "Breakfast",
    "Breads",
    "Seafood",
]

# The exact set of features expected by the model pipeline (order matters)
model_features = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Id_char',
    'Store_Age_Years',
    'Product_Type_Category',
]


def add_engineered_features(df):
    """Recreate the engineered features from raw SuperKart-style columns."""
    df = df.copy()
    # Clean the sugar content column
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace("reg", "Regular")
    # Two-letter product id prefix (FD / DR / NC)
    if "Product_Id_char" not in df.columns and "Product_Id" in df.columns:
        df["Product_Id_char"] = df["Product_Id"].str[:2]
    # Store age in years
    if "Store_Age_Years" not in df.columns and "Store_Establishment_Year" in df.columns:
        df["Store_Age_Years"] = 2025 - df["Store_Establishment_Year"]
    # Perishable vs non-perishable grouping
    if "Product_Type_Category" not in df.columns and "Product_Type" in df.columns:
        df["Product_Type_Category"] = df["Product_Type"].apply(
            lambda x: "Perishables" if x in perishables else "Non Perishables"
        )
    return df


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


# Define an endpoint to predict sales for a single product-store record
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json()

    # Extract the relevant features from the input data. The order of the column names matters.
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a sales prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': round(float(prediction), 2)})


# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Recreate the engineered features so a raw SuperKart-style CSV works
    engineered = add_engineered_features(input_data)

    # Select only the columns the model was trained on
    X = engineered[model_features]

    # Make predictions for all rows
    predictions = [round(float(p), 2) for p in model.predict(X).tolist()]

    # Build the response keyed by Product_Id when available, else by row index
    if "Product_Id" in input_data.columns:
        keys = input_data["Product_Id"].tolist()
    else:
        keys = list(range(len(predictions)))
    output_dict = dict(zip(keys, predictions))

    # Return the predictions dictionary as a JSON response
    return output_dict


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
