from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Load and preprocess data
df = pd.read_csv("media/traffic_route_prediction.csv")

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df.drop(columns=["Date"], inplace=True)

df["Time"] = pd.to_datetime(df["Time"], format='%H:%M:%S')
df["Hour"] = df["Time"].dt.hour
df["Minute"] = df["Time"].dt.minute
df["Second"] = df["Time"].dt.second
df.drop(columns=["Time"], inplace=True)

# Encode categorical variables
categorical_columns = ["Source", "Destination", "Congestion_Level", "Weather", "Day_Type", "Suggested_Route"]
label_encoders = {}

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

    # Add "Unknown" as a default category for unseen labels
    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "Unknown")

# Standardize numerical features
numerical_columns = ["Year", "Month", "Day", "Hour", "Minute", "Second", "Lat_Source", "Lon_Source", "Lat_Destination", "Lon_Destination"]
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Prepare train-test split
X = df.drop(columns=["Vehicle_Count", "Suggested_Route"])
y_vehicle = df["Vehicle_Count"]
y_route = df["Suggested_Route"]

X_train, X_test, y_train_vehicle, y_test_vehicle = train_test_split(X, y_vehicle, test_size=0.2, random_state=42)
X_train_route, X_test_route, y_train_route, y_test_route = train_test_split(X, y_route, test_size=0.2, random_state=42)

# Train models
vehicle_model = RandomForestRegressor(n_estimators=100)
vehicle_model.fit(X_train, y_train_vehicle)

route_model = RandomForestClassifier(n_estimators=100, random_state=42)
route_model.fit(X_train_route, y_train_route)

def predict(request):
    if request.method == "POST":
        input_data = {
            "Source": request.POST["source"],
            "Destination": request.POST["destination"],
            "Year": int(request.POST["year"]),
            "Month": int(request.POST["month"]),
            "Day": int(request.POST["day"]),
            "Hour": int(request.POST["hour"]),
            "Minute": int(request.POST["minute"]),
            "Second": int(request.POST["second"]),
            "Congestion_Level": request.POST["congestion_level"],
            "Weather": request.POST["weather"],
            "Day_Type": request.POST["day_type"],
            "Lat_Source": float(request.POST["lat_source"]),
            "Lon_Source": float(request.POST["lon_source"]),
            "Lat_Destination": float(request.POST["lat_destination"]),
            "Lon_Destination": float(request.POST["lon_destination"])
        }

        # Encode categorical variables (Exclude 'Suggested_Route' from input)
        for col in categorical_columns:
            if col == "Suggested_Route":
                continue  # Skip Suggested_Route during input processing

            if input_data[col] in label_encoders[col].classes_:
                input_data[col] = label_encoders[col].transform([input_data[col]])[0]
            else:
                print(f"⚠ Warning: Unseen category '{input_data[col]}' in {col}. Assigning 'Unknown'.")
                input_data[col] = label_encoders[col].transform(["Unknown"])[0]

        # Normalize numerical values
        input_df = pd.DataFrame([input_data])

        # Standardize numerical features
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

        # Reorder the columns to match the training data
        input_df = input_df[X.columns]

        # Predict
        predicted_vehicle_count = vehicle_model.predict(input_df)[0]
        predicted_route = route_model.predict(input_df)[0]

        predicted_route = int(predicted_route)  # Convert to integer

        # Assign route names
        route_mapping = {
            1: "Outer Ring Road",
            2: "PV Narasimha Rao Expressway",
            3: "Necklace Road",
            4: "Rajiv Gandhi Airport Road",
            5: "Mehdipatnam Expressway"
        }

        predicted_route_name = route_mapping.get(predicted_route, "Unknown Route")

        return render(request, "prediction_result.html", {
            "vehicle_count": predicted_vehicle_count,
            "suggested_route": predicted_route_name
        })

    return render(request, "trafic_input_form.html")

def home(request):
    return render(request, "homepage.html")


import pandas as pd

def view_dataset(request):
    # Load the dataset
    df = pd.read_csv("media/traffic_route_prediction.csv")

    # Drop unwanted columns
    columns_to_drop = ["Vehicle_Count", "Suggested_Route"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Show only first 300 rows
    df = df.head(300)

    # Convert dataframe to HTML
    dataset_html = df.to_html(classes='table table-bordered table-hover table-sm', index=False)

    return render(request, "view_dataset.html", {"dataset_html": dataset_html})



from django.shortcuts import render
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def train_model(request):
    # Train vehicle model
    vehicle_model = RandomForestRegressor(n_estimators=100)
    vehicle_model.fit(X_train, y_train_vehicle)
    
    # Train route model
    route_model = RandomForestClassifier(n_estimators=100, random_state=42)
    route_model.fit(X_train_route, y_train_route)

    # Predict on the test set
    y_pred_vehicle = vehicle_model.predict(X_test)
    y_pred_route = route_model.predict(X_test_route)

    # Metrics for vehicle model
    mae_vehicle = mean_absolute_error(y_test_vehicle, y_pred_vehicle)
    mse_vehicle = mean_squared_error(y_test_vehicle, y_pred_vehicle)
    r2_vehicle = r2_score(y_test_vehicle, y_pred_vehicle)

    # Metrics for route model
    accuracy_route = accuracy_score(y_test_route, y_pred_route)

    # Generate graphs
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Vehicle Count Prediction Graph
    ax[0].scatter(y_test_vehicle, y_pred_vehicle, color='blue', label='Predicted vs Actual')
    ax[0].plot([y_test_vehicle.min(), y_test_vehicle.max()], [y_test_vehicle.min(), y_test_vehicle.max()], color='red', lw=2)
    ax[0].set_xlabel("Actual Vehicle Count")
    ax[0].set_ylabel("Predicted Vehicle Count")
    ax[0].set_title("Vehicle Count Prediction")

    # Route Prediction Accuracy
    ax[1].bar(["Route Model Accuracy"], [accuracy_route], color='green')
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Route Prediction Accuracy")

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Pass the metrics and image data to the template
    context = {
        "mae_vehicle": mae_vehicle,
        "mse_vehicle": mse_vehicle,
        "r2_vehicle": r2_vehicle,
        "accuracy_route": accuracy_route,
        "img_data": img_data
    }

    return render(request, "model_training_result.html", context)
