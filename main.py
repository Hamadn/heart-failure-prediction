import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and examine the data
dataset = pd.read_csv("./heart.csv")

# Create figure for categorical features distribution
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Distribution of Categorical Features by Heart Disease")

# Plot categorical features
categorical_features = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
for idx, feature in enumerate(categorical_features):
    row = idx // 3
    col = idx % 3
    if row < 2 and col < 3:  # Check if we have a valid subplot position
        sns.countplot(data=dataset, x=feature, hue="HeartDisease", ax=axes[row, col])
        axes[row, col].set_title(feature)
        axes[row, col].tick_params(axis="x", rotation=45)

# Remove empty subplot
if len(categorical_features) < 6:
    axes[1, 2].remove()

plt.tight_layout()
plt.show()

# Create correlation matrix for numerical features
numerical_features = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]
correlation_matrix = dataset[numerical_features + ["HeartDisease"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features", pad=20)
plt.tight_layout()
plt.show()


# Preprocess the data
def preprocess_data(df):
    # Create a copy of the dataframe
    df_processed = df.copy()

    # Encode categorical variables
    le = LabelEncoder()
    categorical_features = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]
    for feature in categorical_features:
        df_processed[feature] = le.fit_transform(df_processed[feature])

    # Separate features and target
    X = df_processed.drop("HeartDisease", axis=1)
    y = df_processed["HeartDisease"]

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
    ]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X, y


# Preprocess the data
X, y = preprocess_data(dataset)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print results
print("\nRandom Forest Model Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x="importance", y="feature")
plt.title("Feature Importance in Heart Disease Prediction", pad=20)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix", pad=20)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()


def encode_patient_data(patient_data, le_dict, scaler):
    patient_df = pd.DataFrame([patient_data])

    # Encode categorical features
    categorical_features = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]
    for feature in categorical_features:
        patient_df[feature] = le_dict[feature].transform([patient_data[feature]])

    # Scale numerical features
    numerical_features = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
    ]
    patient_df[numerical_features] = scaler.transform(patient_df[numerical_features])

    return patient_df


def predict_heart_disease(
    patient_data,
    model,
    le_dict,
    scaler,
):
    # Convert patient data to DataFrame
    patient_df = encode_patient_data(patient_data, le_dict, scaler)

    # Make prediction
    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    return {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "probability": probability,
    }


if __name__ == "__main__":
    # Create and store label encoders for each categorical feature
    le_dict = {}
    categorical_features = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]
    for feature in categorical_features:
        le_dict[feature] = LabelEncoder()
        le_dict[feature].fit(dataset[feature])

    # Create and fit the scaler
    scaler = StandardScaler()
    numerical_features = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
    ]
    scaler.fit(dataset[numerical_features])

    # Preprocess the complete dataset and train the model
    X, y = preprocess_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Sample patient data for testing
    sample_patients = [
        {
            "Age": 65,
            "Sex": "M",
            "ChestPainType": "ASY",
            "RestingBP": 140,
            "Cholesterol": 289,
            "FastingBS": 1,
            "RestingECG": "Normal",
            "MaxHR": 140,
            "ExerciseAngina": "Y",
            "Oldpeak": 1.5,
            "ST_Slope": "Flat",
        },
        {
            "Age": 45,
            "Sex": "F",
            "ChestPainType": "NAP",
            "RestingBP": 120,
            "Cholesterol": 180,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 160,
            "ExerciseAngina": "N",
            "Oldpeak": 0.0,
            "ST_Slope": "Up",
        },
        {
            "Age": 52,
            "Sex": "M",
            "ChestPainType": "ATA",
            "RestingBP": 135,
            "Cholesterol": 250,
            "FastingBS": 0,
            "RestingECG": "ST",
            "MaxHR": 150,
            "ExerciseAngina": "N",
            "Oldpeak": 0.8,
            "ST_Slope": "Up",
        },
    ]

    # Make predictions for sample patients
    print("\nHeart Disease Prediction Results for Sample Patients:")
    print("-" * 50)

    for i, patient_data in enumerate(sample_patients, 1):
        result = predict_heart_disease(patient_data, rf_model, le_dict, scaler)

        print(f"\nPatient {i}:")
        print("Patient Details:")
        for key, value in patient_data.items():
            print(f"{key}: {value}")
        print("\nPrediction Results:")
        print(f"Risk Level: {result['prediction']}")
        print(f"Probability of Heart Disease: {result['probability']:.2%}")
        print("-" * 50)
