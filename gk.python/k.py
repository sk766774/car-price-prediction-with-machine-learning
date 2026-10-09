
# 1. Load dataset
df = pd.read_csv("car_data.csv")

# 2. Data cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Convert columns into numeric values
for col in ["Year", "Mileage", "Horsepower", "Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0]

# 3. Feature engineering
df["Car_Age"] = 2026 - df["Year"]
df = df[df["Car_Age"] >= 0]

# Features and target
X = df[["Brand", "Car_Age", "Mileage", "Horsepower"]]
y = df["Price"]

# 4. Preprocessing
numeric_features = ["Car_Age", "Mileage", "Horsepower"]
categorical_features = ["Brand"]

numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Model evaluation
predictions = model.predict(X_test)


print("Model Evaluation")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# 8. Visualize actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price: Actual vs Predicted")
plt.tight_layout()
plt.show()

# 9. Predict price for a new car
new_car = pd.DataFrame([{
    "Brand": "Maruti",
    "Car_Age": 3,
    "Mileage": 22,
    "Horsepower": 75
}])

predicted_price = model.predict(new_car)[0]

print("Estimated Car Price: ₹", round(predicted_price, 2))