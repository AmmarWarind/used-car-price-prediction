import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/Used_Car_Price_Regression_Dataset.xlsx"
RANDOM_STATE = 42

categorical_features = [
    "Brand_Tier", "City_Market", "Fuel_Type",
    "Transmission", "Accident_History"
]
numerical_features = [
    "Car_Age_Years", "Mileage_KM", "Engine_Size_CC",
    "Horsepower_HP", "Previous_Owners", "Service_History_Score"
]

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
    }

def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Car_Data")
    X = df.drop(columns=["Car_ID", "Price_USD"])
    y = df["Price_USD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    scaled = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ])

    tree = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ])

    linear = Pipeline([
        ("preprocessor", scaled),
        ("model", LinearRegression())
    ])
    linear.fit(X_train, y_train)
    print("Linear Regression:", evaluate(linear, X_test, y_test))

    rf = Pipeline([
        ("preprocessor", tree),
        ("model", RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    print("Baseline Random Forest:", evaluate(rf, X_test, y_test))

    cv = cross_validate(
        rf, X_train, y_train, cv=5,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        n_jobs=-1
    )
    print("Baseline CV R2:", cv["test_r2"].mean())

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.7, 1.0]
    }

    search = GridSearchCV(
        rf, param_grid=param_grid, cv=5,
        scoring="r2", n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print("Best CV R2:", search.best_score_)

    tuned = search.best_estimator_
    tuned_cv = cross_validate(
        tuned, X_train, y_train, cv=5,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        n_jobs=-1
    )
    print("Tuned mean CV R2:", tuned_cv["test_r2"].mean())
    print("Final tuned holdout:", evaluate(tuned, X_test, y_test))

if __name__ == "__main__":
    main()
