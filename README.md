# Used Car Price Prediction

Supervised machine-learning regression project for predicting used-car prices from vehicle and market characteristics.

## Workflow
EDA → preprocessing → train/test split → Linear Regression baseline → Random Forest → diagnostics → 5-fold cross-validation → hyperparameter tuning → final holdout evaluation.

## Dataset
The project uses the `Car_Data` sheet from `Used_Car_Price_Regression_Dataset.xlsx`.

Target: `Price_USD`

Predictors: `Brand_Tier`, `City_Market`, `Car_Age_Years`, `Mileage_KM`, `Engine_Size_CC`, `Horsepower_HP`, `Fuel_Type`, `Transmission`, `Previous_Owners`, `Accident_History`, `Service_History_Score`.

`Car_ID` is excluded because it is an identifier.

## Results

| Model | Test R² | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.8973 | $2,719.12 | $4,163.21 |
| Random Forest | 0.9624 | $1,914.74 | $2,517.87 |
| Tuned Random Forest | 0.9647 | $1,883.02 | $2,440.04 |

Baseline RF 5-fold CV: mean R² 0.9583 (std 0.0055).

Tuned RF 5-fold CV: mean R² 0.9595, mean MAE $1,894.28, mean RMSE $2,410.20.

Best tuned configuration:
- n_estimators=300
- max_depth=20
- max_features=0.7
- min_samples_leaf=1
- min_samples_split=2

## Structure

```
used-car-price-prediction/
├── data/
├── notebooks/
├── src/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

See `reports/Used_Car_Price_Prediction_Final_Report.pdf` for the full technical report.
