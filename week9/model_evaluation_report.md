# House Price Prediction — Model Evaluation Report

## Project goal
Use house attributes to estimate the listed `Price`. This is a learning exercise, not a property appraisal.

## Data and preparation
- Source: `house_data.csv`; 300 usable records and 6 predictor columns.
- Target: `Price`. `Property_ID` is an identifier and is excluded.
- Predictors: Area, Bedrooms, Bathrooms, Age, Location, Property_Type. Numeric fields are median-imputed and scaled; categorical fields are mode-imputed and one-hot encoded.
- Split: 80% training (240 rows) and 20% test (60 rows), random_state=42. Preprocessing is learned inside each model pipeline from training data only.

## Test-set results
Best RMSE among the two models compared: **Random Forest**.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Random Forest | 1,596,584.66 | 4,693,364,907,084.39 | 2,166,417.53 | 0.967 |
| Linear Regression | 2,188,736.34 | 8,454,330,868,276.76 | 2,907,633.21 | 0.941 |

MAE is the average absolute error. MSE and RMSE give more weight to large errors. R² compares the predictions with a simple test-set mean baseline. The dataset does not identify currency, so errors are in source price units.

## Interpretation
Use `predictions_vs_actual.png` to see how closely predictions follow actual values. The residual plot and feature-importance chart provide additional diagnostics. Permutation importance is a model-specific association, not proof a feature causes prices to change.

## Limitations
There are 300 rows, no sale date, condition score, or detailed location. The random holdout may not represent future markets. No external validation was available. Check recent comparable sales and test errors by location before practical use.

## Reproduce
Install `requirements.txt`, then run `python train_model.py`. The metrics, predictions, and diagnostic charts are regenerated. The notebook also explains each step over seven learning days.
