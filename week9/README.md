# Week 9 House Price Prediction

Open this folder in VS Code. Install dependencies with `python -m pip install -r requirements.txt`. Follow `house_price_prediction.ipynb` from Day 1 through Day 7, or run `python train_model.py` to regenerate the metrics, predictions, report, and charts.

The 300-row dataset uses `Price` as the target. Models use an 80/20 train/test split. The notebook compares linear regression and random forest, and calculates MAE, MSE, RMSE, and R². `Property_ID` is excluded as an identifier. Currency is not specified; predictions are in dataset price units and are for learning, not appraisal.
