# Preprocessing Report --- Customer Churn Prediction

## 1. Project Overview

**Goal:** Preprocess customer data, engineer features, and build a
machine-learning model that predicts which customers are likely to leave
(churn).

**Dataset:** `churn_data.csv` --- 500 customers and 9 columns.

  -----------------------------------------------------------------------
  Column                  Data type               Description
  ----------------------- ----------------------- -----------------------
  `CustomerID`            ID                      Dropped because it has
                                                  no predictive value.

  `Tenure`                Numeric                 Number of months with
                                                  the company (1--71).

  `MonthlyCharges`        Numeric                 Current monthly bill
                                                  (20--199).

  `TotalCharges`          Numeric                 Total billed to date
                                                  (159--7,992).

  `Contract`              Categorical (ordered)   Month-to-month, One
                                                  year, or Two year.

  `PaymentMethod`         Categorical (nominal)   Bank Transfer, Credit
                                                  Card, or Electronic
                                                  Check.

  `PaperlessBilling`      Binary                  Yes / No.

  `SeniorCitizen`         Binary                  0 / 1.

  `Churn`                 Target                  1 = customer left; 0 =
                                                  customer stayed.
  -----------------------------------------------------------------------

## 2. Data Exploration 

-   No missing values were found.
-   No duplicate customer IDs were found.
-   **Class imbalance:** 53 of 500 customers churned (10.6%).
    -   Used a stratified train/test split.
    -   Used stratified 5-fold cross-validation.
    -   Used `class_weight="balanced"` to account for class imbalance.
-   **Strongest signal --- tenure:** Every churner has a tenure of 11
    months or less.
    -   The 0--1 year group has a churn rate of 63%.
    -   All other tenure groups have a churn rate of 0%.
-   **Contract type and churn:**

  Contract type      Churn rate
  ---------------- ------------
  Month-to-month          20.6%
  One year                 4.3%
  Two year                 6.9%

-   `TotalCharges` is not equal to `Tenure × MonthlyCharges`; therefore,
    the ratio between these values may contain useful information.

## 3. Categorical Encoding 

Four categorical encoding methods were explored.

  -----------------------------------------------------------------------
  Encoding method         Applied to              Rationale
  ----------------------- ----------------------- -----------------------
  Label encoding          `PaperlessBilling`      Binary variable;
  (`LabelEncoder`)                                represented as a single
                                                  0/1 column.

  Ordinal encoding        `Contract`              Has a natural order of
  (`OrdinalEncoder`)                              commitment:
                                                  Month-to-month = 0, One
                                                  year = 1, Two year = 2.

  One-hot encoding        `PaymentMethod`,        Nominal categories have
  (`OneHotEncoder`)       `TenureGroup`           no natural order, so
                                                  one-hot encoding avoids
                                                  implying a false
                                                  ranking. `drop="first"`
                                                  was used in the
                                                  analysis to avoid the
                                                  dummy-variable trap.

  Frequency encoding      `PaymentMethod`         Compact, single-column
                          (exploratory)           alternative for
                                                  categorical variables
                                                  with many unique
                                                  values.
  -----------------------------------------------------------------------

**Pipeline configuration:** `OneHotEncoder(handle_unknown="ignore")`
allows the pipeline to handle categories that were not present during
training.

## 4. Feature Scaling 

Two scaling methods were evaluated.

  -----------------------------------------------------------------------
  Scaler                  Output                  When to use
  ----------------------- ----------------------- -----------------------
  Min-Max scaling         Rescales values to the  Useful for bounded
                          range \[0, 1\].         inputs, neural
                                                  networks, and
                                                  distance-based models.
                                                  Sensitive to outliers.

  Standard scaling        Transforms values to    Common choice for
                          have a mean of 0 and a  linear and logistic
                          standard deviation of   regression models.
                          1.                      
  -----------------------------------------------------------------------

Logistic-regression cross-validation ROC-AUC was compared using raw
numeric features, no scaling, Min-Max scaling, and Standard scaling
(refer to the notebook output).

**Final choice:** `StandardScaler` is used in the final pipeline.
Tree-based models are generally scale-invariant, but scaling was
retained to make the preprocessing pipeline more model-agnostic.

## 5. Outlier Detection and Handling 

Outliers were evaluated using the Interquartile Range (IQR) method and
the z-score method.

  ----------------------------------------------------------------------------------
  Feature                    Q1         Q3 IQR             IQR Count with    Maximum
                                           bounds     outliers \|z\| \> 3      \|z\|
  ------------------ ---------- ---------- -------- ---------- ---------- ----------
  `Tenure`                   19         54 −33.5 to          0          0       1.72
                                           106.5                          

  `MonthlyCharges`           67        158 −69.5 to          0          0       1.81
                                           294.5                          

  `TotalCharges`          2,237      6,267 −3,807            0          0       1.81
                                           to                             
                                           12,311                         
  ----------------------------------------------------------------------------------

**Finding:** No outliers were detected by either method in the dataset.

**Handling strategy:** A custom `IQRCapper` transformer is retained in
the pipeline. It winsorizes values at the limits:

-   Lower bound: `Q1 − 1.5 × IQR`
-   Upper bound: `Q3 + 1.5 × IQR`

The transformer learns these limits from the training data only and
applies them to future data. For example, an unusually high
`MonthlyCharges` value of 999 would be capped at 294.5 based on the
reported training-data bounds.

Skewed ratio features are transformed using `log1p` before scaling.

## 6. Feature Selection 

Four feature-selection methods were combined using an average-rank
approach:

1.  Correlation with the target.
2.  Mutual information.
3.  Random Forest feature importance.
4.  ANOVA F-test using `SelectKBest`.

### Top 10 features by average rank

    Rank Feature
  ------ ---------------------
       1 `IsNewCustomer`
       2 `Tenure`
       3 `ChargePerTenure`
       4 `CLV`
       5 `AvgMonthlySpend`
       6 `PaymentEfficiency`
       7 `ContractToTenure`
       8 `TenureGroup_4+yr`
       9 `TenureGroup_2-4yr`
      10 `Contract`

### Multicollinearity check

Features with an absolute correlation greater than 0.85 were reviewed.

  -------------------------------------------------------------------------------------
  Feature pair                                        Correlation Observation / action
  ---------------------------------- ---------------------------- ---------------------
  `AutoPay` and                                              1.00 Duplicate
  `PaymentMethod_Electronic Check`                                information; keep
                                                                  only one in a linear
                                                                  model.

  `MonthlyCharges` and                                       0.87 Strong correlation;
  `HighMonthlyCharge`                                             review for
                                                                  redundancy.
  -------------------------------------------------------------------------------------

### Low-value features

The following features showed near-zero correlation and low feature
importance:

-   `SeniorCitizen`
-   `PaperlessBilling`
-   `PaymentMethod`
-   `TotalCharges`

## 7. Final Preprocessing and Modeling Pipeline 

The final pipeline starts with the raw input data, engineers additional
features, applies column-specific preprocessing, and then trains a
class-balanced Random Forest model.

``` text
Raw data (7 columns)
    |
    v
FeatureEngineer
    |  Adds 9 engineered features
    v
ColumnTransformer
    |
    |-- Raw numeric features
    |      IQRCapper -> StandardScaler
    |
    |-- Ratio numeric features
    |      log1p -> StandardScaler
    |
    |-- Binary flags
    |      Passthrough
    |
    |-- Contract
    |      OrdinalEncoder
    |
    |-- PaymentMethod and TenureGroup
    |      OneHotEncoder(handle_unknown="ignore")
    |
    |-- PaperlessBilling
    |      OrdinalEncoder (No=0, Yes=1)
    |
    v
Random Forest
(class_weight="balanced")
    |
    v
Prediction
```

**Pipeline output:** 21 model-ready features.

**Saved pipeline:** `churn_pipeline.joblib`

## 8. Model Results

### 5-fold cross-validation results on the training set

  Model                   Mean ROC-AUC   Standard deviation
  --------------------- -------------- --------------------
  Logistic Regression            0.987                0.002
  Random Forest              **0.992**            **0.005**

### Held-out test-set results

The test set contains 125 customers, including 13 churners.

  Metric              Result
  ----------------- --------
  Accuracy              0.97
  Churn recall          1.00
  Churn precision       0.76
  ROC-AUC               1.00

**Interpretation:** The scores are unusually high because churn in this
dataset follows an almost deterministic tenure pattern: all churners
have a tenure of 11 months or less. Performance on real-world customer
data is likely to be lower.

## 9. Testing and Validation

All automated checks in the notebook passed.

  ----------------------------------------------------------------------------------
                           No. Test                            Expected / observed
                                                               result
  ---------------------------- ------------------------------- ---------------------
                             1 Check pipeline output for       No NaNs; original row
                               missing values and row-count    count preserved.
                               changes.                        

                             2 Test an unseen category         Unseen category
                               (`PaymentMethod = "Crypto"`).   handled by the
                                                               encoder.

                             3 Test an extreme value           Value capped;
                               (`MonthlyCharges = 10000`).     prediction returned
                                                               successfully.

                             4 Test the edge case              No division-by-zero
                               `Tenure = 0`.                   error.

                             5 Check predicted probabilities.  All probabilities are
                                                               within \[0, 1\].

                             6 Perform a prediction sanity     A 2-month,
                               check.                          month-to-month
                                                               customer paying
                                                               180/month receives a
                                                               96% churn
                                                               probability; a
                                                               60-month,
                                                               two-year-contract
                                                               customer receives 0%.

                             7 Save and reload the pipeline.   Reloaded pipeline
                                                               produces identical
                                                               predictions.
  ----------------------------------------------------------------------------------

## 10. Conclusion

The project completed the core preprocessing and modeling workflow for
customer churn prediction. It included data exploration, class-imbalance
handling, categorical encoding, feature scaling, outlier protection,
feature engineering, feature selection, and model validation.

The Random Forest model achieved a mean 5-fold cross-validation ROC-AUC
of **0.992 ± 0.005** and a held-out test ROC-AUC of **1.00**. These
results should be interpreted in the context of the dataset's unusually
strong relationship between tenure and churn. Further validation on a
larger, more representative dataset is recommended before using the
model in a production setting.
